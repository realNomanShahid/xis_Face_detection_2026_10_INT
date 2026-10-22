#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_retinaface_onnx.py
=========================

  pip install torch torchvision        (the CPU version is enough, this runs once)
  python3 pytorch_to_onnx_retinaface.py --weights Resnet50_Final.pth
  or
  python3 pytorch_to_onnx_retinaface.py --weights Resnet50_Final.pth --width 1280 --height 736       (Recommended)


Result: retinaface_r50_640x640.onnx
  input  "input" : 1 x 3 x H x W   BGR image, mean (104, 117, 123) subtracted, no scaling
  output "loc"   : 1 x N x 4       box offsets
  output "conf"  : 1 x N x 2       softmax scores [background, face]
  N = number of anchors for that size (640x640 -> 16800)

Notes
  * Width and height must be multiples of 32 and are FIXED in the ONNX file.
    Pick the size you want to run at, for example:
        --width 640  --height 640    default, square
        --width 640  --height 384    16:9 video, about 40% less work than 640x640
        --width 1280 --height 736    16:9 720p video at full size (small faces), heavier
  * The model code comes from github.com/biubug6/Pytorch_Retinaface; it is cloned
    automatically next to this script if it is not there yet (needs git + internet).
"""
import argparse
import importlib.util
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="RetinaFace-ResNet50 .pth -> .onnx",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", required=True, help="Resnet50_Final.pth")
    ap.add_argument("--output", default="", help="default: retinaface_r50_<W>x<H>.onnx")
    ap.add_argument("--width", type=int, default=640, help="network input width (multiple of 32)")
    ap.add_argument("--height", type=int, default=640, help="network input height (multiple of 32)")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--repo", default="Pytorch_Retinaface", help="folder with the RetinaFace code")
    a = ap.parse_args()

    if a.width % 32 or a.height % 32:
        sys.exit("--width and --height must be multiples of 32")
    if not os.path.exists(a.weights):
        sys.exit(f"Weights file not found: {a.weights}")
    out = a.output or f"retinaface_r50_{a.width}x{a.height}.onnx"

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        sys.exit("PyTorch is not installed. Run:  pip install torch torchvision")

    # ---------------------------------------------------------------- 1) model code
    if not os.path.isdir(a.repo):
        print("Cloning the RetinaFace code ...")
        subprocess.run(["git", "clone", "-q", "https://github.com/biubug6/Pytorch_Retinaface.git", a.repo],
                       check=True)
    # newer torchvision no longer accepts 'pretrained=' and we do not need the ImageNet download
    p = os.path.join(a.repo, "models", "retinaface.py")
    src = open(p).read()
    patched = src.replace("models.resnet50(pretrained=cfg['pretrain'])", "models.resnet50(weights=None)")
    if patched != src:
        open(p, "w").write(patched)

    sys.path.insert(0, os.path.abspath(a.repo))
    spec = importlib.util.spec_from_file_location("rf_config", os.path.join(a.repo, "data", "config.py"))
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    cfg = dict(cfg_mod.cfg_re50)
    cfg["pretrain"] = False

    from models.retinaface import RetinaFace
    import models.net as repo_net

    # With a fixed size that is a multiple of 32, "upsample-x2" is exactly what the original
    # code computes from tensor sizes. A constant scale exports to a simpler, TensorRT-friendly ONNX.
    def fpn_forward(self, inp):
        inp = list(inp.values())
        o1 = self.output1(inp[0])
        o2 = self.output2(inp[1])
        o3 = self.output3(inp[2])
        o2 = self.merge2(o2 + F.interpolate(o3, scale_factor=2, mode="nearest"))
        o1 = self.merge1(o1 + F.interpolate(o2, scale_factor=2, mode="nearest"))
        return [o1, o2, o3]

    repo_net.FPN.forward = fpn_forward

    # ---------------------------------------------------------------- 2) weights
    try:
        state = torch.load(a.weights, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  (older file format)
        state = torch.load(a.weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    net = RetinaFace(cfg=cfg, phase="test")
    res = net.load_state_dict(state, strict=False)
    n_params = len(net.state_dict())
    print(f"Weights loaded: {n_params - len(res.missing_keys)}/{n_params} tensors matched "
          f"(missing {len(res.missing_keys)}, unexpected {len(res.unexpected_keys)})")
    if len(res.missing_keys) > 0.1 * n_params:
        sys.exit("Most weights did not match. Is this the Resnet50 file (Resnet50_Final.pth), "
                 "not the MobileNet one?")
    net.eval()

    class Wrap(torch.nn.Module):
        """Keep only box offsets and face scores (the 5 landmarks are not needed for blurring)."""

        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            loc, conf, _ = self.n(x)
            return loc, conf

    model = Wrap(net).eval()
    dummy = torch.randn(1, 3, a.height, a.width)

    # ---------------------------------------------------------------- 3) export
    kwargs = dict(input_names=["input"], output_names=["loc", "conf"],
                  opset_version=a.opset, do_constant_folding=True)
    done = False
    with torch.no_grad():
        for extra in ({"dynamo": False}, {}):
            try:
                torch.onnx.export(model, dummy, out, **kwargs, **extra)
                done = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"[note] export attempt {extra or 'default'} failed: {str(e)[:200]}")
    if not done:
        sys.exit("ONNX export failed (see the messages above).")
    print(f"Saved {out}  ({os.path.getsize(out) / 1e6:.0f} MB)")

    # ---------------------------------------------------------------- 4) self-check
    n_expected = sum(-(-a.height // s) * -(-a.width // s) * 2 for s in (8, 16, 32))
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        x = torch.randn(1, 3, a.height, a.width)
        with torch.no_grad():
            ref_loc, ref_conf = model(x)
        o_loc, o_conf = sess.run(None, {"input": x.numpy()})
        print("ONNX outputs:", [(o.name, o.shape) for o in sess.get_outputs()],
              f"| expected anchors N = {n_expected}")
        print("max difference vs PyTorch:  loc %.2e   conf %.2e"
              % (np.abs(o_loc - ref_loc.numpy()).max(), np.abs(o_conf - ref_conf.numpy()).max()))
    except ImportError:
        print("(onnxruntime not installed, skipped the self-check. Optional: pip install onnxruntime)")

    print("\nNext step, run the pipeline (DeepStream builds the TensorRT engine on the first run):")
    print(f"  python3 face_blur_deepstream_retinaface.py --source video.mp4 \\\n"
          f"      --onnx {out} --engine {out}_b1_gpu0_fp16.engine \\\n"
          f"      --det-width {a.width} --det-height {a.height}")


if __name__ == "__main__":
    main()