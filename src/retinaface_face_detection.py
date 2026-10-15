#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_blur_deepstream_retinaface.py
==================================
Face detection (RetinaFace ResNet-50, ONNX -> TensorRT) + GPU face blur for live
streams or video files, using NVIDIA DeepStream for decode, batching, inference and encode.

Setup (once)
------------
  * x86 Linux + NVIDIA GPU + DeepStream 7.x or 8.x with the Python bindings (pyds).
    (Jetson is NOT supported: pyds.get_nvds_buf_surface_gpu is x86-only.)
  * pip install numpy cupy-cuda12x opencv-python-headless nvidia-ml-py requests

Run examples
------------
  # real IP camera (RTSP over TCP) with the live window
  python3 face_blur_deepstream_retinaface.py --source rtsp://user:pass@192.168.1.20:554/stream1 \
          --onnx retinaface_r50_1280x736.onnx --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
          --width 1280 --height 720 --conf 0.4 --preview

  # Rutube page link with the live window
  python3 face_blur_deepstream_retinaface.py --rutube-page https://rutube.ru/live/video/<id>/ \
          --onnx retinaface_r50_1280x736.onnx --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
          --width 1280 --height 720 --max-height 720 --conf 0.4 --out-fps 13 --preview --max-seconds 60

  # video file
  python3 face_blur_deepstream_retinaface.py --source clip.mp4 \
          --onnx retinaface_r50_1280x736.onnx --conf 0.4 --preview

Output files
------------
  output_blurred.mkv      the blurred video (use --output name.mp4 for MP4)
  metrics.json            performance report (+ a privacy-audit summary)
  detection_logs.jsonl    one line per frame: frame index, detection count, confidence
                          summary (min/mean/max), timing
  privacy_audit.jsonl     one line per frame: blur operation status, detected face count,
                          output safety status
  The two .jsonl logs are written next to metrics.json (or in --log-dir). --no-logs turns them off.
  sample_frames/          ONLY created when you ask for it: --save-samples (every 100 frames)
                          or --sample-every N. Without these flags nothing is stored.
  detected_instances/     EVERY frame where a face was detected and blurred .
                          On by default; disable with --no-save-detections.

Press Ctrl+C at any time: the script finalises the video file and prints the report.
"""
import argparse
import ctypes
import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np

try:
    import cupy as cp
    import cv2
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
    import pyds
except ImportError as _e:  # pragma: no cover
    sys.exit(f"Missing Python package: {_e}\nSee the 'Setup' section at the top of this file.")


# ----------------------------------------------------------------------------------
# YAML configs (configs/*.yaml via src/config_loader.py) — defaults live here, not in call sites
# ----------------------------------------------------------------------------------
def load_pipeline_cfg():
    """Load all YAML sections. Returns {} if configs/ or PyYAML is missing."""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.config_loader import load_all
        cfg = load_all()
        print(f"[config] loaded YAML from {root / 'configs'}")
        return cfg
    except Exception as e:  # noqa: BLE001
        print(f"[config] YAML not loaded ({e}); using CLI / built-in defaults only")
        return {}


def cfg_get(cfg, section, key, default=None):
    sec = cfg.get(section) if isinstance(cfg, dict) else None
    if not isinstance(sec, dict):
        return default
    val = sec.get(key, default)
    return default if val is None else val


def build_rtsp_from_camera(cam: dict) -> str | None:
    """Build rtsp:// URI from configs/camera.yaml (password URL-encoded)."""
    if not cam:
        return None
    if cam.get("rtsp_url"):
        return str(cam["rtsp_url"])
    ip = cam.get("ip")
    path = cam.get("rtsp_path") or "/"
    if not ip:
        return None
    port = int(cam.get("rtsp_port") or 554)
    user = cam.get("onvif_user") or ""
    password = cam.get("onvif_password")
    env_name = cam.get("onvif_password_env") or "CAM_ONVIF_PASSWORD"
    if not password:
        password = os.environ.get(env_name) or ""
    userinfo = ""
    if user:
        userinfo = quote(str(user), safe="")
        if password:
            userinfo += ":" + quote(str(password), safe="")
        userinfo += "@"
    if not path.startswith("/"):
        path = "/" + path
    return f"rtsp://{userinfo}{ip}:{port}{path}"


# ----------------------------------------------------------------------------------
# CUDA kernels: separable box blur inside an ellipse, all faces of a frame in 2 launches
# ----------------------------------------------------------------------------------
CUDA_SRC = r"""
extern "C" {

// pass 1: horizontal average of the original frame -> tmp   (inside every face box)
__global__ void blur_h(const unsigned char* frame, unsigned char* tmp,
                       const int* boxes, int pitch) {
    const int f = blockIdx.z;
    const int* b = boxes + f * 5;
    const int x1 = b[0], y1 = b[1], x2 = b[2], y2 = b[3], r = b[4];
    const int x = x1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= x2 || y >= y2) return;

    const int n = 2 * r + 1;
    int s0 = 0, s1 = 0, s2 = 0;
    const unsigned char* row = frame + (size_t)y * pitch;
    for (int d = -r; d <= r; ++d) {
        int xx = x + d;
        xx = xx < x1 ? x1 : (xx > x2 - 1 ? x2 - 1 : xx);
        const unsigned char* p = row + xx * 4;
        s0 += p[0]; s1 += p[1]; s2 += p[2];
    }
    unsigned char* q = tmp + (size_t)y * pitch + x * 4;
    q[0] = (unsigned char)((s0 + n / 2) / n);
    q[1] = (unsigned char)((s1 + n / 2) / n);
    q[2] = (unsigned char)((s2 + n / 2) / n);
}

// pass 2: vertical average of tmp -> frame, only inside the face ellipse (alpha untouched)
__global__ void blur_v(unsigned char* frame, const unsigned char* tmp,
                       const int* boxes, int pitch) {
    const int f = blockIdx.z;
    const int* b = boxes + f * 5;
    const int x1 = b[0], y1 = b[1], x2 = b[2], y2 = b[3], r = b[4];
    const int x = x1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= x2 || y >= y2) return;

    const float cx = 0.5f * (float)(x1 + x2 - 1), cy = 0.5f * (float)(y1 + y2 - 1);
    const float ax = 0.5f * (float)(x2 - x1),     ay = 0.5f * (float)(y2 - y1);
    const float ex = ((float)x - cx) / ax, ey = ((float)y - cy) / ay;
    if (ex * ex + ey * ey > 1.0f) return;

    const int n = 2 * r + 1;
    int s0 = 0, s1 = 0, s2 = 0;
    for (int d = -r; d <= r; ++d) {
        int yy = y + d;
        yy = yy < y1 ? y1 : (yy > y2 - 1 ? y2 - 1 : yy);
        const unsigned char* p = tmp + (size_t)yy * pitch + x * 4;
        s0 += p[0]; s1 += p[1]; s2 += p[2];
    }
    unsigned char* q = frame + (size_t)y * pitch + x * 4;
    q[0] = (unsigned char)((s0 + n / 2) / n);
    q[1] = (unsigned char)((s1 + n / 2) / n);
    q[2] = (unsigned char)((s2 + n / 2) / n);
}

}
"""


# ----------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------
def setp(el, name, value):
    """set_property that never kills the script if a property differs between versions."""
    try:
        el.set_property(name, value)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [note] {el.get_name()}: property '{name}' not set ({str(e)[:90]})")
        return False


def to_uri(src):
    if "://" in src:
        return src
    path = os.path.abspath(src)
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    return "file://" + path


def resolve_rutube(page_url, max_h):
    """Rutube page link -> HLS (.m3u8) link, choosing the best quality up to max_h."""
    from urllib.parse import urljoin

    import requests

    hdr = {"User-Agent": "Mozilla/5.0", "Referer": "https://rutube.ru/"}
    m = re.search(r"[0-9a-f]{32}", page_url)
    if not m:
        sys.exit("No Rutube video id found in --rutube-page")
    vid = m.group(0)
    r = requests.get(
        f"https://rutube.ru/api/play/options/{vid}/",
        params={"no_404": "true", "referer": f"https://rutube.ru/video/{vid}/", "pver": "v2"},
        headers=hdr, timeout=20)
    found = []

    def walk(o):
        if isinstance(o, str):
            if o.startswith("http") and ".m3u8" in o:
                found.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(r.json())
    if not found:
        sys.exit("Rutube gave no HLS link (region block or API change). "
                 "Get a direct link another way and use --source.")
    url = found[0]
    txt = requests.get(url, headers=hdr, timeout=20).text
    if "#EXT-X-STREAM-INF" in txt:
        lines, variants = txt.splitlines(), []
        for i, ln in enumerate(lines):
            if ln.startswith("#EXT-X-STREAM-INF"):
                mm = re.search(r"RESOLUTION=\d+x(\d+)", ln)
                bb = re.search(r"BANDWIDTH=(\d+)", ln)
                j = i + 1
                while j < len(lines) and lines[j].startswith("#"):
                    j += 1
                if j < len(lines):
                    variants.append((int(mm.group(1)) if mm else 0,
                                     int(bb.group(1)) if bb else 0,
                                     urljoin(url, lines[j].strip())))
        if variants:
            ok = [v for v in variants if 0 < v[0] <= max_h] or variants
            best = max(ok, key=lambda v: (v[0], v[1]))
            print(f"Rutube heights available: {sorted({v[0] for v in variants})} -> using {best[0]}")
            url = best[2]
    return url


def guess_model_name(*paths):
    """'retinaface_r50_640x640.onnx' -> 'RetinaFace-R50'"""
    for pth in paths:
        base = os.path.basename(pth or "").lower()
        m = re.search(r"retinaface[_-]?(r50|r18|mnet|mobile\w*|resnet\d+)?", base)
        if m:
            tag = (m.group(1) or "").upper()
            return f"RetinaFace-{tag}" if tag else "RetinaFace"
    return "RetinaFace"


def nms_np(boxes, scores, thr):
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= thr]
    return np.array(keep, dtype=np.int64)


# ----------------------------------------------------------------------------------
# RetinaFace output decoding (numpy, tiny)
#
# Model outputs (biubug6 / Pytorch_Retinaface style export), each with N rows:
#     loc     (N, 4)   box regression  (dcx, dcy, dw, dh)
#     conf    (N, 2)   [background, face]  (logits or already softmaxed)
#     landms  (N, 10)  5 landmarks (not needed for blurring, ignored)
# Priors (anchors) per stride 8 / 16 / 32 with min sizes (16,32) / (64,128) / (256,512).
# Variances (0.1, 0.2).
# ----------------------------------------------------------------------------------
class RetinaFaceDecoder:
    STEPS = (8, 16, 32)
    MIN_SIZES = ((16, 32), (64, 128), (256, 512))
    VARIANCES = (0.1, 0.2)

    def __init__(self, in_w, in_h, conf, nms_iou):
        self.in_w, self.in_h = in_w, in_h
        self.conf, self.nms_iou = conf, nms_iou
        pri = []
        for step, mins in zip(self.STEPS, self.MIN_SIZES):
            fh, fw = int(np.ceil(in_h / step)), int(np.ceil(in_w / step))
            ys, xs = np.mgrid[:fh, :fw]                              # row i, column j
            cx = ((xs.reshape(-1) + 0.5) * step / in_w).astype(np.float32)
            cy = ((ys.reshape(-1) + 0.5) * step / in_h).astype(np.float32)
            k = len(mins)
            cx, cy = np.repeat(cx, k), np.repeat(cy, k)             # order: cell, then min_size
            m = np.array(mins, dtype=np.float32)
            pw = np.tile(m / in_w, fh * fw)
            ph = np.tile(m / in_h, fh * fw)
            pri.append(np.stack([cx, cy, pw, ph], axis=1))
        self.priors = np.concatenate(pri).astype(np.float32)       # (N, 4) cx, cy, w, h  (0..1)
        self.n_priors = len(self.priors)
        self.softmax = True

    def build_plan(self, outs):
        """outs: list of (name, dims, array). Returns {kind: layer_index}."""
        plan = {}
        for i, (_, dims, arr) in enumerate(outs):
            if len(dims) < 1:
                continue
            c = int(dims[-1])
            kind = {4: "loc", 2: "conf", 10: "landms"}.get(c)
            if kind and arr.size == c * self.n_priors:
                plan[kind] = i
        missing = [k for k in ("loc", "conf") if k not in plan]
        if missing:
            table = "\n".join(f"    {n}: dims={d} elements={a.size}" for n, d, a in outs)
            raise RuntimeError(
                "Could not match the engine outputs to RetinaFace (missing: %s).\n"
                "Engine outputs seen by nvinfer:\n%s\n"
                "Expected loc (N,4) and conf (N,2) with N = %d for a %dx%d input.\n"
                "Check --det-width / --det-height equal the ONNX input size."
                % (missing, table, self.n_priors, self.in_w, self.in_h))
        cf = outs[plan["conf"]][2].reshape(-1, 2)
        s = cf.sum(axis=1)
        already_prob = (cf.min() >= 0.0 and cf.max() <= 1.0 and float(np.abs(s - 1.0).mean()) < 1e-2)
        self.softmax = not already_prob
        print(f"RetinaFace conf output: {'logits -> softmax applied' if self.softmax else 'already probabilities'}"
              f" | priors: {self.n_priors} for {self.in_w}x{self.in_h}")
        return plan

    def decode(self, plan, outs, frame_w, frame_h):
        # nvinfer letterbox: one scale for both axes, padding on the right / bottom
        scale = min(self.in_w / frame_w, self.in_h / frame_h)
        loc = outs[plan["loc"]][2].reshape(-1, 4)
        cf = outs[plan["conf"]][2].reshape(-1, 2)
        if self.softmax:
            d = np.clip(cf[:, 0].astype(np.float64) - cf[:, 1], -50.0, 50.0)
            score = (1.0 / (1.0 + np.exp(d))).astype(np.float32)
        else:
            score = cf[:, 1].astype(np.float32)

        keep = np.flatnonzero(score >= self.conf)
        if keep.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        if keep.size > 1000:
            keep = keep[np.argsort(-score[keep])[:1000]]
        sc = score[keep]
        pri, l = self.priors[keep], loc[keep]

        v0, v1 = self.VARIANCES
        c = pri[:, :2] + l[:, :2] * v0 * pri[:, 2:]
        wh = pri[:, 2:] * np.exp(np.clip(l[:, 2:] * v1, -10.0, 10.0))
        norm = np.concatenate([c - wh / 2.0, c + wh / 2.0], axis=1)
        mul = np.array([self.in_w, self.in_h, self.in_w, self.in_h], dtype=np.float32)
        boxes = (norm * mul / scale).astype(np.float32)

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h)
        ok = ((boxes[:, 2] - boxes[:, 0]) >= 2) & ((boxes[:, 3] - boxes[:, 1]) >= 2)
        boxes, sc = boxes[ok], sc[ok]
        if len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        idx = nms_np(boxes, sc, self.nms_iou)
        return np.hstack([boxes[idx], sc[idx, None]]).astype(np.float32)


def make_rois(boxes, W, H, pad, blur_frac):
    """float boxes -> int32 [x1, y1, x2, y2, blur_radius] with padding, clipped to the frame."""
    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.int32)
    b = boxes[:, :4].astype(np.float64)
    bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    x1 = np.floor(np.clip(b[:, 0] - bw * pad, 0, W - 1)).astype(np.int32)
    y1 = np.floor(np.clip(b[:, 1] - bh * pad, 0, H - 1)).astype(np.int32)
    x2 = np.ceil(np.clip(b[:, 2] + bw * pad, 1, W)).astype(np.int32)
    y2 = np.ceil(np.clip(b[:, 3] + bh * pad, 1, H)).astype(np.int32)
    w, h = x2 - x1, y2 - y1
    r = np.clip((blur_frac * np.maximum(w, h)).astype(np.int32), 3, 60)
    rois = np.stack([x1, y1, x2, y2, r], axis=1).astype(np.int32)
    return rois[(w >= 2) & (h >= 2)]


# ----------------------------------------------------------------------------------
# metrics helpers
# ----------------------------------------------------------------------------------
class Stat:
    """Collects timings (seconds), skips the first `warm` samples, reports ms."""

    def __init__(self, warm):
        self.warm, self.n, self.v = warm, 0, []

    def add(self, sec):
        self.n += 1
        if self.n > self.warm:
            self.v.append(sec * 1000.0)

    def summary(self):
        if not self.v:
            return dict(mean=0.0, p50=0.0, p95=0.0, max=0.0, n=0)
        a = np.array(self.v)
        return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                    p95=float(np.percentile(a, 95)), max=float(a.max()), n=int(a.size))


class GpuMonitor(threading.Thread):
    def __init__(self, gpu_index, active_fn, period=0.5):
        super().__init__(daemon=True)
        self.active_fn, self.period = active_fn, period
        self.ok, self.name, self.total_mb = False, "n/a", 0
        self.util, self.mem, self.enc, self.dec = [], [], [], []
        self._halt = threading.Event()
        try:
            import pynvml as nv
            nv.nvmlInit()
            self.nv = nv
            self.h = nv.nvmlDeviceGetHandleByIndex(gpu_index)
            name = nv.nvmlDeviceGetName(self.h)
            self.name = name.decode() if isinstance(name, bytes) else str(name)
            self.total_mb = int(nv.nvmlDeviceGetMemoryInfo(self.h).total // 2 ** 20)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[note] NVML not available, GPU utilisation will not be reported ({e}). "
                  f"Install with: pip install nvidia-ml-py")

    def run(self):
        nv = getattr(self, "nv", None)
        while self.ok and not self._halt.is_set():
            try:
                self.mem.append(int(nv.nvmlDeviceGetMemoryInfo(self.h).used // 2 ** 20))
                if self.active_fn():
                    self.util.append(int(nv.nvmlDeviceGetUtilizationRates(self.h).gpu))
                    try:
                        self.enc.append(int(nv.nvmlDeviceGetEncoderUtilization(self.h)[0]))
                        self.dec.append(int(nv.nvmlDeviceGetDecoderUtilization(self.h)[0]))
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()


# ----------------------------------------------------------------------------------
# live preview window (a second, tiny GStreamer pipeline: appsrc -> videoconvert -> X window)
# It is fed from the probe, throttled, and can never block or crash the main pipeline.
# ----------------------------------------------------------------------------------
class Preview:
    def __init__(self, max_w, max_h, max_fps, draw_boxes):
        self.max_w, self.max_h = int(max_w), int(max_h)
        self.min_gap = 1.0 / max(float(max_fps), 1.0)
        self.draw_boxes = draw_boxes
        self.ok = True                  # False = preview switched off (window closed / error)
        self.pipe = None
        self.appsrc = None
        self.pw = self.ph = 0
        self.sx = self.sy = 1.0
        self.last = 0.0

    def due(self, now):
        if not self.ok or now - self.last < self.min_gap:
            return False
        self.last = now
        return True

    def _start(self, W, H):
        sc = min(self.max_w / float(W), self.max_h / float(H), 1.0)   # fit inside, never upscale
        self.pw = max(2, (int(W * sc) // 2) * 2)
        self.ph = max(2, (int(H * sc) // 2) * 2)
        self.sx, self.sy = self.pw / float(W), self.ph / float(H)

        pipe = Gst.Pipeline.new("preview")
        src = Gst.ElementFactory.make("appsrc", "psrc")
        conv = Gst.ElementFactory.make("videoconvert", "pconv")
        sink = None
        for name in ("ximagesink", "xvimagesink", "autovideosink"):
            sink = Gst.ElementFactory.make(name, "psink")
            if sink:
                break
        if not (src and conv and sink):
            raise RuntimeError("GStreamer display elements are missing "
                               "(Ubuntu: sudo apt install gstreamer1.0-x gstreamer1.0-plugins-base)")
        setp(src, "caps", Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self.pw},height={self.ph},framerate=0/1"))
        setp(src, "format", Gst.Format.TIME)
        setp(src, "is-live", True)
        setp(src, "do-timestamp", True)
        setp(src, "block", False)
        setp(src, "max-bytes", self.pw * self.ph * 3 * 4)
        setp(sink, "sync", False)                    # show frames immediately
        for el in (src, conv, sink):
            pipe.add(el)
        if not src.link(conv) or not conv.link(sink):
            raise RuntimeError("could not link the preview pipeline")
        bus = pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_msg)
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("could not open the preview window (is a desktop session running?)")
        self.pipe, self.appsrc = pipe, src
        print(f"Live preview window: {self.pw}x{self.ph} (max {self.max_w}x{self.max_h})")

    def _on_msg(self, bus, msg):
        if msg.type in (Gst.MessageType.ERROR, Gst.MessageType.EOS):
            if msg.type == Gst.MessageType.ERROR:
                err, _dbg = msg.parse_error()
                print(f"[preview] window closed or display error ({err.message}); "
                      f"the pipeline keeps running without the preview.")
            else:
                print("[preview] window closed; the pipeline keeps running without the preview.")
            self.ok = False
            GLib.idle_add(self.close)
        return True

    def show(self, rgb, det, persons, fps):
        """rgb: HxWx3 uint8 (blurred frame). det: Nx5 detections of the CURRENT frame."""
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        if self.appsrc is None:
            self._start(W, H)
        small = rgb if (self.pw, self.ph) == (W, H) else cv2.resize(
            rgb, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(small[:, :, ::-1])                 # RGB -> BGR
        if self.draw_boxes and len(det):
            for x1, y1, x2, y2, _s in det:
                cv2.rectangle(img, (int(x1 * self.sx), int(y1 * self.sy)),
                              (int(x2 * self.sx), int(y2 * self.sy)), (0, 255, 0), 1)
        # small overlay, top-left corner
        bw, bh = min(130, self.pw), min(40, self.ph)
        img[0:bh, 0:bw] = img[0:bh, 0:bw] // 3                        # darken the corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, f"Persons: {persons}", (6, 16), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"FPS: {fps:.1f}", (6, 34), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.appsrc.emit("push-buffer", Gst.Buffer.new_wrapped(img.tobytes()))

    def disable(self):
        self.ok = False
        self.close()

    def close(self):
        self.ok = False
        pipe, self.pipe = self.pipe, None
        if pipe is not None:
            try:
                pipe.set_state(Gst.State.NULL)
            except Exception:  # noqa: BLE001
                pass
        return False


# ----------------------------------------------------------------------------------
# the application
# ----------------------------------------------------------------------------------
class App:
    def __init__(self, a):
        self.a = a
        self.uri = to_uri(a.source)
        scheme = self.uri.split("://")[0].lower()
        self.live = scheme in ("rtsp", "http", "https")
        self.is_rtsp = scheme == "rtsp"
        self.is_hls = scheme in ("http", "https")
        self.dec = RetinaFaceDecoder(a.det_width, a.det_height, a.conf, a.nms)
        self.plan = None

        # steady output timestamps: on by default for HLS / http(s) (bursty), off for RTSP and files,
        # unless the user gives --out-fps explicitly. --no-retime always turns it off.
        self.do_retime = (not a.no_retime) and (self.is_hls or a.out_fps > 0)
        self.nominal_fps = None          # frame rate announced by the stream (if any)
        self.retime_dur = None           # ns per output frame, decided at the first frame
        self.out_base = None
        self.out_n = 0
        self.retime_failed = False

        # live preview window (optional)
        self.preview = None
        self.fps_ts = deque()            # arrival times of the last ~2 s (for the FPS shown)
        self.cur_fps = 0.0
        if a.preview:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                print("[note] --preview needs a desktop session (no DISPLAY found). "
                      "Preview is OFF; the pipeline runs normally.")
            else:
                self.preview = Preview(a.preview_w, a.preview_h, a.preview_fps, not a.no_preview_boxes)

        # compile + warm the CUDA kernels now, so the first video frame is not slow
        mod = cp.RawModule(code=CUDA_SRC)
        self.k_h, self.k_v = mod.get_function("blur_h"), mod.get_function("blur_v")
        fr = cp.zeros(64 * 256, dtype=cp.uint8)
        tm = cp.zeros_like(fr)
        bx = cp.asarray(np.array([[4, 4, 40, 40, 5]], dtype=np.int32))
        for k in (self.k_h, self.k_v):
            k((3, 3, 1), (16, 16, 1), (fr, tm, bx, np.int32(256)))
        cp.cuda.Stream.null.synchronize()
        self.tmp = None

        w = a.warmup
        self.st = {k: Stat(w) for k in ("detect", "post", "blur", "encode", "e2e")}
        self.t_mux, self.t_det_in, self.t_enc_in = OrderedDict(), OrderedDict(), OrderedDict()
        self.received = 0
        self.encoded = 0
        self.frame_idx = 0
        self.held = 0
        self.raw_counts = []
        self.n_saved_detections = 0
        self.frame_times = []
        self.pts = []
        self.last_boxes, self.miss = np.zeros((0, 5), np.float32), 0
        self.failed = False
        self.stopping = False
        self.src = None
        self.mux_sink_pad = None
        self.active_frames = 0
        self.first_frame_wall = None
        self.last_frame_wall = None
        self.warm_wall = self.warm_cpu = self.last_cpu = None
        self.frame_size = None
        self.pipeline = None
        self.loop = None
        self.mon = GpuMonitor(a.gpu, lambda: self.frame_idx >= self.a.warmup)
        # logs / audit
        self.det_dt = OrderedDict()                   # pts -> nvinfer time of that frame (s)
        self.f_det = self.f_aud = None
        self.det_log_path = self.aud_log_path = None
        self.audit = OrderedDict((k, 0) for k in (
            "applied", "applied_from_previous_frame", "partial", "skipped_no_faces", "failed"))

    # -------------------------------------------------------------- pipeline
    def write_nvinfer_config(self):
        a = self.a
        lines = [
            "[property]", f"gpu-id={a.gpu}",
            # RetinaFace (biubug6 / Pytorch_Retinaface): BGR input, subtract (104,117,123), no scaling
            "net-scale-factor=1.0", "offsets=104;117;123", "model-color-format=1",
            f"model-engine-file={os.path.abspath(a.engine)}",
        ]
        if a.onnx:
            lines.append(f"onnx-file={os.path.abspath(a.onnx)}")
        lines += [
            "batch-size=1", "network-mode=2", "process-mode=1",
            "network-type=100", "output-tensor-meta=1",
            "maintain-aspect-ratio=1", "symmetric-padding=0",
            "gie-unique-id=1", "interval=0", "num-detected-classes=1", "cluster-mode=4",
        ]
        path = os.path.join(tempfile.gettempdir(), "retinaface_pgie_config.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def build(self):
        a = self.a
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("face-blur")

        def mk(factory, name):
            el = Gst.ElementFactory.make(factory, name)
            if not el:
                sys.exit(f"Could not create '{factory}'. Is DeepStream installed and its plugins "
                         f"visible to GStreamer (GST_PLUGIN_PATH)?")
            self.pipeline.add(el)
            return el

        def mkq(name, leaky=0, size=6):
            q = mk("queue", name)
            setp(q, "max-size-buffers", size)
            setp(q, "max-size-bytes", 0)
            setp(q, "max-size-time", 0)
            if leaky:
                setp(q, "leaky", leaky)
            return q

        unified = int(getattr(pyds, "NVBUF_MEM_CUDA_UNIFIED", 3))

        src = mk("nvurisrcbin", "src")
        self.src = src
        setp(src, "uri", self.uri)
        if self.live:
            setp(src, "rtsp-reconnect-interval", 2)
            setp(src, "rtsp-reconnect-attempts", -1)
        else:
            setp(src, "file-loop", False)
        if self.is_rtsp:
            # TCP + jitter buffer: no lost packets, so no smeared / corrupted frames
            if a.rtsp_protocol == "tcp":
                setp(src, "select-rtp-protocol", 4)          # 4 = TCP only
            setp(src, "latency", int(a.latency))
            setp(src, "udp-buffer-size", 16 * 1024 * 1024)
            setp(src, "drop-on-latency", False)
            setp(src, "num-extra-surfaces", 10)

        mux = mk("nvstreammux", "mux")
        setp(mux, "batch-size", 1)
        setp(mux, "width", a.width)                      # ignored by the new nvstreammux
        setp(mux, "height", a.height)
        setp(mux, "batched-push-timeout", 33000)
        setp(mux, "live-source", 1 if self.live else 0)
        setp(mux, "nvbuf-memory-type", unified)
        setp(mux, "gpu-id", a.gpu)

        q1 = mkq("q1", 2 if a.leaky else 0, 2 if a.leaky else 6)
        pgie = mk("nvinfer", "pgie")
        setp(pgie, "config-file-path", self.write_nvinfer_config())
        q2 = mkq("q2")

        conv1 = mk("nvvideoconvert", "conv1")
        setp(conv1, "nvbuf-memory-type", unified)
        setp(conv1, "gpu-id", a.gpu)
        caps1 = mk("capsfilter", "caps1")
        caps1.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

        q3 = mkq("q3")
        conv2 = mk("nvvideoconvert", "conv2")
        setp(conv2, "gpu-id", a.gpu)
        caps2 = mk("capsfilter", "caps2")
        caps2.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

        enc = mk("nvv4l2h264enc" if a.codec == "h264" else "nvv4l2h265enc", "enc")
        setp(enc, "bitrate", a.bitrate)
        setp(enc, "iframeinterval", 30)
        setp(enc, "idrinterval", 30)
        setp(enc, "gpu-id", a.gpu)
        if a.codec == "h264":
            setp(enc, "insert-sps-pps", 1)   # helps players start and seek correctly
        parse = mk("h264parse" if a.codec == "h264" else "h265parse", "parse")
        setp(parse, "config-interval", -1)

        # Container from output extension. MKV is much more reliable for live/HLS saves.
        # Fragmented MP4 (fragment-duration) often only plays in VLC and can look corrupted.
        out_ext = os.path.splitext(a.output)[1].lower()
        if out_ext in (".mkv", ".webm"):
            mux_out = mk("matroskamux", "mkvmux")
            print(f"  [mux] using matroskamux for {a.output}")
        else:
            mux_out = mk("qtmux", "mp4mux")
            # Do NOT set fragment-duration - it breaks Windows Media Player and many teammates' players
            print(f"  [mux] using qtmux (progressive MP4) for {a.output}")

        sink = mk("filesink", "sink")
        setp(sink, "location", a.output)
        setp(sink, "sync", False)
        setp(sink, "async", False)

        chain = [mux, q1, pgie, q2, conv1, caps1, q3, conv2, caps2, enc, parse, mux_out, sink]
        for left, right in zip(chain, chain[1:]):
            if not left.link(right):
                sys.exit(f"Could not link {left.get_name()} -> {right.get_name()}")

        def on_pad_added(_el, pad):
            caps = pad.get_current_caps() or pad.query_caps(None)
            name = caps.get_structure(0).get_name() if caps and caps.get_size() else ""
            if not (name.startswith("video") or pad.get_name().startswith("vsrc")):
                return
            # remember the frame rate the stream announces (used for steady output timestamps)
            try:
                if caps and caps.get_size():
                    ok, num, den = caps.get_structure(0).get_fraction("framerate")
                    if ok and num > 0 and den > 0 and 1.0 <= num / den <= 120.0:
                        self.nominal_fps = num / den
            except Exception:  # noqa: BLE001
                pass
            req = getattr(mux, "request_pad_simple", None) or mux.get_request_pad
            sinkpad = req("sink_0")
            self.mux_sink_pad = sinkpad
            if sinkpad.is_linked():
                return
            if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
                print("[error] could not link the source to nvstreammux")

        src.connect("pad-added", on_pad_added)

        P = Gst.PadProbeType.BUFFER
        mux.get_static_pad("src").add_probe(P, self.p_mux_out, None)
        pgie.get_static_pad("sink").add_probe(P, self.p_det_in, None)
        pgie.get_static_pad("src").add_probe(P, self.p_det_out, None)
        caps1.get_static_pad("src").add_probe(P, self.p_main, None)
        if self.do_retime:
            caps2.get_static_pad("src").add_probe(P, self.p_retime, None)   # runs before enc sink probe
        enc.get_static_pad("sink").add_probe(P, self.p_enc_in, None)
        parse.get_static_pad("src").add_probe(P, self.p_enc_out, None)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus)

    # -------------------------------------------------------------- timing probes
    @staticmethod
    def _stamp(store, buf):
        store[buf.pts] = time.perf_counter()
        if len(store) > 400:
            store.popitem(last=False)

    @staticmethod
    def _lap(store, buf):
        t0 = store.pop(buf.pts, None)
        return None if t0 is None else time.perf_counter() - t0

    def p_mux_out(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            self.received += 1
            self._stamp(self.t_mux, buf)
        return Gst.PadProbeReturn.OK

    def p_det_in(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            self._stamp(self.t_det_in, buf)
        return Gst.PadProbeReturn.OK

    def p_det_out(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            dt = self._lap(self.t_det_in, buf)
            if dt is not None:
                self.st["detect"].add(dt)
                self.det_dt[buf.pts] = dt
                if len(self.det_dt) > 400:
                    self.det_dt.popitem(last=False)
        return Gst.PadProbeReturn.OK

    def p_retime(self, pad, info, _):
        """Give every saved frame a steady timestamp (fixed step) so the video plays smoothly,
        even when the source (HLS) delivers frames in bursts with long gaps."""
        buf = info.get_buffer()
        if buf is None or self.retime_failed:
            return Gst.PadProbeReturn.OK
        try:
            if self.retime_dur is None:
                fps = self.a.out_fps if self.a.out_fps > 0 else (self.nominal_fps or 25.0)
                self.retime_dur = int(round(1e9 / fps))
                src_txt = ("--out-fps" if self.a.out_fps > 0
                           else ("stream frame rate" if self.nominal_fps else "default"))
                print(f"Steady output timestamps: {fps:.2f} fps ({src_txt})")
            old = int(buf.pts)
            if self.out_base is None:
                self.out_base = 0 if old == int(Gst.CLOCK_TIME_NONE) else old
            new = self.out_base + self.out_n * self.retime_dur
            self.out_n += 1
            t0 = self.t_mux.pop(old, None)        # keep the end-to-end latency statistics working
            buf.pts = new
            buf.duration = self.retime_dur
            if t0 is not None:
                self.t_mux[new] = t0
        except Exception:  # noqa: BLE001
            self.retime_failed = True
            print("\n[warn] steady-timestamp probe failed, saving original timestamps instead:\n"
                  + traceback.format_exc())
        return Gst.PadProbeReturn.OK

    def p_enc_in(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            self._stamp(self.t_enc_in, buf)
        return Gst.PadProbeReturn.OK

    def p_enc_out(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            dt = self._lap(self.t_enc_in, buf)
            if dt is not None:
                self.st["encode"].add(dt)
            e2e = self._lap(self.t_mux, buf)
            if e2e is not None:
                self.st["e2e"].add(e2e)
                self.encoded += 1
        return Gst.PadProbeReturn.OK

    # -------------------------------------------------------------- main probe
    def p_main(self, pad, info, _):
        if self.failed:
            return Gst.PadProbeReturn.OK
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        try:
            bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            node = bm.frame_meta_list
            while node is not None:
                try:
                    fm = pyds.NvDsFrameMeta.cast(node.data)
                except StopIteration:
                    break
                self._process(buf, fm)
                try:
                    node = node.next
                except StopIteration:
                    break
        except Exception:  # noqa: BLE001
            self.failed = True
            print("\n[error] the frame probe failed:\n" + traceback.format_exc())
            GLib.idle_add(self.loop.quit)
        return Gst.PadProbeReturn.OK

    @staticmethod
    def _gpu_frame(buf, batch_id):
        """Wrap the GPU RGBA surface as a CuPy array (no copy)."""
        dtype, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(buf), batch_id)
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
        mem = cp.cuda.UnownedMemory(ptr, size, None)
        return cp.ndarray(shape=shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(mem, 0),
                          strides=strides)

    @staticmethod
    def _tensors(fm):
        node = fm.frame_user_meta_list
        while node is not None:
            try:
                um = pyds.NvDsUserMeta.cast(node.data)
            except StopIteration:
                break
            if um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
                outs = []
                for i in range(tm.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tm, i)
                    nd = layer.inferDims.numDims
                    dims = [int(layer.inferDims.d[k]) for k in range(nd)]
                    n = int(np.prod(dims)) if dims else 0
                    half = int(layer.dataType) == 1
                    ctype = ctypes.c_uint16 if half else ctypes.c_float
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctype))
                    arr = np.ctypeslib.as_array(ptr, shape=(n,))
                    if half:
                        arr = arr.view(np.float16).astype(np.float32)
                    outs.append((layer.layerName, dims, arr))
                return outs
            try:
                node = node.next
            except StopIteration:
                break
        return None

    def _blur(self, frame, rois, W, H):
        pitch = int(frame.strides[0])
        need = H * pitch
        if self.tmp is None or self.tmp.size < need:
            self.tmp = cp.empty(need, dtype=cp.uint8)
        boxes_gpu = cp.asarray(rois)
        maxw = int((rois[:, 2] - rois[:, 0]).max())
        maxh = int((rois[:, 3] - rois[:, 1]).max())
        grid = ((maxw + 15) // 16, (maxh + 15) // 16, int(len(rois)))
        args = (frame, self.tmp, boxes_gpu, np.int32(pitch))
        self.k_h(grid, (16, 16, 1), args)
        self.k_v(grid, (16, 16, 1), args)

    def _process(self, buf, fm):
        a = self.a
        idx = self.frame_idx
        t0 = time.perf_counter()

        frame = self._gpu_frame(buf, fm.batch_id)             # (H, W, 4) RGBA on the GPU
        H, W = int(frame.shape[0]), int(frame.shape[1])
        outs = self._tensors(fm)
        if outs is None:
            raise RuntimeError("No tensor output meta on the frame. nvinfer needs "
                               "network-type=100 and output-tensor-meta=1 (set by this script).")
        if self.plan is None:
            print("\nEngine outputs as seen by nvinfer:")
            for n, d, arr in outs:
                print(f"    {n}: dims={d} elements={arr.size}")
            self.plan = self.dec.build_plan(outs)
            self.frame_size = (W, H)
            print(f"Frame size at nvinfer: {W}x{H} | RetinaFace plan OK | blur kernels ready\n")
            if a.max_seconds > 0:
                GLib.timeout_add_seconds(int(a.max_seconds), self._timer_stop)

        det = self.dec.decode(self.plan, outs, W, H)
        raw = len(det)
        self.raw_counts.append(raw)
        boxes, held_used = det, False
        if raw:
            self.last_boxes, self.miss = det, 0
        else:
            self.miss += 1
            if self.miss <= a.hold and len(self.last_boxes):
                boxes = self.last_boxes
                self.held += 1
                held_used = True
        rois = make_rois(boxes, W, H, a.pad, a.blur_frac)
        t1 = time.perf_counter()

        # Periodic samples AND/OR every frame where a face was detected and blurred
        sample = a.sample_every > 0 and idx % a.sample_every == 0
        save_det = (not a.no_save_detections) and raw > 0 and len(rois) > 0
        need_cpu = sample or save_det
        orig = cp.asnumpy(frame[:, :, :3]) if need_cpu else None

        t2 = time.perf_counter()
        blur_err = None
        try:
            if len(rois):
                self._blur(frame, rois, W, H)
            cp.cuda.Stream.null.synchronize()                  # blurred pixels are final from here
        except Exception as e:  # noqa: BLE001
            blur_err = e
        t3 = time.perf_counter()

        self._log_frame(buf, idx, raw, det, held_used, len(rois), t1 - t0, t3 - t2, blur_err)
        if blur_err is not None:
            raise blur_err

        if need_cpu:
            after = cp.asnumpy(frame[:, :, :3])
            side = np.ascontiguousarray(np.hstack([orig, after])[:, :, ::-1])   # RGB -> BGR
            sc = 1800.0 / side.shape[1]
            if sc < 1.0:
                side = cv2.resize(side, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            if sample:
                cv2.imwrite(os.path.join(a.sample_dir, f"frame_{idx:06d}.jpg"), side,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            if save_det:
                # only the blurred frame (not side-by-side)
                 blurred = np.ascontiguousarray(after[:, :, ::-1])   # RGB -> BGR
                 name = f"det_{idx:06d}_faces{raw}.jpg"
                 cv2.imwrite(os.path.join(a.detected_dir, name), blurred,
                     [cv2.IMWRITE_JPEG_QUALITY, 85])
                 self.n_saved_detections += 1

        # --- current FPS (average over the last 2 seconds) + live preview window ----------
        tn = time.perf_counter()
        self.fps_ts.append(tn)
        while tn - self.fps_ts[0] > 2.0:
            self.fps_ts.popleft()
        if len(self.fps_ts) > 1 and tn > self.fps_ts[0]:
            self.cur_fps = (len(self.fps_ts) - 1) / (tn - self.fps_ts[0])
        pv = self.preview
        if pv is not None and pv.ok and pv.due(tn):
            try:
                pv.show(cp.asnumpy(frame[:, :, :3]), det, raw, self.cur_fps)
            except Exception:  # noqa: BLE001  (the preview must never stop the real work)
                print("\n[warn] live preview stopped because of an error (the run continues):\n"
                      + traceback.format_exc())
                pv.disable()

        self.st["post"].add(t1 - t0)
        self.st["blur"].add(t3 - t2)
        if len(self.pts) < 300:
            self.pts.append(int(buf.pts))
        now = time.perf_counter()
        if self.first_frame_wall is None:
            self.first_frame_wall = now
        if not self.stopping:                                   # shutdown frames do not count
            self.last_frame_wall = now
            self.last_cpu = time.process_time()
            self.active_frames += 1
            if idx >= a.warmup:
                self.frame_times.append(now)
        if idx == a.warmup:
            self.warm_wall, self.warm_cpu = now, time.process_time()
        self.frame_idx += 1
        if self.frame_idx % 100 == 0:
            el = now - self.first_frame_wall
            print(f"  {self.frame_idx} frames | {el:.0f}s | {self.frame_idx / max(el, 1e-9):.1f} fps "
                  f"| faces now: {raw}")

    # -------------------------------------------------------------- logs / privacy audit
    def open_logs(self):
        a = self.a
        self.f_det = self.f_aud = None
        if a.no_logs:
            return
        d = a.log_dir or os.path.dirname(os.path.abspath(a.metrics_json))
        os.makedirs(d, exist_ok=True)
        self.det_log_path = os.path.join(d, "detection_logs.jsonl")
        self.aud_log_path = os.path.join(d, "privacy_audit.jsonl")
        self.f_det = open(self.det_log_path, "w", encoding="utf-8")
        self.f_aud = open(self.aud_log_path, "w", encoding="utf-8")

    def close_logs(self):
        for f in (self.f_det, self.f_aud):
            if f is not None:
                try:
                    f.flush()
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
        self.f_det = self.f_aud = None

    def _log_frame(self, buf, idx, raw, det, held_used, n_blur, post_s, blur_s, err):
        """One line in detection_logs.jsonl and one in privacy_audit.jsonl for this frame."""
        # --- blur status + output safety status ---------------------------------------
        if err is not None:
            status, safety = "failed", "FAIL"
        elif raw > 0:
            status, safety = ("applied", "PASS") if n_blur >= raw else ("partial", "WARN")
        elif held_used and n_blur > 0:
            status, safety = "applied_from_previous_frame", "PASS_HELD"
        else:
            status, safety = "skipped_no_faces", "NO_FACES_DETECTED"
        self.audit[status] += 1
        if self.f_det is None:
            return

        pts = int(buf.pts)
        ts = datetime.now().astimezone().isoformat(timespec="milliseconds")
        det_s = self.det_dt.pop(buf.pts, None)
        conf = None
        if raw:
            s = det[:, 4]
            conf = {"min": round(float(s.min()), 4), "mean": round(float(s.mean()), 4),
                    "max": round(float(s.max()), 4)}
        rec = {
            "frame": idx, "pts_ns": pts, "ts": ts,
            "detections": raw, "confidence": conf, "held_boxes_used": held_used,
            "timing_ms": {
                "detect": None if det_s is None else round(det_s * 1000.0, 3),
                "decode": round(post_s * 1000.0, 3),
                "blur": round(blur_s * 1000.0, 3),
                "probe_total": round((post_s + blur_s) * 1000.0, 3)},
        }
        self.f_det.write(json.dumps(rec) + "\n")

        aud = {
            "frame": idx, "pts_ns": pts, "ts": ts,
            "blur_status": status,
            "faces_detected": raw,
            "faces_blurred": int(n_blur),
            "blur_source": ("current_frame" if raw else ("previous_frame" if held_used else "none")),
            "output_safety": safety,
        }
        if err is not None:
            aud["error"] = f"{type(err).__name__}: {str(err)[:200]}"
        self.f_aud.write(json.dumps(aud) + "\n")
        if idx % 200 == 0:
            self.f_det.flush()
            self.f_aud.flush()

    def audit_summary(self):
        c = dict(self.audit)
        if c["failed"]:
            overall = "FAIL"
        elif c["partial"]:
            overall = "WARN"
        else:
            overall = "PASS"
        return {"frames": int(sum(c.values())), "by_status": c, "overall": overall,
                "note": "PASS = every detected face was blurred. It cannot prove that no face "
                        "was missed by the detector."}

    # -------------------------------------------------------------- control
    def _timer_stop(self):
        self.request_stop()
        return False

    def _sigint(self):
        self.request_stop()
        return True

    def request_stop(self):
        if self.stopping:
            return
        self.stopping = True
        print("\nStopping: sending EOS so the video file is finalised ...")
        if self.live and self.src is not None:
            setp(self.src, "rtsp-reconnect-interval", 0)
        if self.mux_sink_pad is not None:
            self.mux_sink_pad.send_event(Gst.Event.new_eos())
        else:
            self.pipeline.send_event(Gst.Event.new_eos())
        GLib.timeout_add_seconds(15, self._force_quit)

    def _force_quit(self):
        print("[warn] EOS did not finish in 15 s, quitting anyway (the video file may be incomplete).")
        self.loop.quit()
        return False

    def on_bus(self, bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"\n[GStreamer ERROR] {err.message}\n{dbg}")
            self.failed = True
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            w, dbg = msg.parse_warning()
            print(f"[GStreamer warning] {w.message}")
        return True

    def run(self):
        self.build()
        self.open_logs()
        self.loop = GLib.MainLoop()
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, self._sigint)
        self.mon.start()
        run_start = time.perf_counter()
        cpu_start = time.process_time()
        print(f"Starting pipeline ({'live' if self.live else 'file'} source): {self.uri[:110]}")
        print("(first run: nvinfer builds the TensorRT engine from the ONNX, this can take a few minutes)")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            if self.preview is not None:
                self.preview.close()
            self.mon.stop()
            self.close_logs()
        self.wall_total = time.perf_counter() - run_start
        self.cpu_total = time.process_time() - cpu_start
        return self.report()

    # -------------------------------------------------------------- report
    def report(self):
        a = self.a
        n = self.frame_idx
        if n == 0:
            print("\nNo frames were processed. Check the messages above.")
            return 1
        ft = np.array(self.frame_times)
        steady_fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 2 and ft[-1] > ft[0] else 0.0
        whole_fps = self.active_frames / max(self.last_frame_wall - self.first_frame_wall, 1e-9)
        iv = np.diff(ft) * 1000.0 if len(ft) > 2 else np.array([0.0])
        per_s = np.array([0])
        if len(ft) > 2:
            per_s = np.bincount(np.floor(ft - ft[0]).astype(int))
            if len(per_s) > 1:
                per_s = per_s[:-1]
        fps_med, fps_min = float(np.median(per_s)), int(per_s.min())
        empty_s = int((per_s == 0).sum())
        gap_max = float(iv.max())
        pts = np.diff(np.array(self.pts, dtype=np.float64)) if len(self.pts) > 3 else np.array([])
        pts = pts[pts > 0]
        src_fps = 1e9 / float(np.median(pts)) if pts.size else 0.0
        if self.live:
            src_fps = 0.0
        rt = steady_fps / src_fps if src_fps else 0.0
        s = {k: v.summary() for k, v in self.st.items()}
        probe_mean = s["post"]["mean"] + s["blur"]["mean"]
        stages = {"detect (nvinfer)": s["detect"]["mean"], "probe (decode+blur)": probe_mean,
                  "encode (NVENC)": s["encode"]["mean"]}
        slow = max(stages, key=stages.get)
        est_max = 1000.0 / stages[slow] if stages[slow] > 0 else 0.0
        raw = np.array(self.raw_counts)
        cpu_cores = 0.0
        if self.warm_wall is not None and self.last_frame_wall > self.warm_wall:
            cpu_cores = (self.last_cpu - self.warm_cpu) / (self.last_frame_wall - self.warm_wall)
        m = self.mon
        avg = lambda v: float(np.mean(v)) if v else 0.0  # noqa: E731

        line = "=" * 64
        print("\n" + line)
        print(" PERFORMANCE REPORT")
        print(line)
        print(f" GPU                        : {m.name} ({m.total_mb} MB)")
        print(f" Pipeline                   : DeepStream + TensorRT engine, GPU blur (CUDA)")
        print(f" Model / input size         : {a.model_name} / {a.det_width}x{a.det_height}"
              f" (conf {a.conf}, NMS {a.nms})")
        if self.live:
            print(f" Source                     : {self.frame_size[0]}x{self.frame_size[1]} at nvinfer, live "
                  f"(delivering ~{steady_fps:.1f} fps)")
        else:
            print(f" Source                     : {self.frame_size[0]}x{self.frame_size[1]} at nvinfer, "
                  f"~{src_fps:.1f} fps (file)")
        if self.do_retime and self.retime_dur:
            print(f" Saved video timing         : steady {1e9 / self.retime_dur:.2f} fps "
                  f"(smooth playback){' [FAILED, original timestamps]' if self.retime_failed else ''}")
        else:
            print(f" Saved video timing         : original timestamps")
        print(f" Frames processed           : {n}  (first {a.warmup} excluded from timings)")
        print(f" Total run time             : {self.wall_total:.1f} s")
        print("-" * 64)
        print(f" Steady-state FPS           : {steady_fps:.1f}")
        print(f" Whole-run FPS              : {whole_fps:.1f}")
        dropped = max(self.received - n, 0)
        headroom = est_max / steady_fps if steady_fps else 0.0
        if self.live:
            in_flight = max(10, int(0.01 * n))
            keeping = dropped <= in_flight and headroom >= 1.0
            print(f" Keeping up with the stream : {'YES' if keeping else 'NO'}"
                  f"  (headroom {headroom:.1f}x, frames lost {dropped})")
            print(f" Delivered FPS per second   : median {fps_med:.0f} / min {fps_min}"
                  f"  (seconds with no frames: {empty_s})")
            print(f" Longest gap between frames : {gap_max:.0f} ms")
            if headroom >= 5:
                print(" Diagnosis                  : the PIPELINE IS NOT THE LIMIT (it is mostly idle).")
                print("                              FPS = what the source delivers.")
                if gap_max > 1000 or empty_s > 0 or fps_min < 0.5 * max(fps_med, 1):
                    print("                              The source is bursty / stalling: check network, "
                          "HLS quality or relay.")
            else:
                print(" Diagnosis                  : the pipeline is close to its own limit.")
        elif src_fps:
            print(f" Speed vs the video's rate  : {rt:.2f}x real time (file mode runs as fast as possible)")
        print(f" Frames: at mux / processed / encoded : {self.received} / {n} / {self.encoded}")
        print(f" Estimated max FPS          : {est_max:.1f}  (limited by: {slow})")
        print("-" * 64)
        print(" Latency per frame (ms)          mean    p50    p95    max")
        rows = [("Detect (nvinfer)", s["detect"]), ("Post: map+decode+NMS", s["post"]),
                ("Blur (GPU kernels)", s["blur"]), ("Encode (NVENC)", s["encode"]),
                ("End-to-end mux->file", s["e2e"])]
        for name, v in rows:
            print(f"   {name:<22}      {v['mean']:6.1f} {v['p50']:6.1f} {v['p95']:6.1f} {v['max']:6.1f}")
        print(f"   {'Frame interval':<22}      {iv.mean():6.1f} {np.percentile(iv, 50):6.1f} "
              f"{np.percentile(iv, 95):6.1f} {iv.max():6.1f}")
        print("-" * 64)
        print(f" Faces per frame (avg / max): {raw.mean():.2f} / {raw.max()}")
        print(f" Frames with 0 detections   : {(raw == 0).sum()} ({(raw == 0).mean() * 100:.1f}%)")
        print(f" Frames kept by 'hold' trick: {self.held}")
        print(f" Total face detections      : {int(raw.sum())}")
        print("-" * 64)
        print(f" CPU cores used (avg)       : {cpu_cores:.2f}   <- low = work is on the GPU")
        if m.ok:
            print(f" GPU utilisation avg / peak : {avg(m.util):.0f}% / {max(m.util or [0])}%")
            print(f" NVENC / NVDEC util avg     : {avg(m.enc):.0f}% / {avg(m.dec):.0f}%")
            print(f" GPU memory peak (whole GPU): {max(m.mem or [0])} MB")
        print(line)
        print(f" Output video   : {os.path.abspath(a.output)}")
        if a.sample_every > 0:
            print(f" Sample frames  : {os.path.abspath(a.sample_dir)}  (left original, right blurred)")
        if not a.no_save_detections:
            print(f" Detected frames: {os.path.abspath(a.detected_dir)}  "
                  f"({self.n_saved_detections} images, left original / right blurred)")
        print(f" Metrics JSON   : {os.path.abspath(a.metrics_json)}")
        if self.det_log_path:
            print(f" Detection log  : {os.path.abspath(self.det_log_path)}")
            print(f" Privacy audit  : {os.path.abspath(self.aud_log_path)}")
        asum = self.audit_summary()
        print(f" Privacy audit result: {asum['overall']}  {asum['by_status']}")

        report = dict(
            gpu=m.name, model=a.model_name, det_width=a.det_width, det_height=a.det_height, conf=a.conf, nms=a.nms,
            source=self.uri, live=self.live, frame_size=self.frame_size, source_fps=src_fps,
            frames=n, warmup_frames=a.warmup, steady_fps=steady_fps, whole_run_fps=whole_fps,
            realtime_factor=rt, headroom=headroom, delivered_fps_median=fps_med, delivered_fps_min=fps_min,
            seconds_without_frames=empty_s, longest_gap_ms=gap_max, frames_dropped=dropped,
            frames_received=self.received, frames_encoded=self.encoded,
            estimated_max_fps=est_max, bottleneck=slow, latency_ms=s,
            frame_interval_ms=dict(mean=float(iv.mean()), p50=float(np.percentile(iv, 50)),
                                   p95=float(np.percentile(iv, 95)), max=float(iv.max())),
            avg_faces_per_frame=float(raw.mean()), max_faces=int(raw.max()),
            frames_zero_detections=int((raw == 0).sum()), frames_held=self.held,
            total_detections=int(raw.sum()), cpu_cores_avg=cpu_cores,
            gpu_util_avg=avg(m.util), gpu_util_peak=max(m.util or [0]),
            nvenc_util_avg=avg(m.enc), nvdec_util_avg=avg(m.dec),
            gpu_mem_peak_mb=max(m.mem or [0]), output=os.path.abspath(a.output),
            output_fps=(1e9 / self.retime_dur) if (self.do_retime and self.retime_dur) else None,
            privacy_audit_summary=asum,
            detection_log=os.path.abspath(self.det_log_path) if self.det_log_path else None,
            privacy_audit_log=os.path.abspath(self.aud_log_path) if self.aud_log_path else None,
            sample_dir=os.path.abspath(a.sample_dir) if a.sample_every > 0 else None,
            detected_dir=os.path.abspath(a.detected_dir) if not a.no_save_detections else None,
            n_saved_detections=self.n_saved_detections)
        with open(a.metrics_json, "w") as f:
            json.dump(report, f, indent=2, default=float)
        return 0


# ----------------------------------------------------------------------------------

def parse_args():
    """CLI overrides YAML defaults from configs/*.yaml (via src/config_loader.py)."""
    cfg = load_pipeline_cfg()
    det = cfg.get("detection") or {}
    blur = cfg.get("blur") or {}
    pipe = cfg.get("pipeline") or {}
    cam = cfg.get("camera") or {}
    enc = cfg.get("encode") or {}
    prev = cfg.get("preview") or {}
    logc = cfg.get("logging") or {}

    p = argparse.ArgumentParser(
        description="DeepStream + TensorRT RetinaFace-R50 face blur (defaults from configs/)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default=cfg_get(cfg, "pipeline", "source"),
                   help="file path, rtsp://..., or http(s)://...m3u8 (or camera.yaml if omitted)")
    p.add_argument("--rutube-page", default=cfg_get(cfg, "pipeline", "rutube_page"),
                   help="Rutube page link; the HLS link is found automatically")
    p.add_argument("--onnx", default=str(det.get("onnx_path") or "retinaface_r50_1280x736.onnx"),
                   help="RetinaFace ONNX; nvinfer builds the TensorRT engine from it on first run")
    p.add_argument("--engine", default=str(det.get("engine_path") or ""),
                   help="TensorRT engine path (default from detection.yaml or <onnx>_b1_gpu0_fp16.engine)")
    p.add_argument("--model-name", default=str(det.get("model_name") or ""),
                   help="name shown in the report (default: guessed from the file name)")
    p.add_argument("--det-width", type=int, default=int(det.get("det_width") or 0),
                   help="ONNX input width (0 = read from file name)")
    p.add_argument("--det-height", type=int, default=int(det.get("det_height") or 0),
                   help="ONNX input height (0 = same auto rule)")
    p.add_argument("--conf", type=float, default=float(det.get("conf") if det.get("conf") is not None else 0.15),
                   help="face score threshold (lower = fewer misses)")
    p.add_argument("--nms", type=float, default=float(det.get("nms") if det.get("nms") is not None else 0.4))
    p.add_argument("--pad", type=float, default=float(blur.get("pad") if blur.get("pad") is not None else 0.20),
                   help="enlarge each face box by this fraction")
    p.add_argument("--blur-frac", type=float,
                   default=float(blur.get("blur_frac") if blur.get("blur_frac") is not None else 0.25),
                   help="blur radius as a fraction of the face size (bigger = stronger)")
    p.add_argument("--hold", type=int, default=int(blur.get("hold_frames") if blur.get("hold_frames") is not None else 5),
                   help="keep last boxes this many frames if a face is missed")
    p.add_argument("--output", default=str(pipe.get("output") or "output_blurred.mkv"))
    p.add_argument("--out-dir", default=str(pipe.get("out_dir") or "logs"),
                   help="folder where the video, logs, metrics and sample frames are saved")
    p.add_argument("--codec", choices=["h264", "h265"], default=str(enc.get("codec") or "h264"))
    p.add_argument("--bitrate", type=int, default=int(enc.get("bitrate") or 8_000_000))
    p.add_argument("--preview", action="store_true", default=bool(prev.get("enabled") or False),
                   help="open a small live window with the blurred video and a top-left overlay")
    p.add_argument("--no-preview", action="store_true", help="force preview off even if enabled in YAML")
    pw = int(prev.get("max_width") or 1000)
    ph = int(prev.get("max_height") or 1000)
    p.add_argument("--preview-size", default=f"{pw}x{ph}",
                   help="maximum preview window size WIDTHxHEIGHT")
    p.add_argument("--preview-fps", type=float, default=float(prev.get("fps") if prev.get("fps") is not None else 15),
                   help="preview refresh rate (does not affect saved video)")
    p.add_argument("--no-preview-boxes", action="store_true",
                   default=not bool(prev.get("draw_boxes", True)),
                   help="do not draw the green detection boxes in the preview window")
    p.add_argument("--out-fps", type=float, default=float(pipe.get("out_fps") if pipe.get("out_fps") is not None else 0),
                   help="fixed frame rate of the saved video (0 = automatic for HLS)")
    p.add_argument("--no-retime", action="store_true", default=bool(pipe.get("no_retime") or False),
                   help="keep original timestamps even for HLS")
    p.add_argument("--rtsp-protocol", choices=["tcp", "udp"],
                   default=str(cam.get("rtsp_protocol") or "tcp"),
                   help="RTSP transport")
    p.add_argument("--latency", type=int, default=int(cam.get("latency_ms") if cam.get("latency_ms") is not None else 500),
                   help="RTSP jitter buffer in ms")
    p.add_argument("--sample-dir", default=str(logc.get("sample_dir") or "sample_frames"))
    p.add_argument("--save-samples", action="store_true", default=bool(logc.get("save_samples") or False),
                   help="save before/after sample frames")
    p.add_argument("--sample-every", type=int, default=int(logc.get("sample_every") or 0),
                   help="save a before/after image every N frames (0 = off)")
    p.add_argument("--detected-dir", default=str(logc.get("detected_dir") or "detected_instances"),
                   help="folder for frames where a face was detected and blurred")
    p.add_argument("--no-save-detections", action="store_true",
                   default=not bool(logc.get("save_detections", True)),
                   help="do not save per-detection images")
    p.add_argument("--log-dir", default=str(logc.get("log_dir") or ""),
                   help="folder for detection_logs.jsonl and privacy_audit.jsonl")
    p.add_argument("--no-logs", action="store_true",
                   default=not bool(logc.get("write_jsonl_logs", True)),
                   help="do not write the two .jsonl log files")
    p.add_argument("--metrics-json", default=str(logc.get("metrics_json") or "metrics.json"))
    p.add_argument("--max-seconds", type=float,
                   default=float(pipe.get("max_seconds") if pipe.get("max_seconds") is not None else 0),
                   help="stop after N seconds (0 = until the end)")
    p.add_argument("--warmup", type=int,
                   default=int(pipe.get("warmup_frames") if pipe.get("warmup_frames") is not None else 30),
                   help="frames excluded from the timings")
    p.add_argument("--width", type=int, default=int(pipe.get("width") or 1536),
                   help="nvstreammux width (match your source)")
    p.add_argument("--height", type=int, default=int(pipe.get("height") or 864),
                   help="nvstreammux height (match your source)")
    p.add_argument("--max-height", type=int,
                   default=int(pipe.get("max_height_rutube") if pipe.get("max_height_rutube") is not None else 720),
                   help="stream quality cap for --rutube-page")
    p.add_argument("--leaky", action="store_true", default=bool(pipe.get("leaky") or False),
                   help="live: drop old frames instead of queueing when slow")
    p.add_argument("--gpu", type=int, default=int(pipe.get("gpu_id") if pipe.get("gpu_id") is not None else 0))
    a = p.parse_args()

    if a.no_preview:
        a.preview = False

    if a.rutube_page:
        a.source = resolve_rutube(a.rutube_page, a.max_height)
        print("Rutube HLS link:", a.source[:100], "...")
        with open("rutube_link.txt", "w") as _f:
            _f.write(a.source)
        print("(full link saved in rutube_link.txt)")

    if not a.source:
        built = build_rtsp_from_camera(cam)
        if built:
            a.source = built
            print("[config] source from camera.yaml (password redacted in logs if present)")
        else:
            p.error("give --source or --rutube-page, or set camera/source in configs/")

    ms = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", a.preview_size)
    if not ms:
        p.error("--preview-size must look like 1000x1000")
    a.preview_w, a.preview_h = int(ms.group(1)), int(ms.group(2))

    if not (a.det_width and a.det_height):
        mm = re.search(r"(\d+)x(\d+)", os.path.basename(a.onnx))
        a.det_width = a.det_width or (int(mm.group(1)) if mm else 640)
        a.det_height = a.det_height or (int(mm.group(2)) if mm else 640)
    print(f"Model input size: {a.det_width}x{a.det_height}")
    if not a.engine:
        a.engine = f"{a.onnx}_b1_gpu{a.gpu}_fp16.engine"
    a.model_name = a.model_name or guess_model_name(a.onnx, a.engine)
    if not os.path.exists(a.engine) and not os.path.exists(a.onnx):
        p.error(f"neither the engine ({a.engine}) nor the ONNX ({a.onnx}) was found")

    os.makedirs(a.out_dir, exist_ok=True)

    def _in_dir(path):
        return path if os.path.isabs(path) else os.path.join(a.out_dir, os.path.basename(path))

    a.output = _in_dir(a.output)
    a.metrics_json = _in_dir(a.metrics_json)
    a.sample_dir = _in_dir(a.sample_dir)
    a.detected_dir = _in_dir(a.detected_dir)
    a.log_dir = a.log_dir or a.out_dir
    if a.save_samples and a.sample_every <= 0:
        a.sample_every = 100
    if a.sample_every > 0:
        os.makedirs(a.sample_dir, exist_ok=True)
    if not a.no_save_detections:
        os.makedirs(a.detected_dir, exist_ok=True)
    a.cfg = cfg
    return a


if __name__ == "__main__":
    sys.exit(App(parse_args()).run())