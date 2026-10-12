#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_blur_deepstream.py
=======================
Face detection (SCRFD-2.5G TensorRT engine) + GPU face blur for live streams or
video files, using NVIDIA DeepStream for decode, batching, inference and encode.

Where the work happens
----------------------
  nvurisrcbin (NVDEC decode, RTSP/HLS/file, auto-reconnect)
    -> nvstreammux -> queue
    -> nvinfer (TensorRT engine, resize + normalise + inference on GPU)
    -> queue -> nvvideoconvert (to RGBA, on GPU)
    -> [python probe: decode SCRFD outputs (tiny, CPU) + CUDA blur kernels (GPU)]
    -> queue -> nvvideoconvert (to NV12) -> NVENC encoder -> parser -> mp4 file

The queues put the stages on separate threads, so decode / inference / blur /
encode overlap (pipeline parallelism). Frames never leave the GPU, except the
small SCRFD output tensors (about 1 MB) that nvinfer copies for the decoder.

Setup (once)
------------
  * x86 Linux + NVIDIA GPU + DeepStream 7.x or 8.x with the Python bindings (pyds).
    (Jetson is NOT supported: pyds.get_nvds_buf_surface_gpu is x86-only.)
  * pip install numpy cupy-cuda12x opencv-python-headless nvidia-ml-py requests
    (use cupy-cuda11x if your CUDA is 11.x)

Run examples
------------
  # video file, as fast as possible (benchmark + saved result)
  python3 face_blur_deepstream.py --source clip.mp4 --engine scrfd_2.5g.engine

  # live stream (RTSP or HLS .m3u8), run for 60 seconds
  python3 face_blur_deepstream.py --source rtsp://127.0.0.1:8554/cam \
          --engine scrfd_2.5g.engine --max-seconds 60 --width 1280 --height 720

  # Rutube page link (the script finds the HLS link itself)
  python3 face_blur_deepstream.py --rutube-page https://rutube.ru/live/video/<id>/ \
          --engine scrfd_2.5g.engine --max-seconds 60

  # engine built with another TensorRT version? give the ONNX too, nvinfer rebuilds it
  python3 face_blur_deepstream.py --source clip.mp4 --engine scrfd_2.5g.engine \
          --onnx scrfd_2.5g_bnkps.onnx

Press Ctrl+C at any time: the script finalises the mp4 and prints the report.
"""
import argparse
import ctypes
import json
import os
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict

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
    import re
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
# SCRFD output decoding (numpy, tiny: ~17k anchors)
# ----------------------------------------------------------------------------------
class ScrfdDecoder:
    STRIDES = (8, 16, 32)

    def __init__(self, size, conf, nms_iou):
        self.size, self.conf, self.nms_iou = size, conf, nms_iou
        self.centers, self.anchor_to_stride = {}, {}
        for s in self.STRIDES:
            n = size // s
            c = np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32)   # (x, y)
            c = (c * s).reshape(-1, 2)
            self.centers[s] = np.repeat(c, 2, axis=0)                           # 2 anchors / cell
            self.anchor_to_stride[n * n * 2] = s

    def build_plan(self, outs):
        """outs: list of (name, dims, array). Returns {(kind, stride): layer_index}."""
        plan = {}
        for i, (_, dims, arr) in enumerate(outs):
            if len(dims) < 2:
                continue
            c = int(dims[-1])
            kind = {1: "score", 4: "bbox", 10: "kps"}.get(c)
            stride = self.anchor_to_stride.get(arr.size // c) if c else None
            if kind and stride:
                plan[(kind, stride)] = i
        missing = [(k, s) for k in ("score", "bbox") for s in self.STRIDES if (k, s) not in plan]
        if missing:
            table = "\n".join(f"    {n}: dims={d} elements={a.size}" for n, d, a in outs)
            raise RuntimeError(
                "Could not match the engine outputs to SCRFD (missing: %s).\n"
                "Engine outputs seen by nvinfer:\n%s\n"
                "Expected 3 score layers (N,1), 3 bbox layers (N,4) with N = 2*(%d/stride)^2.\n"
                "Check --det-size equals the engine input size. If the dims above look like [1]\n"
                "or the batch dimension is missing, re-export the ONNX with a leading batch\n"
                "dimension on every output (see the notes that came with this script)."
                % (missing, table, self.size))
        return plan

    def decode(self, plan, outs, frame_w, frame_h):
        scale = min(self.size / frame_w, self.size / frame_h)    # nvinfer letterbox scale
        bl, sl = [], []
        for s in self.STRIDES:
            sc = outs[plan[("score", s)]][2].reshape(-1)
            keep = np.flatnonzero(sc >= self.conf)
            if keep.size == 0:
                continue
            bb = outs[plan[("bbox", s)]][2].reshape(-1, 4)[keep] * s
            c = self.centers[s][keep]
            bl.append(np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1],
                                c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], axis=1))
            sl.append(sc[keep])
        if not bl:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = np.concatenate(bl).astype(np.float32) / scale
        sc = np.concatenate(sl).astype(np.float32)
        if sc.size > 400:
            top = np.argsort(-sc)[:400]
            boxes, sc = boxes[top], sc[top]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h)
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
# the application
# ----------------------------------------------------------------------------------
class App:
    def __init__(self, a):
        self.a = a
        self.uri = to_uri(a.source)
        self.live = self.uri.split("://")[0].lower() in ("rtsp", "http", "https")
        self.dec = ScrfdDecoder(a.det_size, a.conf, a.nms)
        self.plan = None

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

    # -------------------------------------------------------------- pipeline
    def write_nvinfer_config(self):
        a = self.a
        lines = [
            "[property]", f"gpu-id={a.gpu}",
            "net-scale-factor=0.0078125", "offsets=127.5;127.5;127.5", "model-color-format=0",
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
        path = os.path.join(tempfile.gettempdir(), "scrfd_pgie_config.txt")
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
            setp(src, "rtsp-reconnect-interval", 2)      # same idea as the teammate's capture thread
            setp(src, "rtsp-reconnect-attempts", -1)
        else:
            setp(src, "file-loop", False)

        mux = mk("nvstreammux", "mux")
        setp(mux, "batch-size", 1)
        setp(mux, "width", a.width)                      # ignored by the new nvstreammux
        setp(mux, "height", a.height)
        setp(mux, "batched-push-timeout", 33000)
        setp(mux, "live-source", 1 if self.live else 0)
        setp(mux, "nvbuf-memory-type", unified)
        setp(mux, "gpu-id", a.gpu)

        q1 = mkq("q1", 2 if a.leaky else 0, 2 if a.leaky else 6)   # leaky = drop old frames when slow
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
        setp(enc, "gpu-id", a.gpu)
        parse = mk("h264parse" if a.codec == "h264" else "h265parse", "parse")
        mux_out = mk("qtmux", "mp4mux")
        if self.live:
            setp(mux_out, "fragment-duration", 2000)     # live recording stays playable even if cut
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
        """Wrap the GPU RGBA surface as a CuPy array (no copy). Pattern from NVIDIA's
        deepstream-imagedata-multistream-cupy sample."""
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
            print(f"Frame size at nvinfer: {W}x{H} | SCRFD plan OK | blur kernels ready\n")
            if a.max_seconds > 0:
                GLib.timeout_add_seconds(int(a.max_seconds), self._timer_stop)

        boxes = self.dec.decode(self.plan, outs, W, H)
        raw = len(boxes)
        self.raw_counts.append(raw)
        if raw:
            self.last_boxes, self.miss = boxes, 0
        else:
            self.miss += 1
            if self.miss <= a.hold and len(self.last_boxes):
                boxes = self.last_boxes
                self.held += 1
        rois = make_rois(boxes, W, H, a.pad, a.blur_frac)
        t1 = time.perf_counter()

        sample = a.sample_every > 0 and idx % a.sample_every == 0
        orig = cp.asnumpy(frame[:, :, :3]) if sample else None

        t2 = time.perf_counter()
        if len(rois):
            self._blur(frame, rois, W, H)
        cp.cuda.Stream.null.synchronize()                      # blurred pixels are final from here
        t3 = time.perf_counter()

        if sample:
            after = cp.asnumpy(frame[:, :, :3])
            side = np.ascontiguousarray(np.hstack([orig, after])[:, :, ::-1])   # RGB -> BGR
            sc = 1800.0 / side.shape[1]
            if sc < 1.0:
                side = cv2.resize(side, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(a.sample_dir, f"frame_{idx:06d}.jpg"), side,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

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
        print("\nStopping: sending EOS so the mp4 is finalised ...")
        if self.live and self.src is not None:
            # without this, nvurisrcbin sees "no data" after the stop and reconnects the camera
            setp(self.src, "rtsp-reconnect-interval", 0)
        if self.mux_sink_pad is not None:
            self.mux_sink_pad.send_event(Gst.Event.new_eos())      # EOS goes straight into the pipeline
        else:
            self.pipeline.send_event(Gst.Event.new_eos())
        GLib.timeout_add_seconds(15, self._force_quit)

    def _force_quit(self):
        print("[warn] EOS did not finish in 15 s, quitting anyway (the mp4 may be incomplete).")
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
        self.loop = GLib.MainLoop()
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, self._sigint)
        self.mon.start()
        run_start = time.perf_counter()
        cpu_start = time.process_time()
        print(f"Starting pipeline ({'live' if self.live else 'file'} source): {self.uri[:110]}")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            self.mon.stop()
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
        pts = np.diff(np.array(self.pts, dtype=np.float64)) if len(self.pts) > 3 else np.array([])
        pts = pts[pts > 0]
        src_fps = 1e9 / float(np.median(pts)) if pts.size else 0.0
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
        print(f" Model / input size         : SCRFD-2.5G / {a.det_size}x{a.det_size}"
              f" (conf {a.conf}, NMS {a.nms})")
        print(f" Source                     : {self.frame_size[0]}x{self.frame_size[1]} at nvinfer, "
              f"~{src_fps:.1f} fps ({'live' if self.live else 'file'})")
        print(f" Frames processed           : {n}  (first {a.warmup} excluded from timings)")
        print(f" Total run time             : {self.wall_total:.1f} s")
        print("-" * 64)
        print(f" Steady-state FPS           : {steady_fps:.1f}")
        print(f" Whole-run FPS              : {whole_fps:.1f}")
        dropped = max(self.received - n, 0)
        headroom = est_max / steady_fps if steady_fps else 0.0
        if self.live:
            keeping = dropped <= 2 and headroom >= 1.0
            print(f" Keeping up with the stream : {'YES' if keeping else 'NO'}"
                  f"  (headroom {headroom:.1f}x, frames dropped {dropped})")
            print(f" Note                       : a live source arrives at its own rate (~{steady_fps:.0f} fps);")
            print(f"                              FPS cannot go higher than that. Use the headroom number.")
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
        print(f" Metrics JSON   : {os.path.abspath(a.metrics_json)}")

        report = dict(
            gpu=m.name, model="SCRFD-2.5G", det_size=a.det_size, conf=a.conf, nms=a.nms,
            source=self.uri, live=self.live, frame_size=self.frame_size, source_fps=src_fps,
            frames=n, warmup_frames=a.warmup, steady_fps=steady_fps, whole_run_fps=whole_fps,
            realtime_factor=rt, headroom=headroom, frames_dropped=dropped, frames_received=self.received, frames_encoded=self.encoded,
            estimated_max_fps=est_max, bottleneck=slow, latency_ms=s,
            frame_interval_ms=dict(mean=float(iv.mean()), p50=float(np.percentile(iv, 50)),
                                   p95=float(np.percentile(iv, 95)), max=float(iv.max())),
            avg_faces_per_frame=float(raw.mean()), max_faces=int(raw.max()),
            frames_zero_detections=int((raw == 0).sum()), frames_held=self.held,
            total_detections=int(raw.sum()), cpu_cores_avg=cpu_cores,
            gpu_util_avg=avg(m.util), gpu_util_peak=max(m.util or [0]),
            nvenc_util_avg=avg(m.enc), nvdec_util_avg=avg(m.dec),
            gpu_mem_peak_mb=max(m.mem or [0]), output=os.path.abspath(a.output))
        with open(a.metrics_json, "w") as f:
            json.dump(report, f, indent=2, default=float)
        return 0


# ----------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DeepStream + TensorRT SCRFD face blur",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", help="file path, rtsp://..., or http(s)://...m3u8")
    p.add_argument("--rutube-page", help="Rutube page link; the HLS link is found automatically")
    p.add_argument("--engine", required=True, help="SCRFD TensorRT engine file")
    p.add_argument("--onnx", default="", help="optional ONNX; nvinfer rebuilds the engine if needed")
    p.add_argument("--det-size", type=int, default=640, help="engine input size (square)")
    p.add_argument("--conf", type=float, default=0.35, help="detection threshold (lower = fewer misses)")
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--pad", type=float, default=0.20, help="enlarge each face box by this fraction")
    p.add_argument("--blur-frac", type=float, default=0.25,
                   help="blur radius as a fraction of the face size (bigger = stronger)")
    p.add_argument("--hold", type=int, default=5, help="keep last boxes this many frames if a face is missed")
    p.add_argument("--output", default="output_blurred.mp4")
    p.add_argument("--codec", choices=["h264", "h265"], default="h264")
    p.add_argument("--bitrate", type=int, default=8_000_000)
    p.add_argument("--sample-dir", default="sample_frames")
    p.add_argument("--sample-every", type=int, default=100, help="save before/after image every N frames (0 = off)")
    p.add_argument("--metrics-json", default="metrics.json")
    p.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds (0 = until the end)")
    p.add_argument("--warmup", type=int, default=30, help="frames excluded from the timings")
    p.add_argument("--width", type=int, default=1920, help="nvstreammux width (match your source)")
    p.add_argument("--height", type=int, default=1080, help="nvstreammux height (match your source)")
    p.add_argument("--max-height", type=int, default=720, help="stream quality cap for --rutube-page")
    p.add_argument("--leaky", action="store_true", help="live: drop old frames instead of queueing when slow")
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    if a.rutube_page:
        a.source = resolve_rutube(a.rutube_page, a.max_height)
        print("Rutube HLS link:", a.source[:100], "...")
    if not a.source:
        p.error("give --source or --rutube-page")
    if not os.path.exists(a.engine) and not a.onnx:
        p.error(f"engine file not found: {a.engine} (or pass --onnx so it can be built)")
    if a.sample_every > 0:
        os.makedirs(a.sample_dir, exist_ok=True)
    return a


if __name__ == "__main__":
    sys.exit(App(parse_args()).run())