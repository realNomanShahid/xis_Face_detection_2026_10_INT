#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retinaface_face_blur_gpu_ds_preview.py
======================================
Face detection (RetinaFace ResNet-50, ONNX -> TensorRT) + face blur for live streams or
video files, using NVIDIA DeepStream for decode, batching, inference and encode.

The whole per-frame path runs on the GPU (decode, TensorRT, RetinaFace post-processing,
NMS, ROI building, blur, JPEG encode, NVENC) and now also the LIVE PREVIEW.

NEW IN THIS VERSION: the live preview (--preview) is a DeepStream branch, not OpenCV.
  Before: full frame copied GPU -> CPU, cv2.resize, cv2.rectangle / putText, appsrc -> ximagesink.
  Now   : tee after the blur -> queue -> nvvideoconvert (GPU scale) -> nvdsosd (GPU draws the boxes
          and the FPS text) -> nveglglessink (GPU / EGL window).
          Only the face boxes (N x 5 floats) are read back to build the OSD metadata; pixels never
          leave the GPU. The saved video is never touched by the preview.
  The CPU / GPU report is recalculated for this: the preview work moves from the CPU column to the
  GPU column, and the profiler counts what is left on the CPU (box readback + OSD metadata).

NEW IN THIS VERSION: built-in CPU / GPU measurement and an optimization report.
  * From the moment the program starts, a sampler thread records CPU and GPU usage
    (system CPU %, this process' CPU cores, RAM, GPU busy %, memory-controller %, NVENC %,
     NVDEC %, VRAM, power, temperature, SM clock).
  * Every stage of the per-frame path is timed: GPU stages with CUDA events, CPU stages with
    thread CPU time, and the places where the CPU just waits for the GPU are listed separately.
  * At the end the program prints AND SAVES a report that contains
        - CPU and GPU usage at start / steady state / peak / at finish
        - which function runs on the GPU and which on the CPU (every function in this file)
        - the workload split GPU % / CPU % (= 100 %, measured busy time per frame)
        - the code split GPU % / CPU % (= 100 %, executable lines of this file)
        - the key metrics used to judge optimization, automatic bottleneck hints,
          metric definitions and caveats
  * Use --no-profile to switch off the per-stage timing (the resource sampler stays on).

Files written next to metrics.json (in --out-dir):
  optimization_report.txt            the human-readable report (also printed at the end)
  optimization_report.json           the same numbers, machine-readable
  optimization_report_timeline.csv   CPU / GPU usage sampled over time (plot it in Excel)

  nvurisrcbin (NVDEC decode, RTSP/HLS/file, auto-reconnect)                        GPU (NVDEC)
    -> nvstreammux -> queue                                                         GPU
    -> nvinfer (TensorRT engine, resize + normalise + inference)                    GPU
    -> queue -> nvvideoconvert (to RGBA)                                            GPU
    -> [python probe]
         upload raw model outputs (loc/conf) -> GPU
         GPU decode kernel  : softmax, box decode, clip, size filter                GPU
         GPU top-K sort + GPU bitmask NMS (CUDA kernels)                            GPU
         GPU ROI building + compaction                                              GPU
         1 tiny readback (7 numbers: counts, grid size, confidence stats)           control only
         CUDA blur kernels (in place on the NVMM surface)                           GPU
         JPEG of detected frames: nvJPEG encode on GPU (CPU fallback if missing)    GPU
    -> tee -+-> queue -> nvvideoconvert (to NV12)                                   GPU
            |   -> NVENC encoder -> parser -> mp4 / mkv file                          GPU (NVENC)
            +-> (only with --preview) queue(leaky) -> nvvideoconvert (scale)          GPU
                -> nvdsosd (boxes + FPS text) -> nveglglessink (window)               GPU

What intentionally stays on the CPU (not compute): GStreamer plumbing, argument parsing,
JSON logs, file writes (done in a background thread), the small on-screen-display metadata
for the preview (face boxes + FPS text), and the one-time start-up work (anchor table,
output layer matching on the first frame).

Setup (once)
------------
  * x86 Linux + NVIDIA GPU + DeepStream 7.x or 8.x with the Python bindings (pyds).
    (Jetson is NOT supported: pyds.get_nvds_buf_surface_gpu is x86-only.)
  * pip install numpy cupy-cuda12x opencv-python-headless nvidia-ml-py requests
  * optional, for GPU JPEG encoding:  pip install nvidia-nvimgcodec-cu12
    (without it the script prints a note and encodes JPEGs with OpenCV in a background thread)

Run examples
------------
  # real IP camera (RTSP over TCP) with the live window
  python3 retinaface_face_blur_gpu_ds_preview.py --source rtsp://user:pass@192.168.1.20:554/stream1 \
          --onnx retinaface_r50_1280x736.onnx --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
          --width 1280 --height 720 --conf 0.4 --preview

  # Rutube page link with the live window
  python3 retinaface_face_blur_gpu_ds_preview.py --rutube-page https://rutube.ru/live/video/<id>/ \
          --onnx retinaface_r50_1280x736.onnx --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
          --width 1280 --height 720 --max-height 720 --conf 0.4 --out-fps 13 --preview --max-seconds 60

  # video file (benchmark: no preview, no per-frame images, no logs)
  python3 retinaface_face_blur_gpu_ds_preview.py --source clip.mp4 \
          --onnx retinaface_r50_1280x736.onnx --conf 0.4 --no-save-detections --no-logs

Output files
------------
  output_blurred.mkv      the blurred video (use --output name.mp4 for MP4)
  metrics.json            performance report (+ a privacy-audit summary + the optimization report)
  optimization_report.*   see above
  detection_logs.jsonl    one line per frame: frame index, detection count, confidence
                          summary (min/mean/max), timing
  privacy_audit.jsonl     one line per frame: blur operation status, detected face count,
                          output safety status
  The two .jsonl logs are written next to metrics.json (or in --log-dir). --no-logs turns them off.
  sample_frames/          ONLY created when you ask for it: --save-samples (every 100 frames)
                          or --sample-every N. Without these flags nothing is stored.
  detected_instances/     EVERY frame where a face was detected and blurred (blurred frame only).
                          On by default; disable with --no-save-detections.

Press Ctrl+C at any time: the script finalises the video file and prints the report.
"""
import argparse
import ast
import csv
import ctypes
import functools
import json
import os
import platform
import queue
import re
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict, deque
from datetime import datetime

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
# CUDA kernels
#   blur_h / blur_v   : separable box blur inside an ellipse, all faces in 2 launches
#   rf_decode         : RetinaFace softmax + box decode + clip + size filter, one thread per anchor
#   nms_mask/nms_reduce: bitmask NMS (pairwise IoU on the GPU, then a one-block greedy pass)
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

// RetinaFace decode, one thread per anchor.
//   score : face probability (softmax over [background, face] or already a probability);
//           set to -1 when the decoded box is smaller than 2 px (so it can never pass the threshold)
//   box   : x1, y1, x2, y2 in ORIGINAL frame pixels (letterbox undone, clipped to the frame)
__global__ void rf_decode(const float* loc, const float* conf, const float* pri,
                          int n, int use_softmax, float v0, float v1,
                          float in_w, float in_h, float scale, float fw, float fh,
                          float* out_score, float* out_box) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float s;
    if (use_softmax) {
        float d = conf[2 * i] - conf[2 * i + 1];
        d = fminf(fmaxf(d, -50.0f), 50.0f);
        s = 1.0f / (1.0f + expf(d));
    } else {
        s = conf[2 * i + 1];
    }

    const float pcx = pri[4 * i], pcy = pri[4 * i + 1], pw = pri[4 * i + 2], ph = pri[4 * i + 3];
    const float cx = pcx + loc[4 * i]     * v0 * pw;
    const float cy = pcy + loc[4 * i + 1] * v0 * ph;
    const float w  = pw * expf(fminf(fmaxf(loc[4 * i + 2] * v1, -10.0f), 10.0f));
    const float h  = ph * expf(fminf(fmaxf(loc[4 * i + 3] * v1, -10.0f), 10.0f));

    float x1 = (cx - 0.5f * w) * in_w / scale;
    float y1 = (cy - 0.5f * h) * in_h / scale;
    float x2 = (cx + 0.5f * w) * in_w / scale;
    float y2 = (cy + 0.5f * h) * in_h / scale;
    x1 = fminf(fmaxf(x1, 0.0f), fw);  x2 = fminf(fmaxf(x2, 0.0f), fw);
    y1 = fminf(fmaxf(y1, 0.0f), fh);  y2 = fminf(fmaxf(y2, 0.0f), fh);
    if ((x2 - x1) < 2.0f || (y2 - y1) < 2.0f) s = -1.0f;

    out_score[i] = s;
    out_box[4 * i]     = x1;
    out_box[4 * i + 1] = y1;
    out_box[4 * i + 2] = x2;
    out_box[4 * i + 3] = y2;
}

// NMS step 1: boxes are sorted by score (descending). mask[i, w] has bit b set when box
// (w*64 + b) overlaps box i with IoU > thr and comes AFTER i. n is read from device memory,
// so the host never has to know how many candidates survived the threshold.
__global__ void nms_mask(const float* boxes, const int* n_ptr, int nwords, float thr,
                         unsigned long long* mask) {
    const int n = *n_ptr;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y;
    if (i >= n) return;

    const float ax1 = boxes[4 * i], ay1 = boxes[4 * i + 1], ax2 = boxes[4 * i + 2], ay2 = boxes[4 * i + 3];
    const float aa = fmaxf(0.0f, ax2 - ax1) * fmaxf(0.0f, ay2 - ay1);
    unsigned long long bits = 0ULL;
    const int j0 = w * 64;
    for (int b = 0; b < 64; ++b) {
        const int j = j0 + b;
        if (j >= n) break;
        if (j <= i) continue;
        const float bx1 = boxes[4 * j], by1 = boxes[4 * j + 1], bx2 = boxes[4 * j + 2], by2 = boxes[4 * j + 3];
        const float iw = fmaxf(0.0f, fminf(ax2, bx2) - fmaxf(ax1, bx1));
        const float ih = fmaxf(0.0f, fminf(ay2, by2) - fmaxf(ay1, by1));
        const float inter = iw * ih;
        const float ab = fmaxf(0.0f, bx2 - bx1) * fmaxf(0.0f, by2 - by1);
        const float iou = inter / (aa + ab - inter + 1e-9f);
        if (iou > thr) bits |= (1ULL << b);
    }
    mask[(size_t)i * nwords + w] = bits;
}

// NMS step 2: greedy pass over the sorted boxes in ONE thread block. A box that is not yet
// suppressed is kept and suppresses everything its mask row says. Needs nwords*8 bytes of
// dynamic shared memory.
__global__ void nms_reduce(const unsigned long long* mask, const int* n_ptr, int nwords,
                           int* keep_idx, int* keep_cnt) {
    extern __shared__ unsigned long long removed[];
    __shared__ int cnt;
    const int n = *n_ptr;
    const int t = threadIdx.x;
    for (int w = t; w < nwords; w += blockDim.x) removed[w] = 0ULL;
    if (t == 0) cnt = 0;
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        const bool gone = ((removed[i >> 6] >> (i & 63)) & 1ULL) != 0ULL;
        __syncthreads();
        if (!gone) {
            if (t == 0) { keep_idx[cnt] = i; cnt += 1; }
            for (int w = t; w < nwords; w += blockDim.x)
                removed[w] |= mask[(size_t)i * nwords + w];
        }
        __syncthreads();
    }
    if (t == 0) *keep_cnt = cnt;
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


# ==================================================================================
# PROFILING: per-stage CPU / GPU timing, CPU + GPU resource sampler, code split, report
# ==================================================================================
# Which functions of THIS file run on the GPU. Everything that is not listed runs on the CPU.
# "GPU" = the heavy work of the function executes on the GPU (CUDA / CuPy / nvJPEG); the few
# Python lines inside only enqueue that work. The CUDA kernels in CUDA_SRC are counted as GPU.
DEVICE_OF = {
    "RetinaFaceDecoder.process_gpu": "GPU",
    "RetinaFaceDecoder.warm": "GPU",
    "App._blur": "GPU",
    "App._to_gpu_f32": "GPU",
    "App._gpu_resize": "GPU",
    "JpegSaver._encode": "GPU",
}

# Pipeline elements that are NOT Python code in this file (DeepStream / GStreamer plugins).
PIPELINE_MAP = [
    ("nvurisrcbin (video decode)", "GPU (NVDEC hardware); network/demux part on CPU (light)"),
    ("nvstreammux (batching)", "GPU"),
    ("nvinfer (resize + normalise + TensorRT)", "GPU (TensorRT); output tensors are copied to host by nvinfer"),
    ("nvvideoconvert x2 (RGBA / NV12)", "GPU"),
    ("nvv4l2h264enc / nvv4l2h265enc", "GPU (NVENC hardware)"),
    ("preview branch: tee + nvvideoconvert (scale)", "GPU (only with --preview)"),
    ("preview branch: nvdsosd (boxes + FPS text)", "GPU (only with --preview)"),
    ("preview branch: nveglglessink (window)", "GPU / EGL (only with --preview)"),
    ("h264parse, matroskamux/qtmux, filesink", "CPU (light)"),
    ("GStreamer queues / streaming threads", "CPU (light)"),
]

INCLUSIVE = "[inclusive] whole frame probe (App._process)"
JOIN_GPU = "host: enqueue GPU work (launch overhead)"


class _NullSpan:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_SPAN = _NullSpan()


class _CpuSpan:
    """Times a block with wall clock and THREAD CPU time (sleeping/blocked time is not CPU)."""
    __slots__ = ("p", "name", "dev", "kind", "func", "in_proc", "t0", "c0")

    def __init__(self, p, name, dev, kind, func, in_proc):
        self.p, self.name, self.dev, self.kind, self.func, self.in_proc = p, name, dev, kind, func, in_proc

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.c0 = time.thread_time()
        return self

    def __exit__(self, *exc):
        self.p.add(self.name, self.dev, self.kind, self.func,
                   wall=time.perf_counter() - self.t0, cpu=time.thread_time() - self.c0,
                   in_proc=self.in_proc)
        return False


class _GpuSpan:
    """Brackets a block of GPU work with two CUDA events (true GPU-side elapsed time)."""
    __slots__ = ("p", "name", "kind", "func", "e0", "t0", "c0")

    def __init__(self, p, name, kind, func):
        self.p, self.name, self.kind, self.func = p, name, kind, func

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.c0 = time.thread_time()
        self.e0 = self.p._event()
        self.e0.record()
        return self

    def __exit__(self, *exc):
        e1 = self.p._event()
        e1.record()
        self.p._pending.append((self.name, self.kind, self.func, self.e0, e1,
                                time.perf_counter() - self.t0, time.thread_time() - self.c0))
        return False


class Profiler:
    """Collects per-stage timings. Switched on after the warm-up frames (see App._process)."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.active = False
        self.frames = 0
        self.lock = threading.Lock()
        self.fn = OrderedDict()
        self._pool, self._pending = [], []

    def _event(self):
        return self._pool.pop() if self._pool else cp.cuda.Event()

    def add(self, name, device, kind, func, wall=0.0, cpu=0.0, gpu_ms=0.0, in_proc=False):
        if not self.active:
            return
        with self.lock:
            d = self.fn.get(name)
            if d is None:
                d = self.fn[name] = dict(device=device, kind=kind, func=func, in_proc=in_proc,
                                         calls=0, wall=0.0, cpu=0.0, gpu_ms=0.0)
            d["calls"] += 1
            d["wall"] += wall
            d["cpu"] += cpu
            d["gpu_ms"] += gpu_ms

    def cpu(self, name, kind, func, in_proc=True):
        return _CpuSpan(self, name, "CPU", kind, func, in_proc) if self.active else _NULL_SPAN

    def wait(self, name, kind, func, in_proc=True):
        """Host thread blocked waiting for the GPU (not counted as CPU work or GPU work)."""
        return _CpuSpan(self, name, "WAIT", kind, func, in_proc) if self.active else _NULL_SPAN

    def gpu(self, name, kind, func):
        return _GpuSpan(self, name, kind, func) if self.active else _NULL_SPAN

    def flush_gpu(self):
        """Turn the recorded CUDA events into milliseconds (call after the GPU has been synced)."""
        pend, self._pending = self._pending, []
        for name, kind, func, e0, e1, hw, hc in pend:
            try:
                e1.synchronize()
                ms = float(cp.cuda.get_elapsed_time(e0, e1))
            except Exception:  # noqa: BLE001
                ms = 0.0
            self.add(name, "GPU", kind, func, gpu_ms=ms, in_proc=True)
            self.add(JOIN_GPU, "CPU", "kernel launch / enqueue", "all GPU stages",
                     wall=hw, cpu=hc, in_proc=True)
            self._pool.extend((e0, e1))


NULL_PROF = Profiler(enabled=False)


def timed(name):
    """Decorator for the GStreamer probe callbacks of App (pure CPU Python)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *a, **k):
            p = self.prof
            if not p.active:
                return fn(self, *a, **k)
            t0, c0 = time.perf_counter(), time.thread_time()
            try:
                return fn(self, *a, **k)
            finally:
                p.add(name, "CPU", "pipeline callback (python)", f"App.{fn.__name__}",
                      wall=time.perf_counter() - t0, cpu=time.thread_time() - c0, in_proc=False)
        return wrapper
    return deco


class ResourceMonitor(threading.Thread):
    """Samples CPU and GPU usage from the moment the program starts until it ends."""

    def __init__(self, gpu_index, phase_fn, period=0.5):
        super().__init__(daemon=True)
        self.phase_fn = phase_fn
        self.period = max(float(period), 0.1)
        self.ncpu = os.cpu_count() or 1
        self.samples = []
        self.ok = False
        self.gpu_name, self.driver, self.vram_total_mb = "n/a", "n/a", 0
        self._halt = threading.Event()
        self.t_start = time.perf_counter()
        try:
            import pynvml as nv
            nv.nvmlInit()
            self.nv = nv
            self.h = nv.nvmlDeviceGetHandleByIndex(gpu_index)
            name = nv.nvmlDeviceGetName(self.h)
            self.gpu_name = name.decode() if isinstance(name, bytes) else str(name)
            self.vram_total_mb = int(nv.nvmlDeviceGetMemoryInfo(self.h).total // 2 ** 20)
            try:
                drv = nv.nvmlSystemGetDriverVersion()
                self.driver = drv.decode() if isinstance(drv, bytes) else str(drv)
            except Exception:  # noqa: BLE001
                pass
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[note] NVML not available, GPU usage will not be sampled ({e}). "
                  f"Install with: pip install nvidia-ml-py")

    @staticmethod
    def _sys_cpu_times():
        try:
            with open("/proc/stat") as f:
                v = [int(x) for x in f.readline().split()[1:9]]
            return sum(v), v[3] + v[4]                  # total, idle (+iowait)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _rss_mb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except Exception:  # noqa: BLE001
            pass
        return None

    def _gpu_sample(self):
        nv, out = self.nv, {}
        try:
            u = nv.nvmlDeviceGetUtilizationRates(self.h)
            out["gpu_util"], out["gpu_mem_util"] = int(u.gpu), int(u.memory)
        except Exception:  # noqa: BLE001
            pass
        try:
            out["enc"] = int(nv.nvmlDeviceGetEncoderUtilization(self.h)[0])
        except Exception:  # noqa: BLE001
            pass
        try:
            out["dec"] = int(nv.nvmlDeviceGetDecoderUtilization(self.h)[0])
        except Exception:  # noqa: BLE001
            pass
        try:
            out["vram_mb"] = int(nv.nvmlDeviceGetMemoryInfo(self.h).used // 2 ** 20)
        except Exception:  # noqa: BLE001
            pass
        try:
            out["power_w"] = nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
        except Exception:  # noqa: BLE001
            pass
        try:
            out["temp_c"] = int(nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU))
        except Exception:  # noqa: BLE001
            pass
        try:
            out["sm_mhz"] = int(nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM))
        except Exception:  # noqa: BLE001
            pass
        return out

    def run(self):
        prev_sys, prev_proc, prev_t = self._sys_cpu_times(), time.process_time(), time.perf_counter()
        wait = 0.25                                       # first sample soon after start
        while True:
            stopped = self._halt.wait(wait)
            wait = self.period
            now = time.perf_counter()
            dt = max(now - prev_t, 1e-6)
            cur_sys, cur_proc = self._sys_cpu_times(), time.process_time()
            s = {"t": now - self.t_start, "phase": self.phase_fn()}
            if dt >= 0.2:                                  # a very short last interval would be noisy
                if prev_sys and cur_sys and cur_sys[0] > prev_sys[0]:
                    s["sys_cpu"] = 100.0 * ((cur_sys[0] - prev_sys[0]) - (cur_sys[1] - prev_sys[1])) \
                                   / (cur_sys[0] - prev_sys[0])
                cores = (cur_proc - prev_proc) / dt
                s["proc_cores"], s["proc_pct"] = cores, 100.0 * cores / self.ncpu
            s["rss_mb"] = self._rss_mb()
            if self.ok:
                s.update(self._gpu_sample())
            self.samples.append(s)
            if dt >= 0.2:
                prev_sys, prev_proc, prev_t = cur_sys, cur_proc, now
            if stopped:
                break

    def stop(self):
        self._halt.set()

    def stat(self, field, phase=None):
        vals = [x[field] for x in self.samples
                if x.get(field) is not None and (phase is None or x["phase"] == phase)]
        if not vals:
            return None
        return dict(avg=float(np.mean(vals)), peak=float(np.max(vals)), min=float(np.min(vals)),
                    first=float(vals[0]), last=float(vals[-1]), n=len(vals))

    def phase_seconds(self, ph):
        ts = [x["t"] for x in self.samples if x["phase"] == ph]
        return (max(ts) - min(ts) + self.period) if ts else 0.0

    FIELDS = ("t", "phase", "sys_cpu", "proc_cores", "proc_pct", "rss_mb", "gpu_util",
              "gpu_mem_util", "enc", "dec", "vram_mb", "power_w", "temp_c", "sm_mhz")

    def write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "phase(0=init,1=startup,2=steady)", "system_cpu_pct",
                        "process_cpu_cores", "process_cpu_pct_of_machine", "process_rss_mb",
                        "gpu_busy_pct", "gpu_memctrl_busy_pct", "nvenc_pct", "nvdec_pct",
                        "gpu_mem_used_mb", "gpu_power_w", "gpu_temp_c", "gpu_sm_clock_mhz"])
            for s in self.samples:
                w.writerow([("" if s.get(k) is None else (round(s[k], 3) if isinstance(s[k], float) else s[k]))
                            for k in self.FIELDS])


def compute_code_split(path):
    """Static split of the executable lines of this file into GPU and CPU code.

    GPU-KERNEL = lines inside CUDA_SRC; GPU = Python functions listed in DEVICE_OF;
    CPU = everything else. Blank lines, comments and docstrings are not counted."""
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception:  # noqa: BLE001
        return None
    lines = src.splitlines()
    n = len(lines)
    module_owner = ("(module level: imports, constants, entry point)", "CPU")
    owner = {i: module_owner for i in range(1, n + 1)}
    excluded = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                excluded.update(range(body[0].lineno, body[0].end_lineno + 1))

    def first_line(node):
        return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in node.targets):
            for i in range(node.lineno + 1, node.end_lineno):
                owner[i] = ("CUDA kernels (CUDA_SRC)", "GPU-KERNEL")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for i in range(first_line(node), node.end_lineno + 1):
                owner[i] = (node.name, DEVICE_OF.get(node.name, "CPU"))
        elif isinstance(node, ast.ClassDef):
            for i in range(first_line(node), node.end_lineno + 1):
                owner[i] = (f"{node.name} (class header / constants)", "CPU")
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qn = f"{node.name}.{m.name}"
                    for i in range(first_line(m), m.end_lineno + 1):
                        owner[i] = (qn, DEVICE_OF.get(qn, "CPU"))

    per = OrderedDict()
    for i in range(1, n + 1):
        txt = lines[i - 1].strip()
        if not txt or i in excluded:
            continue
        name, dev = owner[i]
        if dev == "GPU-KERNEL":
            if txt.startswith("//"):
                continue
        elif txt.startswith("#"):
            continue
        e = per.setdefault(name, {"device": dev, "lines": 0})
        e["lines"] += 1

    total = sum(e["lines"] for e in per.values())
    gk = sum(e["lines"] for e in per.values() if e["device"] == "GPU-KERNEL")
    gp = sum(e["lines"] for e in per.values() if e["device"] == "GPU")
    cp_ = sum(e["lines"] for e in per.values() if e["device"] == "CPU")
    pct = lambda v: 100.0 * v / total if total else 0.0  # noqa: E731
    return dict(total=total, gpu_kernel=gk, gpu_python=gp, cpu=cp_,
                gpu_kernel_pct=pct(gk), gpu_python_pct=pct(gp), cpu_pct=pct(cp_),
                gpu_pct=pct(gk + gp), functions=per)


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for ln in f:
                if ln.lower().startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "n/a"


def _ram_total_gb():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemTotal:"):
                    return int(ln.split()[1]) / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return None


def build_optimization_report(app, K):
    """Builds the full text report and its JSON twin. `K` holds the numbers computed in App.report()."""
    a, prof, res = app.a, app.prof, app.res
    s, n = K["s"], K["n"]
    frames = max(prof.frames, 1)
    steady_fps = K["steady_fps"]
    L, J = [], {}
    out = L.append
    W = 104

    def hr(ch="="):
        out(ch * W)

    def sec(title):
        out("")
        hr("=")
        out(f" {title}")
        hr("=")

    def f(v, spec=".1f", na="n/a"):
        if v is None:
            return na
        try:
            if v != v:
                return na
            return format(v, spec)
        except Exception:  # noqa: BLE001
            return str(v)

    def st(field):
        return res.stat(field, 2) or res.stat(field)

    # ------------------------------------------------------------------ header
    hr("=")
    out(" OPTIMIZATION & RESOURCE REPORT  (CPU / GPU usage, function map, workload + code split)")
    hr("=")
    out(f" Generated   : {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    out(f" Script      : {os.path.basename(os.path.abspath(__file__))}")
    out(f" Source      : {str(app.uri)[:90]}  ({'live' if app.live else 'file'})")
    out(f" Model       : {a.model_name}  input {a.det_width}x{a.det_height}  conf {a.conf}  NMS {a.nms}  (TensorRT FP16, batch 1)")
    out(f" Frames      : {n} processed ({a.warmup} warm-up frames excluded from timings), run time {app.wall_total:.1f} s")
    out(f" Host        : {_cpu_model()} | {res.ncpu} logical cores | RAM {f(_ram_total_gb(), '.0f')} GB | {platform.platform()}")
    cuda_ver = None
    try:
        cuda_ver = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception:  # noqa: BLE001
        pass
    out(f" GPU         : {res.gpu_name} | {res.vram_total_mb} MB VRAM | driver {res.driver} | "
        f"CUDA runtime {f(cuda_ver, 'd')} | CuPy {getattr(cp, '__version__', 'n/a')}")
    saver_mode = app.saver.mode if app.saver is not None else "not used"
    pv_built = app.preview is not None and app.preview.built
    pvs = ("ON (DeepStream GPU branch" + ("" if app.preview.ok else ", stopped early") + ")") if pv_built else "off"
    out(f" Settings    : save-detections={'ON' if not a.no_save_detections else 'off'}  logs={'ON' if not a.no_logs else 'off'}  "
        f"preview={pvs}  JPEG={saver_mode}")
    out(f" Profiling   : function-level timing {'ON' if prof.enabled else 'OFF (--no-profile)'}; "
        f"resource sampling every {res.period:.2f} s ({len(res.samples)} samples)")
    out(" How to read : section 1 = headline numbers, 2 = CPU/GPU usage over time, 3 = which function runs where,")
    out("               4 = workload split (GPU % + CPU % = 100), 5 = code split (GPU % + CPU % = 100),")
    out("               6 = CPU time accounting, 7 = bottlenecks and hints, 8 = definitions and caveats.")

    # ------------------------------------------------------------------ stage table (needed early)
    stages = []
    other_wall = other_cpu = 0.0
    incl = prof.fn.get(INCLUSIVE)
    if prof.enabled:
        for name, d in prof.fn.items():
            if d["kind"] == "inclusive":
                continue
            is_wait = d["device"] == "WAIT"
            stages.append(dict(name=name, func=d["func"], dev=d["device"], kind=d["kind"],
                               gpu=d["gpu_ms"] / frames,
                               cpu=0.0 if is_wait else d["cpu"] * 1000.0 / frames,
                               wait=d["wall"] * 1000.0 / frames if is_wait else 0.0,
                               calls=d["calls"]))
        if incl:
            ch_wall = sum(d["wall"] for k, d in prof.fn.items() if d["in_proc"] and k != INCLUSIVE)
            ch_cpu = sum(d["cpu"] for k, d in prof.fn.items() if d["in_proc"] and k != INCLUSIVE)
            other_wall = max(incl["wall"] - ch_wall, 0.0)
            other_cpu = max(incl["cpu"] - ch_cpu, 0.0)
            stages.append(dict(name="python glue in the probe (hold logic, bookkeeping)",
                               func="App._process_impl", dev="CPU", kind="python",
                               gpu=0.0, cpu=other_cpu * 1000.0 / frames, wait=0.0, calls=incl["calls"]))
    det_ms = s["detect"]["mean"] if s["detect"]["n"] else 0.0
    enc_ms = s["encode"]["mean"] if s["encode"]["n"] else 0.0
    hw_rows = [
        dict(name="nvinfer: TensorRT inference (probe-to-probe)", func="pipeline: nvinfer", dev="GPU",
             kind="TensorRT (hardware stage)", gpu=det_ms, cpu=0.0, wait=0.0, calls=s["detect"]["n"]),
        dict(name="NVENC: video encode (probe-to-probe)", func="pipeline: nvv4l2h264enc", dev="GPU",
             kind="NVENC (hardware stage)", gpu=enc_ms, cpu=0.0, wait=0.0, calls=s["encode"]["n"]),
    ]
    # Live preview = DeepStream branch (GPU). It runs at --preview-fps, so its cost is spread over ALL frames:
    # per-frame cost = mean cost of a shown frame x (shown frames / processed frames).
    pv_v = app.st["preview"].v if pv_built else []
    pv_ms = (sum(pv_v) / len(pv_v)) * (getattr(app, "pv_shown", 0) / frames) if pv_v else 0.0
    if pv_v:
        hw_rows.append(dict(name="preview branch: nvvideoconvert + nvdsosd (probe-to-probe)",
                            func="pipeline: preview branch", dev="GPU",
                            kind="DeepStream preview (GPU stage)", gpu=pv_ms, cpu=0.0, wait=0.0, calls=len(pv_v)))
    all_rows = stages + hw_rows
    order = {"GPU": 0, "CPU": 1, "WAIT": 2}
    all_rows.sort(key=lambda r: (order.get(r["dev"], 3), -(r["gpu"] + r["cpu"] + r["wait"])))
    pv_cpu = sum(r["cpu"] for r in all_rows if r["name"].startswith("preview:") or "p_pv_" in r["name"])
    gpu_tot = sum(r["gpu"] for r in all_rows)
    cpu_tot = sum(r["cpu"] for r in all_rows)
    wait_tot = sum(r["wait"] for r in all_rows)
    busy = gpu_tot + cpu_tot
    have_split = prof.enabled and busy > 0
    gpu_pct = 100.0 * gpu_tot / busy if have_split else None
    cpu_pct = 100.0 * cpu_tot / busy if have_split else None
    hw_gpu = det_ms + enc_ms + pv_ms
    our_gpu = gpu_tot - hw_gpu
    our_busy = our_gpu + cpu_tot
    our_gpu_pct = 100.0 * our_gpu / our_busy if (have_split and our_busy > 0) else None

    probe_ms = incl["wall"] * 1000.0 / frames if incl else None
    probe_cpu_ms = incl["cpu"] * 1000.0 / frames if incl else None
    probe_busy_pct = probe_ms * steady_fps / 10.0 if (probe_ms and steady_fps) else None
    probe_limit_fps = 1000.0 / probe_ms if probe_ms else None

    code = compute_code_split(os.path.abspath(__file__))

    # ------------------------------------------------------------------ 1. headline numbers
    sec("1. HEADLINE NUMBERS (the metrics to watch while optimizing)")
    e2e, iv = s["e2e"], K["iv"]
    pw = st("power_w")
    energy_j = (pw["avg"] / steady_fps) if (pw and steady_fps) else None
    fps_per_core = steady_fps / K["cpu_cores"] if K["cpu_cores"] > 0 else None
    out(f" THROUGHPUT   steady-state FPS {steady_fps:.1f} | whole-run FPS {K['whole_fps']:.1f} | "
        f"estimated max FPS {K['est_max']:.1f} (limited by {K['slow']}) | headroom {K['headroom']:.1f}x")
    out(f"              frames at mux / processed / encoded: {app.received} / {n} / {app.encoded} | frames lost {K['dropped']}")
    out(f" LATENCY      end-to-end mux->file ms: mean {e2e['mean']:.1f}  p50 {e2e['p50']:.1f}  p95 {e2e['p95']:.1f}  max {e2e['max']:.1f}")
    out(f"              frame interval ms:      mean {iv.mean():.1f}  p50 {np.percentile(iv, 50):.1f}  p95 {np.percentile(iv, 95):.1f}  max {iv.max():.1f}")
    out(f" PROBE THREAD per-frame wall {f(probe_ms, '.2f')} ms (CPU {f(probe_cpu_ms, '.2f')} ms) -> serial limit "
        f"{f(probe_limit_fps, '.0f')} FPS | busy {f(probe_busy_pct, '.0f')} % of the time at the current FPS")
    gu, mu, en, de = st("gpu_util"), st("gpu_mem_util"), st("enc"), st("dec")
    vr, sm = res.stat("vram_mb"), st("sm_mhz")
    out(f" GPU          busy avg {f(gu['avg'] if gu else None, '.0f')} % / peak {f(gu['peak'] if gu else None, '.0f')} % | "
        f"mem-ctrl avg {f(mu['avg'] if mu else None, '.0f')} % | NVENC {f(en['avg'] if en else None, '.0f')} % | "
        f"NVDEC {f(de['avg'] if de else None, '.0f')} %")
    out(f"              VRAM peak {f(vr['peak'] if vr else None, '.0f')} MB of {res.vram_total_mb} | power avg {f(pw['avg'] if pw else None, '.0f')} W | "
        f"SM clock avg {f(sm['avg'] if sm else None, '.0f')} MHz | energy per frame {f(energy_j, '.3f')} J")
    sc_, pc_, pp_, rs_ = st("sys_cpu"), st("proc_cores"), st("proc_pct"), res.stat("rss_mb")
    out(f" CPU          this process {f(pc_['avg'] if pc_ else None, '.2f')} cores avg ({f(pp_['avg'] if pp_ else None, '.1f')} % of the machine) | "
        f"system CPU avg {f(sc_['avg'] if sc_ else None, '.0f')} % | RAM peak {f(rs_['peak'] if rs_ else None, '.0f')} MB | "
        f"FPS per CPU core {f(fps_per_core, '.0f')}")
    out(f" WORKLOAD     GPU {f(gpu_pct, '.1f')} %  /  CPU {f(cpu_pct, '.1f')} %   (measured busy time per frame, section 4)")
    out(f"              intern's own code only (without DeepStream stages: TensorRT, NVENC, preview): GPU {f(our_gpu_pct, '.1f')} % / CPU {f(100 - our_gpu_pct if our_gpu_pct is not None else None, '.1f')} %")
    out(f" CODE         GPU {f(code['gpu_pct'] if code else None, '.1f')} %  /  CPU {f(code['cpu_pct'] if code else None, '.1f')} %   (executable lines of this file, section 5)")
    up_kb = app.up_bytes / 1024.0
    out(f" HOST<->GPU   upload of model outputs {up_kb:.0f} KB/frame ({up_kb * steady_fps / 1024.0:.0f} MB/s) | control readback 28 bytes/frame")
    raw = K["raw"]
    out(f" DETECTION    faces/frame avg {raw.mean():.2f} / max {raw.max()} | frames with 0 faces {(raw == 0).mean() * 100:.1f} % | "
        f"held frames {app.held} | privacy audit {K['asum']['overall']}")

    # ------------------------------------------------------------------ 2. usage over time
    sec("2. CPU AND GPU USAGE: AT START, STEADY STATE, PEAK, AT FINISH")
    out(f" Phases (approx.): init (CUDA kernel build, model setup) {res.phase_seconds(0):.1f} s | "
        f"startup (engine load + warm-up frames) {res.phase_seconds(1):.1f} s | steady state {res.phase_seconds(2):.1f} s")
    out(" 'At start' = first sample ~0.25 s after the program began. 'Steady' = after the warm-up frames.")
    out(" NOTE: on the first run the startup phase includes the TensorRT engine build, which is CPU+GPU heavy.")
    out("")
    cols = ("At start", "Steady avg", "Steady peak", "Overall avg", "Overall peak", "At finish")
    out(f" {'Metric':<34}" + "".join(f"{c:>13}" for c in cols))
    hr("-")
    rows = [("System CPU, all cores %", "sys_cpu", ".0f"), ("This process CPU, cores", "proc_cores", ".2f"),
            ("This process CPU, % of machine", "proc_pct", ".1f"), ("Process RAM (RSS) MB", "rss_mb", ".0f"),
            ("GPU busy % (NVML)", "gpu_util", ".0f"), ("GPU memory-controller busy %", "gpu_mem_util", ".0f"),
            ("NVENC utilisation %", "enc", ".0f"), ("NVDEC utilisation %", "dec", ".0f"),
            ("GPU memory used MB", "vram_mb", ".0f"), ("GPU power W", "power_w", ".0f"),
            ("GPU temperature C", "temp_c", ".0f"), ("GPU SM clock MHz", "sm_mhz", ".0f")]
    usage_json = {}
    for label, key, spec in rows:
        a_all, a_ss = res.stat(key), res.stat(key, 2)
        vals = [a_all["first"] if a_all else None, a_ss["avg"] if a_ss else None, a_ss["peak"] if a_ss else None,
                a_all["avg"] if a_all else None, a_all["peak"] if a_all else None, a_all["last"] if a_all else None]
        out(f" {label:<34}" + "".join(f"{f(v, spec):>13}" for v in vals))
        usage_json[key] = dict(zip(("at_start", "steady_avg", "steady_peak", "overall_avg", "overall_peak", "at_finish"), vals))
    out(" (full time series: optimization_report_timeline.csv)")

    # ------------------------------------------------------------------ 3. function map
    sec("3. WHICH FUNCTION RUNS ON THE GPU AND WHICH ON THE CPU")
    out(" 3a. MEASURED STAGES of the per-frame path (steady state, milliseconds per frame)")
    out("     GPU ms = CUDA-event time on the GPU | CPU ms = thread CPU time | WAIT ms = host blocked waiting for the GPU")
    hr("-")
    if not prof.enabled:
        out(" Function-level profiling was switched off (--no-profile). Re-run without it to get this table.")
    else:
        out(f" {'Stage':<44}{'Function':<26}{'On':<5}{'GPU ms/f':>9}{'CPU ms/f':>9}{'WAIT ms/f':>10}")
        hr("-")
        for r in all_rows:
            out(f" {r['name'][:43]:<44}{r['func'][:25]:<26}{r['dev']:<5}{r['gpu']:>9.3f}{r['cpu']:>9.3f}{r['wait']:>10.3f}")
        hr("-")
        out(f" {'TOTAL':<44}{'':<26}{'':<5}{gpu_tot:>9.3f}{cpu_tot:>9.3f}{wait_tot:>10.3f}")
        out(" Not timed one by one (hardware blocks inside DeepStream): NVDEC decode, nvstreammux, nvvideoconvert x2,")
        out(" and the EGL window render of the preview sink. Their load is visible in the NVDEC % / GPU busy % rows of section 2.")
    out("")
    out(" 3b. PIPELINE ELEMENTS (DeepStream / GStreamer plugins, not Python code)")
    hr("-")
    for el, where in PIPELINE_MAP:
        out(f" {el:<46}{where}")
    out("")
    out(" 3c. EVERY FUNCTION IN THIS FILE (static) - where its work executes")
    out("     'GPU' = the work of the function runs on the GPU (the Python lines inside only enqueue it).")
    hr("-")
    if code:
        out(f" {'Function':<58}{'Runs on':<12}{'Lines':>7}{'% of code':>11}")
        hr("-")
        fl = sorted(code["functions"].items(),
                    key=lambda kv: ({"GPU-KERNEL": 0, "GPU": 1, "CPU": 2}.get(kv[1]["device"], 3), -kv[1]["lines"]))
        for name, e in fl:
            out(f" {name[:57]:<58}{e['device']:<12}{e['lines']:>7}{100.0 * e['lines'] / code['total']:>10.1f}%")
    else:
        out(" (source file could not be read for the static analysis)")

    # ------------------------------------------------------------------ 4. workload split
    sec("4. WORKLOAD SPLIT: HOW MUCH OF THE WORK RUNS ON THE GPU AND ON THE CPU (= 100 %)")
    if have_split:
        out(f" Basis: measured busy time per frame in steady state = {busy:.3f} ms  (GPU {gpu_tot:.3f} ms + CPU {cpu_tot:.3f} ms)")
        out("")
        out(f"   GPU : {gpu_pct:6.1f} %   {gpu_tot:7.3f} ms/frame")
        out(f"           of which DeepStream GPU stages (TensorRT, NVENC, preview) {100.0 * hw_gpu / busy:5.1f} %  ({hw_gpu:.3f} ms)")
        out(f"           of which this program's CUDA kernels / copies / nvJPEG {100.0 * our_gpu / busy:5.1f} %  ({our_gpu:.3f} ms)")
        out(f"   CPU : {cpu_pct:6.1f} %   {cpu_tot:7.3f} ms/frame")
        out(f"   ----------------------------")
        out(f"   TOTAL {gpu_pct + cpu_pct:5.1f} %")
        out("")
        out(f"   Host waiting for the GPU (counted in neither): {wait_tot:.3f} ms/frame")
        out(f"   Only this program's own code (DeepStream stages excluded): GPU {our_gpu_pct:.1f} % / CPU {100 - our_gpu_pct:.1f} %")
        if pv_built:
            out("")
            out(f"   Live preview (DeepStream branch): GPU {pv_ms:.3f} ms/frame  (nvvideoconvert + nvdsosd; the EGL render in the sink is not timed)")
            out(f"                                     CPU {pv_cpu:.3f} ms/frame  (box readback + OSD metadata + gate/stat probes)")
            out("   Before, the preview was OpenCV: a full-frame GPU->CPU copy, resize and drawing on the CPU. That work now runs on")
            out("   the GPU, so the GPU share above is higher and the CPU share lower than with the OpenCV preview. Pixels no longer")
            out("   cross to the CPU; only the face boxes (N x 5 floats) do, to build the OSD metadata.")
        out(" Caveats: stages overlap in time (the pipeline is multi-threaded), so this is a split of work, not of wall-clock")
        out("          latency. NVDEC, nvstreammux and nvvideoconvert are GPU work too but are not in this total.")
    else:
        out(" Not available (function-level profiling is off or no frames were profiled).")

    # ------------------------------------------------------------------ 5. code split
    sec("5. CODE SPLIT: HOW MUCH OF THE CODE RUNS ON THE GPU AND ON THE CPU (= 100 %)")
    if code:
        out(f" Basis: {code['total']} executable lines in this file (blank lines, comments and docstrings not counted)")
        out("")
        out(f"   GPU : {code['gpu_pct']:6.1f} %   ({code['gpu_kernel'] + code['gpu_python']} lines)")
        out(f"           CUDA kernels (CUDA_SRC)                {code['gpu_kernel_pct']:5.1f} %  ({code['gpu_kernel']} lines)")
        out(f"           Python functions whose work is on GPU  {code['gpu_python_pct']:5.1f} %  ({code['gpu_python']} lines)")
        out(f"   CPU : {code['cpu_pct']:6.1f} %   ({code['cpu']} lines)   argument parsing, GStreamer pipeline, probes, logging, report, preview, ...")
        out(f"   ----------------------------")
        out(f"   TOTAL {code['gpu_pct'] + code['cpu_pct']:5.1f} %")
        out("")
        out(" The preview window is rendered by DeepStream plugins (GPU); only its control code is Python, so it counts as CPU lines.")
        out(" Why this is not the same as section 4: most lines of a DeepStream program are setup, plumbing and reporting")
        out(" that run once or cheaply; the few GPU lines (kernels) do almost all of the compute. Use section 4 to judge")
        out(" the compute split and this section only to judge how much of the code base is GPU code.")
    else:
        out(" Not available (source file could not be read).")

    # ------------------------------------------------------------------ 6. CPU time accounting
    sec("6. WHERE THE CPU TIME GOES (steady state)")
    window_cpu = K.get("cpu_window_s")
    attributed = sum(d["cpu"] for k, d in prof.fn.items() if d["kind"] != "inclusive" and d["device"] != "WAIT") + other_cpu
    wait_cpu = sum(d["cpu"] for d in prof.fn.values() if d["device"] == "WAIT")
    cpu_acc = None
    if prof.enabled and window_cpu and window_cpu > 0:
        unattr = max(window_cpu - attributed - wait_cpu, 0.0)
        pct = lambda v: 100.0 * v / window_cpu  # noqa: E731
        out(f" Process CPU time in the measured window : {window_cpu:.2f} s  = {K['cpu_cores']:.2f} cores on average")
        out(f"   attributed to Python stages above      : {attributed:7.2f} s  {pct(attributed):5.1f} %")
        out(f"   spent spinning while waiting for GPU   : {wait_cpu:7.2f} s  {pct(wait_cpu):5.1f} %")
        out(f"   not attributed (GStreamer threads, CUDA/NVML drivers, DeepStream plugins, monitor threads): "
            f"{unattr:.2f} s  {pct(unattr):5.1f} %")
        cpu_acc = dict(window_cpu_s=window_cpu, attributed_s=attributed, spin_wait_s=wait_cpu, unattributed_s=unattr)
    else:
        out(" Not available (profiling off or window too short).")

    # ------------------------------------------------------------------ 7. bottlenecks + hints
    sec("7. BOTTLENECKS AND OPTIMIZATION HINTS (generated from the numbers above)")
    hints = []
    top = sorted([r for r in all_rows if r["dev"] != "WAIT"], key=lambda r: -(r["gpu"] + r["cpu"]))[:5]
    if have_split:
        out(" Top 5 cost items (ms per frame, GPU + CPU):")
        for r in top:
            out(f"   {r['gpu'] + r['cpu']:7.3f} ms  {r['dev']:<4} {r['name'][:80]}")
        out("")
    extra_keys = ("preview", "jpeg", "logs", "copy frame", "copy original", "concat", "resize")
    extras_ms = sum(r["gpu"] + r["cpu"] for r in all_rows if any(k in r["name"] for k in extra_keys))
    if have_split and extras_ms > 0.25 * busy:
        hints.append(f"Per-frame extras (logs, JPEG saving, preview, frame copies) cost {extras_ms:.2f} ms/frame = "
                     f"{100 * extras_ms / busy:.0f} % of busy time. For a pure speed test use --no-save-detections --no-logs "
                     f"and no --preview.")
    if have_split and cpu_pct > 30:
        c = max((r for r in all_rows if r["dev"] == "CPU"), key=lambda r: r["cpu"], default=None)
        hints.append(f"CPU takes {cpu_pct:.0f} % of the busy time; largest CPU item: "
                     f"'{c['name']}' ({c['cpu']:.2f} ms/frame)." if c else f"CPU takes {cpu_pct:.0f} % of the busy time.")
    if probe_busy_pct is not None and probe_busy_pct > 80:
        hints.append(f"The Python probe thread is {probe_busy_pct:.0f} % busy at {steady_fps:.0f} FPS. It runs one frame at a "
                     f"time, so it caps the pipeline at about {probe_limit_fps:.0f} FPS. Reduce per-frame Python work.")
    if gu and gu["avg"] < 50 and not (app.live and K["headroom"] >= 5):
        hints.append(f"GPU is only {gu['avg']:.0f} % busy: the pipeline is limited by CPU / launch overhead / I/O / the source, "
                     f"not by GPU compute.")
    if app.live and K["headroom"] >= 5:
        hints.append("Live source: FPS equals what the camera/stream delivers; the pipeline has spare capacity "
                     f"(headroom {K['headroom']:.1f}x). Use a file source to measure maximum throughput.")
    if wait_tot > 0.3 * (busy if busy > 0 else 1) and have_split:
        hints.append(f"The host waits for the GPU {wait_tot:.2f} ms/frame (sync points after post-processing and after blur). "
                     f"Overlapping them with the next frame's work (double buffering) would hide this.")
    if app.saver is not None and app.saver.nv_enc is None:
        hints.append("JPEG saving uses the CPU fallback. Install nvidia-nvimgcodec-cu12 to encode on the GPU.")
    if K["dropped"] > 0:
        hints.append(f"{K['dropped']} frames were lost between mux and the probe: the pipeline did not keep up at times.")
    if "detect" in str(K["slow"]):
        hints.append("TensorRT inference is the slowest stage: smaller input size, INT8 calibration or a lighter backbone "
                     "(RetinaFace-ResNet18 / MobileNet) would speed it up.")
    hints.append(f"Model outputs make a round trip GPU -> host (done inside nvinfer) -> GPU: {up_kb:.0f} KB per frame. "
                 f"Removing it needs a custom parser or device-side tensor pointers.")
    for i, h in enumerate(hints, 1):
        out(f" {i}. {h}")

    # ------------------------------------------------------------------ 8. definitions
    sec("8. METRIC DEFINITIONS AND CAVEATS")
    for ln in (
        "GPU busy % (NVML)      share of each sample period in which at least one kernel was running. It is NOT SM occupancy:",
        "                       100 % can still be a poorly filled GPU, and a low value means the GPU is waiting for work.",
        "GPU ms (stages)        CUDA events recorded before/after each stage on the GPU timeline: kernel time plus tiny gaps.",
        "CPU ms (stages)        time.thread_time(): CPU seconds used by that thread. Blocked/sleeping time is excluded.",
        "WAIT ms                wall time the host thread was blocked on the GPU (readback, stream sync). Not CPU work, not GPU work.",
        "nvinfer / NVENC ms     probe-to-probe wall time inside the pipeline; includes internal queueing and their own threads.",
        "nvJPEG encode          measured as host wall time of the call; split into GPU part (wall - CPU) and CPU part.",
        "preview branch ms      probe-to-probe wall time of nvvideoconvert + nvdsosd for the frames shown; spread over all frames",
        "                       because the preview runs at --preview-fps only. The EGL render inside the sink is not included.",
        "Workload split         busy time per frame, steady state only. GPU % + CPU % = 100 %; WAIT is reported separately.",
        "Code split             executable lines per function (ast). A line count is not a compute share (see section 5).",
        "Estimated max FPS      1000 / slowest stage mean (theoretical, assumes perfect overlap). Steady-state FPS is the measured one.",
        "Probe thread limit     the Python probe handles one frame at a time; 1000 / its per-frame wall time is a hard FPS ceiling.",
        "Profiling overhead     CUDA events and timers add a small cost per frame (typically a few percent at several hundred FPS).",
        "                       Use --no-profile for a pure speed benchmark; resource sampling alone is negligible.",
        "Sampling               CPU/GPU usage is sampled every --prof-period seconds; short spikes between samples are not seen.",
    ):
        out(" " + ln)
    out("")
    hr("=")
    out(" END OF REPORT")
    hr("=")

    # ------------------------------------------------------------------ JSON twin
    J.update(
        generated=datetime.now().astimezone().isoformat(timespec="seconds"),
        script=os.path.basename(os.path.abspath(__file__)),
        frames=n, profiled_frames=prof.frames, run_seconds=app.wall_total,
        host=dict(cpu=_cpu_model(), logical_cores=res.ncpu, ram_gb=_ram_total_gb(), os=platform.platform()),
        gpu=dict(name=res.gpu_name, vram_mb=res.vram_total_mb, driver=res.driver, cuda_runtime=cuda_ver),
        headline=dict(steady_fps=steady_fps, whole_run_fps=K["whole_fps"], estimated_max_fps=K["est_max"],
                      bottleneck=K["slow"], headroom=K["headroom"], frames_lost=K["dropped"],
                      e2e_ms=e2e, frame_interval_ms=dict(mean=float(iv.mean()), p50=float(np.percentile(iv, 50)),
                                                         p95=float(np.percentile(iv, 95)), max=float(iv.max())),
                      probe_wall_ms=probe_ms, probe_cpu_ms=probe_cpu_ms, probe_limit_fps=probe_limit_fps,
                      probe_busy_pct=probe_busy_pct, energy_per_frame_j=energy_j, fps_per_cpu_core=fps_per_core,
                      upload_kb_per_frame=up_kb, jpeg_saver=saver_mode),
        usage=usage_json,
        workload_split=dict(gpu_pct=gpu_pct, cpu_pct=cpu_pct, busy_ms_per_frame=busy, gpu_ms=gpu_tot,
                            cpu_ms=cpu_tot, host_wait_ms=wait_tot, own_code_gpu_pct=our_gpu_pct,
                            hardware_gpu_ms=hw_gpu, preview_gpu_ms=pv_ms, preview_cpu_ms=pv_cpu,
                            preview_mode=("DeepStream GPU branch" if pv_built else "off")),
        stages=[dict(r) for r in all_rows],
        code_split=(None if not code else dict(
            total_lines=code["total"], gpu_pct=code["gpu_pct"], cpu_pct=code["cpu_pct"],
            gpu_kernel_lines=code["gpu_kernel"], gpu_python_lines=code["gpu_python"], cpu_lines=code["cpu"],
            functions={k: v for k, v in code["functions"].items()})),
        cpu_time_accounting=cpu_acc,
        hints=hints)
    return "\n".join(L), J


# ----------------------------------------------------------------------------------
# RetinaFace post-processing ON THE GPU
#
# Model outputs (biubug6 / Pytorch_Retinaface style export), each with N rows:
#     loc     (N, 4)   box regression  (dcx, dcy, dw, dh)
#     conf    (N, 2)   [background, face]  (logits or already softmaxed)
#     landms  (N, 10)  5 landmarks (not needed for blurring, never uploaded to the GPU)
# Priors (anchors) per stride 8 / 16 / 32 with min sizes (16,32) / (64,128) / (256,512).
# Variances (0.1, 0.2).
#
# Per frame everything below runs as GPU kernels with FIXED array sizes (no host-side
# data-dependent shapes), so there is exactly one small host readback at the end.
# ----------------------------------------------------------------------------------
class RetinaFaceDecoder:
    STEPS = (8, 16, 32)
    MIN_SIZES = ((16, 32), (64, 128), (256, 512))
    VARIANCES = (0.1, 0.2)
    MAX_CANDIDATES = 1000            # same cap as before: best 1000 boxes go into NMS

    def __init__(self, in_w, in_h, conf, nms_iou, mod):
        self.in_w, self.in_h = in_w, in_h
        self.conf, self.nms_iou = conf, nms_iou
        self.prof = None                                           # set by App (profiling)
        # ---- one-time anchor table (start-up only, not per frame) ----
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

        # ---- GPU state ----
        self.priors_g = cp.asarray(np.ascontiguousarray(self.priors))
        self.K = min(self.MAX_CANDIDATES, self.n_priors)
        self.nwords = (self.K + 63) // 64
        self.mask_g = cp.empty((self.K, self.nwords), dtype=cp.uint64)
        self.k_decode = mod.get_function("rf_decode")
        self.k_nms_mask = mod.get_function("nms_mask")
        self.k_nms_reduce = mod.get_function("nms_reduce")

    # ---- one-time, first frame only: match engine outputs to loc / conf (host arrays) ----
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

    # ---- per frame, all on the GPU ----
    def process_gpu(self, loc_g, cf_g, frame_w, frame_h, pad, blur_frac):
        """loc_g (N*4,) and cf_g (N*2,) float32 CuPy arrays.

        Returns (det_g, rois_g, n, m, maxw, maxh, conf_stats)
          det_g   (K, 5) float32 GPU  [x1, y1, x2, y2, score], first n rows valid
          rois_g  (K+1, 5) int32 GPU  [x1, y1, x2, y2, blur_radius], first m rows valid
          n       number of detected faces            (host int, from the 7-number readback)
          m       number of blur regions
          maxw/maxh  largest blur region (used for the CUDA launch grid)
          conf_stats  {"min","mean","max"} of the detections, or None
        """
        prof = self.prof or NULL_PROF
        FN = "RetinaFaceDecoder.process_gpu"
        N, K, nwords = self.n_priors, self.K, self.nwords
        scale = min(self.in_w / frame_w, self.in_h / frame_h)        # nvinfer letterbox
        v0, v1 = self.VARIANCES

        # 1) decode every anchor (softmax, box regression, letterbox undo, clip, size filter)
        score = cp.empty(N, dtype=cp.float32)
        boxes = cp.empty((N, 4), dtype=cp.float32)
        threads = 256
        with prof.gpu("rf_decode kernel (softmax + box decode)", "CUDA kernel", FN):
            self.k_decode(((N + threads - 1) // threads,), (threads,),
                          (loc_g, cf_g, self.priors_g, np.int32(N), np.int32(1 if self.softmax else 0),
                           np.float32(v0), np.float32(v1),
                           np.float32(self.in_w), np.float32(self.in_h), np.float32(scale),
                           np.float32(frame_w), np.float32(frame_h), score, boxes),
                          )

        # 2) best K by score; count how many pass the threshold (stays on the GPU)
        with prof.gpu("top-K sort + gather (CuPy argsort)", "CuPy ops", FN):
            order = cp.argsort(-score)[:K]
            sc = score[order]
            bx = cp.ascontiguousarray(boxes[order])
            n_dev = (sc >= self.conf).sum(dtype=cp.int32).reshape(1)

        # 3) NMS: pairwise IoU bitmask, then greedy pass in one block
        with prof.gpu("nms_mask kernel (pairwise IoU)", "CUDA kernel", FN):
            self.k_nms_mask(((K + 63) // 64, nwords), (64,),
                            (bx, n_dev, np.int32(nwords), np.float32(self.nms_iou), self.mask_g))
        with prof.gpu("nms_reduce kernel (greedy keep)", "CUDA kernel", FN):
            keep_idx = cp.zeros(K, dtype=cp.int32)
            keep_cnt = cp.zeros(1, dtype=cp.int32)
            self.k_nms_reduce((1,), (32,), (self.mask_g, n_dev, np.int32(nwords), keep_idx, keep_cnt),
                              shared_mem=nwords * 8)

        # 4) ROIs: pad, clip, blur radius, compaction (all fixed-shape GPU ops)
        with prof.gpu("ROI build + compaction (CuPy ops)", "CuPy ops", FN):
            det = cp.concatenate([bx[keep_idx], sc[keep_idx][:, None]], axis=1)        # (K, 5)
            ar = cp.arange(K, dtype=cp.int32)
            b = det[:, :4].astype(cp.float64)
            bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            x1 = cp.floor(cp.clip(b[:, 0] - bw * pad, 0, frame_w - 1)).astype(cp.int32)
            y1 = cp.floor(cp.clip(b[:, 1] - bh * pad, 0, frame_h - 1)).astype(cp.int32)
            x2 = cp.ceil(cp.clip(b[:, 2] + bw * pad, 1, frame_w)).astype(cp.int32)
            y2 = cp.ceil(cp.clip(b[:, 3] + bh * pad, 1, frame_h)).astype(cp.int32)
            w, h = x2 - x1, y2 - y1
            r = cp.clip((blur_frac * cp.maximum(w, h)).astype(cp.int32), 3, 60)
            rois = cp.stack([x1, y1, x2, y2, r], axis=1).astype(cp.int32)
            valid = (ar < keep_cnt) & (w >= 2) & (h >= 2)
            pos = cp.cumsum(valid, dtype=cp.int32) - 1
            target = cp.where(valid, pos, K).astype(cp.int32)        # invalid rows -> dump row K
            rois_g = cp.zeros((K + 1, 5), dtype=cp.int32)
            rois_g[target] = rois
            m_dev = valid.sum(dtype=cp.int32)
            maxw_dev = (rois_g[:K, 2] - rois_g[:K, 0]).max()
            maxh_dev = (rois_g[:K, 3] - rois_g[:K, 1]).max()

        # 5) confidence statistics for the log (GPU reduction) + pack the control numbers
        with prof.gpu("confidence stats + pack (CuPy ops)", "CuPy ops", FN):
            vn = ar < keep_cnt
            cs = det[:, 4]
            cmin = cp.where(vn, cs, cp.float32(np.inf)).min()
            cmax = cp.where(vn, cs, cp.float32(-np.inf)).max()
            cmean = (cs * vn).sum() / cp.maximum(keep_cnt[0], 1)
            meta_dev = cp.stack([keep_cnt[0].astype(cp.float32), m_dev.astype(cp.float32),
                                 maxw_dev.astype(cp.float32), maxh_dev.astype(cp.float32),
                                 cmin.astype(cp.float32), cmean.astype(cp.float32),
                                 cmax.astype(cp.float32)])

        # 6) THE one host readback: 7 floats (28 bytes). It is control data: how many faces,
        #    how many blur regions and the CUDA grid size. No pixels, no boxes cross to the CPU.
        with prof.wait("wait: readback of 7 control numbers", "D2H copy + sync", FN):
            meta = meta_dev.get()
        n, m, maxw, maxh = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])
        conf_stats = None
        if n:
            conf_stats = {"min": round(float(meta[4]), 4), "mean": round(float(meta[5]), 4),
                          "max": round(float(meta[6]), 4)}
        return det, rois_g, n, m, maxw, maxh, conf_stats

    def warm(self, pad, blur_frac):
        """Run the whole GPU post-processing chain once on dummy data (compiles CuPy kernels)."""
        loc = cp.zeros(self.n_priors * 4, dtype=cp.float32)
        cf = cp.zeros(self.n_priors * 2, dtype=cp.float32)
        self.process_gpu(loc, cf, self.in_w, self.in_h, pad, blur_frac)
        cp.cuda.Stream.null.synchronize()


# ----------------------------------------------------------------------------------
# JPEG saving: GPU encode (nvJPEG through nvImageCodec) + background file writer
# ----------------------------------------------------------------------------------
class JpegSaver:
    def __init__(self, use_gpu=True, quality=85, max_queue=256):
        self.quality = int(quality)
        self.prof = None                                           # set by App (profiling)
        self.q = queue.Queue(maxsize=max_queue)
        self.nv_enc = self.nv_mod = self.nv_params = None
        if use_gpu:
            self._init_gpu()
        self.mode = ("GPU (nvJPEG via nvImageCodec)" if self.nv_enc is not None
                     else "CPU fallback (OpenCV, background thread)")
        print(f"JPEG saving: {self.mode}")
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    @staticmethod
    def _encode(enc, mod, params, rgb_gpu):
        return bytes(enc.encode(mod.as_image(rgb_gpu), "jpeg", params=params))

    def _init_gpu(self):
        try:
            from nvidia import nvimgcodec
            enc = nvimgcodec.Encoder()
            try:
               params = nvimgcodec.EncodeParams(
               quality_type=nvimgcodec.QualityType.QUALITY,   # JPEG quality scale 0-100
               quality_value=float(self.quality))
            except TypeError:                                       # older nvImageCodec versions
               params = nvimgcodec.EncodeParams(quality=self.quality)
            red = cp.zeros((16, 16, 3), dtype=cp.uint8)
            red[:, :, 0] = 255                                       # pure RED in RGB order
            data = self._encode(enc, nvimgcodec, params, red)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)    # BGR
            if img is None or not (int(img[8, 8, 2]) > 200 and int(img[8, 8, 0]) < 60):
                raise RuntimeError("start-up colour check failed (red came out wrong)")
            self.nv_enc, self.nv_mod, self.nv_params = enc, nvimgcodec, params
        except Exception as e:  # noqa: BLE001
            print(f"[note] GPU JPEG encoder not available ({str(e)[:120]}). "
                  f"Install it with: pip install nvidia-nvimgcodec-cu12  -> using the CPU fallback.")

    def save(self, path, rgb_gpu):
        """rgb_gpu: contiguous (H, W, 3) uint8 CuPy array in RGB order."""
        p = self.prof or NULL_PROF
        if self.nv_enc is not None:
            try:
                t0, c0 = time.perf_counter(), time.thread_time()
                data = self._encode(self.nv_enc, self.nv_mod, self.nv_params, rgb_gpu)
                wall, cpu = time.perf_counter() - t0, time.thread_time() - c0
                p.add("JPEG encode (nvJPEG, host wall time)", "GPU", "nvJPEG (GPU part = wall - CPU part)",
                      "JpegSaver._encode", wall=wall, cpu=cpu, gpu_ms=max(wall - cpu, 0.0) * 1000.0, in_proc=True)
                self.q.put((path, data, None))
                return
            except Exception as e:  # noqa: BLE001
                print(f"[warn] GPU JPEG encode failed ({str(e)[:120]}); switching to the CPU fallback.")
                self.nv_enc = None
        t0, c0 = time.perf_counter(), time.thread_time()
        host_bgr = np.ascontiguousarray(cp.asnumpy(rgb_gpu)[:, :, ::-1])
        self.q.put((path, None, host_bgr))
        p.add("JPEG CPU fallback (D2H copy + queue)", "CPU", "D2H copy + python", "JpegSaver.save",
              wall=time.perf_counter() - t0, cpu=time.thread_time() - c0, in_proc=True)

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            path, data, img = item
            t0, c0 = time.perf_counter(), time.thread_time()
            try:
                if data is not None:
                    with open(path, "wb") as f:
                        f.write(data)
                else:
                    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            except Exception as e:  # noqa: BLE001
                print(f"[warn] could not write {path}: {e}")
            (self.prof or NULL_PROF).add("JPEG file write (background thread)", "CPU", "file I/O (bg thread)",
                                         "JpegSaver._run", wall=time.perf_counter() - t0,
                                         cpu=time.thread_time() - c0, in_proc=False)

    def close(self):
        self.q.put(None)
        self.th.join(timeout=60)


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
# live preview ON THE GPU: a DeepStream branch of the main pipeline (no OpenCV, no appsrc)
# ----------------------------------------------------------------------------------
class GpuPreview:
    """Live preview that stays on the GPU.

      ... conv1 -> caps1 (RGBA, already blurred by the probe) -> tee -+-> q3 -> conv2 -> NVENC -> file  (unchanged)
                                                                      |
                                                                      +-> queue (leaky) -> nvvideoconvert (GPU scale)
                                                                          -> nvdsosd (GPU draws boxes + FPS text)
                                                                          -> nveglglessink (GPU / EGL window)

    * Pixels never leave the GPU. Only the face boxes (N x 5 floats) are read back to build the
      on-screen-display metadata, which is control data.
    * nvvideoconvert always scales into a SEPARATE surface. nvdsosd draws in place, and tee shares
      the buffer between both branches, so without this copy the boxes would be burned into the
      saved video. The saved video is never touched by the preview.
    * A gate probe on the preview branch drops every frame above --preview-fps before it can cost
      any GPU work, and the leaky queue means the preview can never slow the main pipeline down.
    * If the window is closed (or the display fails) the branch is removed and the run continues.
    """
    MAX_RECTS = 16                                   # NVDS_MAX_ELEMENTS_IN_DISPLAY_META

    def __init__(self, max_w, max_h, max_fps, draw_boxes, sink_name, gpu_id):
        self.max_w, self.max_h = int(max_w), int(max_h)
        self.min_gap = 1.0 / max(float(max_fps), 1.0)
        self.draw_boxes, self.sink_name, self.gpu_id = draw_boxes, sink_name, int(gpu_id)
        self.ok = True                  # False = preview switched off (window closed / error)
        self.built = False
        self.pipeline = self.tee = self.conv = self.caps = self.osd = self.sink = None
        self.gate_pad = None
        self.elements = []
        self.W0 = self.H0 = 0           # frame size the branch was built for
        self.pw = self.ph = 0           # preview surface size
        self.sx = self.sy = 1.0         # frame -> preview coordinate scale
        self.last = 0.0

    def due(self, now):
        if not self.ok or now - self.last < self.min_gap:
            return False
        self.last = now
        return True

    def fit(self, W, H):
        """Preview surface size: fit inside max_w x max_h, even numbers, never upscale."""
        sc = min(self.max_w / float(W), self.max_h / float(H), 1.0)
        pw = max(2, (int(W * sc) // 2) * 2)
        ph = max(2, (int(H * sc) // 2) * 2)
        if pw >= W and ph >= H:         # nothing to scale: force a tiny real downscale so that
            pw = max(2, ((W - 2) // 2) * 2)   # nvvideoconvert cannot pass the shared buffer through
            ph = max(2, ((H - 2) // 2) * 2)
        self.pw, self.ph = pw, ph
        self.sx, self.sy = pw / float(W), ph / float(H)
        return pw, ph

    @staticmethod
    def caps_string(pw, ph):
        return f"video/x-raw(memory:NVMM), format=RGBA, width={pw}, height={ph}"

    def preflight(self):
        """Returns None if the display sink can be created and opened, else a reason string."""
        try:
            el = Gst.ElementFactory.make(self.sink_name, "pv_preflight")
            if el is None:
                return (f"GStreamer element '{self.sink_name}' was not found "
                        f"(is DeepStream installed with display support?)")
            ret = el.set_state(Gst.State.READY)
            el.set_state(Gst.State.NULL)
            if ret == Gst.StateChangeReturn.FAILURE:
                return f"'{self.sink_name}' could not open the display (is a desktop session / EGL available?)"
            return None
        except Exception as e:  # noqa: BLE001
            return f"preview pre-check failed ({e})"

    def build(self, pipeline, upstream, main_head, unified, W0, H0):
        """Insert tee after `upstream`, keep `main_head` as the first (main) branch, add the preview branch."""
        mk = Gst.ElementFactory.make
        tee, q, conv = mk("tee", "pv_tee"), mk("queue", "pv_queue"), mk("nvvideoconvert", "pv_conv")
        caps, osd, sink = mk("capsfilter", "pv_caps"), mk("nvdsosd", "pv_osd"), mk(self.sink_name, "pv_sink")
        names = ("tee", "queue", "nvvideoconvert", "capsfilter", "nvdsosd", self.sink_name)
        els = (tee, q, conv, caps, osd, sink)
        missing = [n for n, e in zip(names, els) if e is None]
        if missing:
            print(f"[note] --preview is OFF: missing GStreamer element(s) {missing}.")
            return False

        self.W0, self.H0 = int(W0), int(H0)
        pw, ph = self.fit(self.W0, self.H0)
        setp(q, "max-size-buffers", 2)
        setp(q, "max-size-bytes", 0)
        setp(q, "max-size-time", 0)
        setp(q, "leaky", 2)                          # drop old frames, never block the main pipeline
        setp(conv, "nvbuf-memory-type", unified)
        setp(conv, "gpu-id", self.gpu_id)
        caps.set_property("caps", Gst.Caps.from_string(self.caps_string(pw, ph)))
        setp(osd, "process-mode", 1)                 # 1 = draw on the GPU
        setp(osd, "gpu-id", self.gpu_id)
        setp(sink, "sync", False)                    # show frames immediately
        setp(sink, "async", False)
        setp(sink, "qos", False)
        setp(sink, "window-width", pw)
        setp(sink, "window-height", ph)
        for el in els:
            pipeline.add(el)
        linked = (upstream.link(tee) and tee.link(main_head) and tee.link(q) and q.link(conv)
                  and conv.link(caps) and caps.link(osd) and osd.link(sink))
        if not linked:
            print("[note] --preview is OFF: could not link the preview branch.")
            for el in els:
                pipeline.remove(el)                  # removing also unlinks
            return False
        self.pipeline, self.tee, self.conv, self.caps, self.osd, self.sink = pipeline, tee, conv, caps, osd, sink
        self.elements = [q, conv, caps, osd, sink]
        self.gate_pad = q.get_static_pad("sink").get_peer()      # the tee source pad of this branch
        self.built = True
        print(f"Live preview (DeepStream GPU branch): nvvideoconvert -> nvdsosd -> {self.sink_name}, "
              f"{pw}x{ph} (max {self.max_w}x{self.max_h}), {1.0 / self.min_gap:.0f} fps")
        return True

    def set_frame_size(self, W, H):
        """Called on the first frame when the real frame size differs from the size given at build time."""
        pw, ph = self.fit(int(W), int(H))
        self.W0, self.H0 = int(W), int(H)
        self.caps.set_property("caps", Gst.Caps.from_string(self.caps_string(pw, ph)))
        setp(self.sink, "window-width", pw)
        setp(self.sink, "window-height", ph)
        print(f"Live preview size adjusted to the real frame size: {pw}x{ph}")

    def attach_osd(self, batch_meta, frame_meta, det, fps):
        """Attach on-screen-display metadata (boxes + FPS text) to the frame; nvdsosd draws it on the GPU.
        det: (N, 5) numpy array [x1, y1, x2, y2, score] in frame pixels."""
        sx, sy = self.sx, self.sy
        rows = det if (self.draw_boxes and len(det)) else []
        chunks = [rows[i:i + self.MAX_RECTS] for i in range(0, len(rows), self.MAX_RECTS)] or [[]]
        for ci, chunk in enumerate(chunks):
            dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            dm.num_rects = len(chunk)
            for i, (x1, y1, x2, y2, _s) in enumerate(chunk):
                r = dm.rect_params[i]
                r.left, r.top = float(x1) * sx, float(y1) * sy
                r.width, r.height = float(x2 - x1) * sx, float(y2 - y1) * sy
                r.border_width = 2
                r.border_color.set(0.0, 1.0, 0.0, 1.0)
                r.has_bg_color = 0
            if ci == 0:
                dm.num_labels = 1
                t = dm.text_params[0]
                t.display_text = f"FPS: {fps:.1f}"
                t.x_offset, t.y_offset = 10, 10
                t.font_params.font_name = "Serif"
                t.font_params.font_size = 12
                t.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                t.set_bg_clr = 1
                t.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
            pyds.nvds_add_display_meta_to_frame(frame_meta, dm)

    def owns(self, el):
        """True if a GStreamer element (or one of its parents) belongs to the preview branch."""
        while el is not None:
            if el in self.elements:
                return True
            el = el.get_parent()
        return False

    def remove_branch(self):
        """Remove the preview branch from the running pipeline (window closed / display error).
        The gate probe already drops every buffer, so no data is flowing into it."""
        self.ok = False
        for el in reversed(self.elements):
            try:
                el.set_state(Gst.State.NULL)
            except Exception:  # noqa: BLE001
                pass
        for el in self.elements:
            try:
                self.pipeline.remove(el)
            except Exception:  # noqa: BLE001
                pass
        return False

    def close(self):
        """Nothing to do: the branch lives inside the main pipeline and goes down with it."""
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

        # ---- profiling: CPU/GPU sampler starts NOW (before kernel build and model setup) ----
        self.pipeline_started = False
        self.frame_idx = 0
        self.up_bytes = 0                                  # host->GPU bytes per frame (model outputs)
        self.prof = Profiler(enabled=not a.no_profile)
        self.res = ResourceMonitor(a.gpu, self._phase, a.prof_period)
        self.res.start()

        # CUDA device + kernels (blur + post-processing) compiled now, so frame 1 is not slow
        self.dev = cp.cuda.Device(a.gpu)
        self.dev.use()
        mod = cp.RawModule(code=CUDA_SRC)
        self.dec = RetinaFaceDecoder(a.det_width, a.det_height, a.conf, a.nms, mod)
        self.dec.prof = self.prof
        self.plan = None

        # steady output timestamps: on by default for HLS / http(s) (bursty), off for RTSP and files,
        # unless the user gives --out-fps explicitly. --no-retime always turns it off.
        self.do_retime = (not a.no_retime) and (self.is_hls or a.out_fps > 0)
        self.nominal_fps = None          # frame rate announced by the stream (if any)
        self.retime_dur = None           # ns per output frame, decided at the first frame
        self.out_base = None
        self.out_n = 0
        self.retime_failed = False

        # live preview (optional): a DeepStream branch, rendered on the GPU
        self.preview = None
        self.fps_ts = deque()            # arrival times of the last ~2 s (for the FPS shown)
        self.cur_fps = 0.0
        self._pv_show = False            # set per frame: this frame may enter the preview branch
        self._batch_meta = None          # batch meta of the buffer being processed (for OSD metadata)
        self.pv_shown = 0                # frames sent to the preview while profiling was active
        self.t_pv = OrderedDict()        # preview branch timing (probe-to-probe)
        if a.preview:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                print("[note] --preview needs a desktop session (no DISPLAY found). "
                      "Preview is OFF; the pipeline runs normally.")
            else:
                self.preview = GpuPreview(a.preview_w, a.preview_h, a.preview_fps, not a.no_preview_boxes,
                                          a.preview_sink, a.gpu)

        self.k_h, self.k_v = mod.get_function("blur_h"), mod.get_function("blur_v")
        fr = cp.zeros(64 * 256, dtype=cp.uint8)
        tm = cp.zeros_like(fr)
        bx = cp.asarray(np.array([[4, 4, 40, 40, 5]], dtype=np.int32))
        for k in (self.k_h, self.k_v):
            k((3, 3, 1), (16, 16, 1), (fr, tm, bx, np.int32(256)))
        cp.cuda.Stream.null.synchronize()
        self.tmp = None
        self.dec.warm(a.pad, a.blur_frac)                 # compile + warm the GPU post-processing

        # JPEG saver (GPU nvJPEG when available, background writer thread)
        self.saver = None
        if a.sample_every > 0 or not a.no_save_detections:
            self.saver = JpegSaver(use_gpu=not a.no_gpu_jpeg)
            self.saver.prof = self.prof

        w = a.warmup
        self.st = {k: Stat(w) for k in ("detect", "post", "blur", "extras", "encode", "e2e", "preview")}
        self.st["preview"] = Stat(0)     # preview frames are rare (--preview-fps), so skip none of them
        self.t_mux, self.t_det_in, self.t_enc_in = OrderedDict(), OrderedDict(), OrderedDict()
        self.received = 0
        self.encoded = 0
        self.frame_idx = 0
        self.held = 0
        self.raw_counts = []
        self.n_saved_detections = 0
        self.frame_times = []
        self.pts = []
        self.last_rois = None            # (rois_g, m, maxw, maxh) of the last frame that had faces (GPU)
        self.miss = 0
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

    def _phase(self):
        """0 = init, 1 = pipeline start-up / warm-up frames, 2 = steady state (used by the sampler)."""
        if not getattr(self, "pipeline_started", False):
            return 0
        return 2 if getattr(self, "frame_idx", 0) >= self.a.warmup else 1

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

        head = [mux, q1, pgie, q2, conv1, caps1]
        tail = [q3, conv2, caps2, enc, parse, mux_out, sink]
        for chain in (head, tail):
            for left, right in zip(chain, chain[1:]):
                if not left.link(right):
                    sys.exit(f"Could not link {left.get_name()} -> {right.get_name()}")
        # caps1 -> q3 is the main path. With --preview a tee goes in between: main branch first, preview second.
        pv_ok = False
        if self.preview is not None:
            why = self.preview.preflight()
            if why:
                print(f"[note] --preview is OFF: {why}")
                self.preview = None
            else:
                pv_ok = self.preview.build(self.pipeline, caps1, q3, unified, a.width, a.height)
                if pv_ok:
                    setp(conv1, "output-buffers", 8)     # head-room: the preview branch holds a few frames
                else:
                    self.preview = None
        if not pv_ok and not caps1.link(q3):
            sys.exit("Could not link caps1 -> q3")

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
        if pv_ok:
            self.preview.gate_pad.add_probe(P, self.p_pv_gate, None)               # frame rate limiter
            self.preview.conv.get_static_pad("sink").add_probe(P, self.p_pv_in, None)
            self.preview.osd.get_static_pad("src").add_probe(P, self.p_pv_out, None)

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

    @timed("probe callback: p_mux_out")
    def p_mux_out(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            self.received += 1
            self._stamp(self.t_mux, buf)
        return Gst.PadProbeReturn.OK

    @timed("probe callback: p_det_in")
    def p_det_in(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            self._stamp(self.t_det_in, buf)
        return Gst.PadProbeReturn.OK

    @timed("probe callback: p_det_out")
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

    @timed("probe callback: p_retime")
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

    @timed("probe callback: p_enc_in")
    def p_enc_in(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            self._stamp(self.t_enc_in, buf)
        return Gst.PadProbeReturn.OK

    @timed("probe callback: p_enc_out")
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

    @timed("probe callback: p_pv_gate")
    def p_pv_gate(self, pad, info, _):
        """On the preview branch: only frames that got OSD metadata (at --preview-fps) may pass."""
        if self._pv_show and self.preview is not None and self.preview.ok:
            return Gst.PadProbeReturn.OK
        return Gst.PadProbeReturn.DROP

    @timed("probe callback: p_pv_in")
    def p_pv_in(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            self._stamp(self.t_pv, buf)
        return Gst.PadProbeReturn.OK

    @timed("probe callback: p_pv_out")
    def p_pv_out(self, pad, info, _):
        buf = info.get_buffer()
        if buf is not None:
            dt = self._lap(self.t_pv, buf)
            if dt is not None:
                self.st["preview"].add(dt)
        return Gst.PadProbeReturn.OK

    def _drop_preview_branch(self):
        """Main-loop callback: remove the preview branch after a window close / display error."""
        if self.preview is not None:
            self.preview.remove_branch()
            try:
                if self.pipeline_started and not self.stopping:
                    self.pipeline.set_state(Gst.State.PLAYING)      # make sure the main path keeps running
            except Exception:  # noqa: BLE001
                pass
        return False

    # -------------------------------------------------------------- main probe
    def p_main(self, pad, info, _):
        self._pv_show = False
        if self.failed:
            return Gst.PadProbeReturn.OK
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        try:
            bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            self._batch_meta = bm
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
        """Raw model outputs as seen by nvinfer: list of (name, dims, raw_host_view, is_half).
        No conversion here: fp16 -> fp32 now happens on the GPU."""
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
                    raw = np.ctypeslib.as_array(ptr, shape=(n,))
                    outs.append((layer.layerName, dims, raw, half))
                return outs
            try:
                node = node.next
            except StopIteration:
                break
        return None

    @staticmethod
    def _host_f32(raw, half):
        """First frame only (start-up check of the output layers)."""
        return raw.view(np.float16).astype(np.float32) if half else raw

    @staticmethod
    def _to_gpu_f32(raw, half):
        """Per frame: raw host view -> GPU, fp16 -> fp32 conversion done on the GPU."""
        g = cp.asarray(raw)
        return g.view(cp.float16).astype(cp.float32) if half else g

    @staticmethod
    def _gpu_resize(img, sc):
        """Bilinear downscale on the GPU (only used for the optional side-by-side samples)."""
        try:
            from cupyx.scipy import ndimage as cndi
            out = cndi.zoom(img.astype(cp.float32), (sc, sc, 1), order=1)
            return cp.ascontiguousarray(cp.clip(out, 0, 255).astype(cp.uint8))
        except Exception:  # noqa: BLE001
            return img

    def _blur(self, frame, rois_g, m, maxw, maxh, H):
        pitch = int(frame.strides[0])
        need = H * pitch
        if self.tmp is None or self.tmp.size < need:
            self.tmp = cp.empty(need, dtype=cp.uint8)
        grid = ((maxw + 15) // 16, (maxh + 15) // 16, int(m))
        args = (frame, self.tmp, rois_g, np.int32(pitch))
        with self.prof.gpu("blur_h kernel (horizontal pass)", "CUDA kernel", "App._blur"):
            self.k_h(grid, (16, 16, 1), args)
        with self.prof.gpu("blur_v kernel (vertical pass)", "CUDA kernel", "App._blur"):
            self.k_v(grid, (16, 16, 1), args)

    def _process(self, buf, fm):
        """Wrapper around the per-frame work: switches the profiler on after the warm-up frames
        and measures the whole probe call (wall + CPU) so the glue code can be derived."""
        p = self.prof
        if p.enabled and not p.active and self.frame_idx >= self.a.warmup:
            p.active = True
        if not p.active:
            return self._process_impl(buf, fm)
        p.frames += 1
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            return self._process_impl(buf, fm)
        finally:
            p.add(INCLUSIVE, "-", "inclusive", "App._process",
                  wall=time.perf_counter() - t0, cpu=time.thread_time() - c0, in_proc=False)
            p.flush_gpu()

    def _process_impl(self, buf, fm):
        a = self.a
        prof = self.prof
        self.dev.use()                                        # GStreamer threads start on device 0
        idx = self.frame_idx
        t0 = time.perf_counter()

        with prof.cpu("read tensor meta + wrap GPU frame", "python / ctypes", "App._tensors, App._gpu_frame"):
            frame = self._gpu_frame(buf, fm.batch_id)         # (H, W, 4) RGBA on the GPU
            H, W = int(frame.shape[0]), int(frame.shape[1])
            outs = self._tensors(fm)
        if outs is None:
            raise RuntimeError("No tensor output meta on the frame. nvinfer needs "
                               "network-type=100 and output-tensor-meta=1 (set by this script).")
        if self.plan is None:
            host_outs = [(n, d, self._host_f32(raw, half)) for n, d, raw, half in outs]
            print("\nEngine outputs as seen by nvinfer:")
            for n, d, arr in host_outs:
                print(f"    {n}: dims={d} elements={arr.size}")
            self.plan = self.dec.build_plan(host_outs)
            self.frame_size = (W, H)
            if self.preview is not None and self.preview.built and (W, H) != (self.preview.W0, self.preview.H0):
                try:
                    self.preview.set_frame_size(W, H)      # before the first buffer reaches the preview branch
                except Exception:  # noqa: BLE001
                    print("[warn] could not adjust the preview size:\n" + traceback.format_exc())
            print(f"Frame size at nvinfer: {W}x{H} | RetinaFace plan OK | GPU post-processing + "
                  f"blur kernels ready\n")
            if a.max_seconds > 0:
                GLib.timeout_add_seconds(int(a.max_seconds), self._timer_stop)

        # ---- GPU: decode + top-K + NMS + ROIs (one 28-byte readback at the end) ----
        lname, ldims, lraw, lhalf = outs[self.plan["loc"]]
        cname, cdims, craw, chalf = outs[self.plan["conf"]]
        if not self.up_bytes:
            self.up_bytes = int(lraw.nbytes + craw.nbytes)
        with prof.gpu("upload loc+conf (H2D copy + fp16->fp32)", "H2D copy + CuPy cast", "App._to_gpu_f32"):
            loc_g = self._to_gpu_f32(lraw, lhalf)
            cf_g = self._to_gpu_f32(craw, chalf)
        det_g, rois_g, raw, m, maxw, maxh, conf_stats = self.dec.process_gpu(
            loc_g, cf_g, W, H, a.pad, a.blur_frac)
        self.raw_counts.append(raw)

        held_used = False
        if raw:
            self.last_rois, self.miss = (rois_g, m, maxw, maxh), 0
        else:
            self.miss += 1
            if self.miss <= a.hold and self.last_rois is not None and self.last_rois[1] > 0:
                rois_g, m, maxw, maxh = self.last_rois        # blur regions of the last frame with faces
                self.held += 1
                held_used = True
        t1 = time.perf_counter()

        # Periodic samples AND/OR every frame where a face was detected and blurred
        sample = a.sample_every > 0 and idx % a.sample_every == 0
        save_det = (self.saver is not None) and (not a.no_save_detections) and raw > 0 and m > 0
        orig = None
        if sample and self.saver is not None:
            with prof.gpu("copy original frame (samples)", "D2D copy", "App._process_impl"):
                orig = cp.ascontiguousarray(frame[:, :, :3])  # GPU copy

        t2 = time.perf_counter()
        blur_err = None
        try:
            if m:
                self._blur(frame, rois_g, m, maxw, maxh, H)
            with prof.wait("wait: stream sync after blur", "cudaStreamSynchronize", "App._process_impl"):
                cp.cuda.Stream.null.synchronize()              # blurred pixels are final from here
        except Exception as e:  # noqa: BLE001
            blur_err = e
        t3 = time.perf_counter()

        with prof.cpu("write detection + audit logs", "python / json / file", "App._log_frame"):
            self._log_frame(buf, idx, raw, conf_stats, held_used, m, t1 - t0, t3 - t2, blur_err)
        if blur_err is not None:
            raise blur_err

        if orig is not None or save_det:
            with prof.gpu("copy blurred frame RGB for JPEG", "D2D copy", "App._process_impl"):
                after = cp.ascontiguousarray(frame[:, :, :3])  # blurred RGB, GPU -> GPU copy
            if orig is not None:
                with prof.gpu("concat side-by-side (samples)", "CuPy ops", "App._process_impl"):
                    side = cp.concatenate([orig, after], axis=1)
                sc = 1800.0 / side.shape[1]
                if sc < 1.0:
                    with prof.gpu("resize samples (cupyx zoom)", "CuPy ops", "App._gpu_resize"):
                        side = self._gpu_resize(side, sc)
                self.saver.save(os.path.join(a.sample_dir, f"frame_{idx:06d}.jpg"), side)
            if save_det:
                name = f"det_{idx:06d}_faces{raw}.jpg"          # only the blurred frame
                self.saver.save(os.path.join(a.detected_dir, name), after)
                self.n_saved_detections += 1

        # --- current FPS (average over the last 2 seconds) + live preview window ----------
        tn = time.perf_counter()
        self.fps_ts.append(tn)
        while tn - self.fps_ts[0] > 2.0:
            self.fps_ts.popleft()
        if len(self.fps_ts) > 1 and tn > self.fps_ts[0]:
            self.cur_fps = (len(self.fps_ts) - 1) / (tn - self.fps_ts[0])
        pv = self.preview
        if pv is not None and pv.built and pv.ok and pv.due(tn):
            try:
                with prof.cpu("preview: read boxes + build OSD metadata", "python / pyds metadata",
                              "GpuPreview.attach_osd"):
                    # only the boxes (N x 5 floats) leave the GPU; the frame itself stays there
                    det_host = det_g[:raw].get() if (raw and pv.draw_boxes) else np.zeros((0, 5), np.float32)
                    pv.attach_osd(self._batch_meta, fm, det_host, self.cur_fps)
                self._pv_show = True                        # the gate probe lets this frame into the preview branch
                if prof.active:
                    self.pv_shown += 1
            except Exception:  # noqa: BLE001  (the preview must never stop the real work)
                print("\n[warn] live preview stopped because of an error (the run continues):\n"
                      + traceback.format_exc())
                pv.ok = False
                GLib.idle_add(self._drop_preview_branch)

        t4 = time.perf_counter()
        self.st["post"].add(t1 - t0)
        self.st["blur"].add(t3 - t2)
        self.st["extras"].add(t4 - t3)                         # logging + image saving + preview
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

    def _log_frame(self, buf, idx, raw, conf, held_used, n_blur, post_s, blur_s, err):
        """One line in detection_logs.jsonl and one in privacy_audit.jsonl for this frame.
        `conf` = {"min","mean","max"} computed on the GPU (or None when there are no detections)."""
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
            if self.preview is not None and self.preview.built and self.preview.owns(msg.src):
                # window closed / display problem: drop the preview, keep processing and saving
                print(f"[preview] window closed or display error ({err.message}); "
                      f"the pipeline keeps running without the preview.")
                self.preview.ok = False
                GLib.idle_add(self._drop_preview_branch)
                return True
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
        self.pipeline_started = True
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            if self.preview is not None:
                self.preview.close()
            if self.saver is not None:
                self.saver.close()                              # wait until every JPEG is on disk
            self.mon.stop()
            self.res.stop()
            self.res.join(timeout=3)                            # last sample = "at finish"
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
        probe_mean = s["post"]["mean"] + s["blur"]["mean"] + s["extras"]["mean"]
        stages = {"detect (nvinfer)": s["detect"]["mean"], "probe (post+blur+log/save)": probe_mean,
                  "encode (NVENC)": s["encode"]["mean"]}
        slow = max(stages, key=stages.get)
        est_max = 1000.0 / stages[slow] if stages[slow] > 0 else 0.0
        raw = np.array(self.raw_counts)
        cpu_cores = 0.0
        cpu_window_s = None
        if self.warm_wall is not None and self.last_frame_wall > self.warm_wall:
            cpu_window_s = self.last_cpu - self.warm_cpu
            cpu_cores = cpu_window_s / (self.last_frame_wall - self.warm_wall)
        m = self.mon
        avg = lambda v: float(np.mean(v)) if v else 0.0  # noqa: E731

        line = "=" * 64
        print("\n" + line)
        print(" PERFORMANCE REPORT")
        print(line)
        print(f" GPU                        : {m.name} ({m.total_mb} MB)")
        print(f" Pipeline                   : DeepStream + TensorRT engine, GPU post-processing "
              f"(decode+NMS+ROI) + GPU blur (CUDA)")
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
        rows = [("Detect (nvinfer)", s["detect"]), ("Post (GPU): decode+NMS", s["post"]),
                ("Blur (GPU kernels)", s["blur"]), ("Log+save+preview meta", s["extras"]),
                ("Encode (NVENC)", s["encode"]),
                ("End-to-end mux->file", s["e2e"])]
        if self.preview is not None and self.preview.built and s["preview"]["n"]:
            rows.insert(4, ("Preview branch (GPU)", s["preview"]))
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
                  f"({self.n_saved_detections} images, blurred frame only)")
        print(f" Metrics JSON   : {os.path.abspath(a.metrics_json)}")
        if self.det_log_path:
            print(f" Detection log  : {os.path.abspath(self.det_log_path)}")
            print(f" Privacy audit  : {os.path.abspath(self.aud_log_path)}")
        asum = self.audit_summary()
        print(f" Privacy audit result: {asum['overall']}  {asum['by_status']}")

        # ---- optimization report: CPU/GPU usage, function map, workload + code split ----
        prof_text = prof_json = None
        rep_txt = a.profile_report
        rep_base = os.path.splitext(rep_txt)[0]
        try:
            prof_text, prof_json = build_optimization_report(self, dict(
                n=n, steady_fps=steady_fps, whole_fps=whole_fps, est_max=est_max, slow=slow,
                headroom=headroom, dropped=dropped, cpu_cores=cpu_cores, cpu_window_s=cpu_window_s,
                s=s, iv=iv, raw=raw, asum=asum))
            with open(rep_txt, "w", encoding="utf-8") as fh:
                fh.write(prof_text + "\n")
            with open(rep_base + ".json", "w", encoding="utf-8") as fh:
                json.dump(prof_json, fh, indent=2, default=float)
            self.res.write_csv(rep_base + "_timeline.csv")
            print("\n" + prof_text)
            print(f"\n Optimization report saved : {os.path.abspath(rep_txt)}")
            print(f"                             {os.path.abspath(rep_base + '.json')}")
            print(f"                             {os.path.abspath(rep_base + '_timeline.csv')}")
        except Exception:  # noqa: BLE001  (the report must never break the run)
            print("\n[warn] could not build the optimization report:\n" + traceback.format_exc())

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
            jpeg_saver=(self.saver.mode if self.saver is not None else None),
            privacy_audit_summary=asum,
            detection_log=os.path.abspath(self.det_log_path) if self.det_log_path else None,
            privacy_audit_log=os.path.abspath(self.aud_log_path) if self.aud_log_path else None,
            sample_dir=os.path.abspath(a.sample_dir) if a.sample_every > 0 else None,
            detected_dir=os.path.abspath(a.detected_dir) if not a.no_save_detections else None,
            n_saved_detections=self.n_saved_detections,
            optimization_report_file=os.path.abspath(rep_txt) if prof_text else None,
            optimization_report=prof_json)
        with open(a.metrics_json, "w") as f:
            json.dump(report, f, indent=2, default=float)
        return 0


# ----------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DeepStream + TensorRT RetinaFace-R50 face blur "
                                            "(GPU post-processing + CPU/GPU optimization report)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", help="file path, rtsp://..., or http(s)://...m3u8")
    p.add_argument("--rutube-page", help="Rutube page link; the HLS link is found automatically")
    p.add_argument("--onnx", default="retinaface_r50_640x640.onnx",
                   help="RetinaFace ONNX; nvinfer builds the TensorRT engine from it on first run")
    p.add_argument("--engine", default="",
                   help="TensorRT engine path (default: <onnx>_b1_gpu0_fp16.engine next to the ONNX). "
                        "Built automatically if missing, reused if present.")
    p.add_argument("--model-name", default="", help="name shown in the report (default: guessed from the file name)")
    p.add_argument("--det-width", type=int, default=0,
                   help="ONNX input width (0 = read it from the file name, e.g. ..._1280x736.onnx, else 640)")
    p.add_argument("--det-height", type=int, default=0, help="ONNX input height (0 = same auto rule)")
    p.add_argument("--conf", type=float, default=0.5, help="face score threshold (lower = fewer misses)")
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--pad", type=float, default=0.20, help="enlarge each face box by this fraction")
    p.add_argument("--blur-frac", type=float, default=0.25,
                   help="blur radius as a fraction of the face size (bigger = stronger)")
    p.add_argument("--hold", type=int, default=5, help="keep last boxes this many frames if a face is missed")
    p.add_argument("--output", default="output_blurred.mkv")
    p.add_argument("--out-dir", default="logs",
                   help="folder where the video, logs, metrics and sample frames are saved")
    p.add_argument("--codec", choices=["h264", "h265"], default="h264")
    p.add_argument("--bitrate", type=int, default=8_000_000)
    p.add_argument("--preview", action="store_true",
                   help="open a live window with the blurred video, green face boxes and an FPS overlay. "
                        "Rendered on the GPU by a DeepStream branch (nvvideoconvert -> nvdsosd -> nveglglessink); "
                        "needs a desktop session with EGL.")
    p.add_argument("--preview-size", default="1000x1000",
                   help="maximum preview window size WIDTHxHEIGHT; the video keeps its shape and fits inside (scaled on the GPU)")
    p.add_argument("--preview-fps", type=float, default=15,
                   help="how many frames per second the preview window shows (does not affect the saved video)")
    p.add_argument("--no-preview-boxes", action="store_true",
                   help="do not draw the green detection boxes in the preview window")
    p.add_argument("--preview-sink", default="nveglglessink",
                   help="DeepStream display sink for the GPU preview (nveglglessink on x86; nv3dsink where available)")
    p.add_argument("--out-fps", type=float, default=0,
                   help="fixed frame rate of the saved video (steady timestamps). 0 = automatic: used for "
                        "HLS / http(s) sources only, taken from the stream (fallback 25). Give a number to "
                        "force it for any source type.")
    p.add_argument("--no-retime", action="store_true",
                   help="keep the original (arrival) timestamps in the saved video, even for HLS")
    p.add_argument("--rtsp-protocol", choices=["tcp", "udp"], default="tcp",
                   help="RTSP transport. TCP does not lose packets (recommended).")
    p.add_argument("--latency", type=int, default=500,
                   help="RTSP jitter buffer in ms (bigger = safer on a bad network, more delay)")
    p.add_argument("--sample-dir", default="sample_frames")
    p.add_argument("--save-samples", action="store_true",
                   help="save before/after sample frames (every 100 frames unless --sample-every is given). "
                        "Without --save-samples / --sample-every nothing is stored.")
    p.add_argument("--sample-every", type=int, default=0,
                   help="save a before/after image every N frames (0 = off)")
    p.add_argument("--detected-dir", default="detected_instances",
                   help="folder (under --out-dir) for every frame where a face was detected and blurred")
    p.add_argument("--no-save-detections", action="store_true",
                   help="do not save per-detection images into detected_instances/")
    p.add_argument("--no-gpu-jpeg", action="store_true",
                   help="do not use the GPU JPEG encoder (nvImageCodec); encode with OpenCV in a background thread")
    p.add_argument("--log-dir", default="",
                   help="folder for detection_logs.jsonl and privacy_audit.jsonl (default: next to --metrics-json)")
    p.add_argument("--no-logs", action="store_true", help="do not write the two .jsonl log files")
    p.add_argument("--metrics-json", default="metrics.json")
    p.add_argument("--profile-report", default="optimization_report.txt",
                   help="file name of the saved optimization report (.json and _timeline.csv are written next to it)")
    p.add_argument("--no-profile", action="store_true",
                   help="switch off the per-stage CPU/GPU timing (use for a pure speed benchmark); "
                        "the CPU/GPU usage sampler stays on")
    p.add_argument("--prof-period", type=float, default=0.5,
                   help="seconds between CPU/GPU usage samples")
    p.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds (0 = until the end)")
    p.add_argument("--warmup", type=int, default=30, help="frames excluded from the timings")
    p.add_argument("--width", type=int, default=1920, help="nvstreammux width (match your source)")
    p.add_argument("--height", type=int, default=1080, help="nvstreammux height (match your source)")
    p.add_argument("--max-height", type=int, default=720, help="stream quality cap for --rutube-page")
    p.add_argument("--leaky", action="store_true",
                   help="live: drop old frames instead of queueing when slow "
                        "(NOT recommended for HLS: it drops frames during bursts)")
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    if a.rutube_page:
        a.source = resolve_rutube(a.rutube_page, a.max_height)
        print("Rutube HLS link:", a.source[:100], "...")
        with open("rutube_link.txt", "w") as _f:
            _f.write(a.source)
        print("(full link saved in rutube_link.txt)")
    if not a.source:
        p.error("give --source or --rutube-page")
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
        # put everything in one folder (default: logs/)
    os.makedirs(a.out_dir, exist_ok=True)

    def _in_dir(path):
        return path if os.path.isabs(path) else os.path.join(a.out_dir, os.path.basename(path))

    a.output = _in_dir(a.output)
    a.metrics_json = _in_dir(a.metrics_json)
    a.profile_report = _in_dir(a.profile_report)
    a.sample_dir = _in_dir(a.sample_dir)
    a.detected_dir = _in_dir(a.detected_dir)
    a.log_dir = a.log_dir or a.out_dir
    if a.save_samples and a.sample_every <= 0:
        a.sample_every = 100
    if a.sample_every > 0:
        os.makedirs(a.sample_dir, exist_ok=True)
    if not a.no_save_detections:
        os.makedirs(a.detected_dir, exist_ok=True)
    return a


if __name__ == "__main__":
    sys.exit(App(parse_args()).run())