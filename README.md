# Face Detection + GPU Blur (DeepStream + TensorRT)

Real-time face detection and blurring pipeline using **NVIDIA DeepStream**, **TensorRT**, and custom **CUDA** blur kernels.

**Goal:** privacy-preserving video processing — detect faces and blur them on the GPU with low latency and high throughput.

| Item | Value |
|------|-------|
| Device | NVIDIA GeForce RTX 5080 |
| Runtime | DeepStream + TensorRT (FP16) |
| Best speed (SCRFD-2.5G) | **398.5 FPS** (video file) / **408.5 FPS** (high-rate RTSP) |
| Pipeline | Almost fully on GPU |

---

## Contents

1. [Supported models](#1-supported-models)
2. [Project outcome](#2-project-outcome)
3. [Pipeline overview](#3-pipeline-overview)
4. [Module execution placement](#4-module-execution-placement)
5. [Performance results](#5-performance-results)
6. [How to run](#6-how-to-run)
7. [High-FPS RTSP testing](#7-high-fps-rtsp-testing)
8. [Documentation](#8-documentation)
9. [Folder guide](#9-folder-guide)
10. [Requirements](#10-requirements)
11. [Known limitations](#11-known-limitations)

---

## 1. Supported models

| Model | Input Size | Avg FPS | Total Detections | ONNX |
|-------|------------|---------|------------------|------|
| **SCRFD-2.5G** | 640×640 | **398.5** | 42,802 | [Download](https://media.githubusercontent.com/media/cysin/scrfd_onnx/main/scrfd_2.5g_bnkps_shape640x640.onnx) |
| **SCRFD-10G** | 640×640 | 381.4 | 45,243 | [Download](https://media.githubusercontent.com/media/cysin/scrfd_onnx/main/scrfd_10g_bnkps.onnx) |
| **RetinaFace-R50** | 1280×736 | 346.1 | **77,585** | [Download](https://drive.google.com/uc?export=download&id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW) |

- **SCRFD-2.5G** — fastest, best for real-time
- **SCRFD-10G** — slightly more detections, still very fast
- **RetinaFace-R50** — most faces detected, higher GPU use

Full model details → [docs/MODEL.md](docs/MODEL.md)

---

## 2. Project outcome

1. SCRFD and RetinaFace TensorRT engines prepared for face detection.
2. DeepStream pipeline built: decode → detect → GPU blur → encode.
3. Tested on video files (hundreds of FPS).
4. Tested on real live streams (Rutube / RTSP) — limited to ~25 FPS by the source.
5. Local high-rate RTSP server used to measure true pipeline speed → **~408.5 FPS** (SCRFD-2.5G).

### Final decision

| Decision area | Selected result | Reason |
|---------------|-----------------|--------|
| Best real-time detector | **SCRFD-2.5G** | Fastest (398.5 FPS), lightest GPU use |
| Most detections | **RetinaFace-R50** | 77,585 faces vs ~43k for SCRFD |
| Blur method | Custom CUDA kernels (GPU) | Fast elliptical blur, all faces in two launches |
| Runtime | DeepStream + TensorRT | Full GPU pipeline, zero-copy buffers |

---

## 3. Pipeline overview

```
nvurisrcbin (NVDEC decode)
    → nvstreammux
    → nvinfer (TensorRT face detector)
    → nvvideoconvert (to RGBA)
    → Python probe (CUDA blur kernels)
    → nvvideoconvert (to NV12)
    → nvv4l2h264enc / nvv4l2h265enc (NVENC)
    → filesink
```

Frames stay in GPU memory for almost the entire path.  
Only small detector output tensors are copied to CPU for box decoding and NMS.

---

## 4. Module execution placement

| Module | Device | Notes |
|--------|--------|-------|
| Stream Decode (NVDEC) | **GPU** | Hardware video decode |
| Face Detection (TensorRT) | **GPU** | SCRFD or RetinaFace engine |
| Post-processing (decode + NMS) | CPU (lightweight) | Tiny tensors only |
| Face Blur (Custom CUDA kernels) | **GPU** | Separable box blur inside face ellipse |
| Video Encode (NVENC) | **GPU** | Hardware H.264 / H.265 encode |
| Stream Mux / Queue (DeepStream) | **GPU** | Zero-copy GPU buffers |

Almost the entire pipeline runs on GPU.  
Details → [docs/MODULES_GPU.md](docs/MODULES_GPU.md)

---

## 5. Performance results

**Device:** NVIDIA GeForce RTX 5080  
**Input:** 1920×1080 video file | 1,800 frames | 0 dropped

| Metric | SCRFD 2.5G | SCRFD 10G | RetinaFace-R50 |
|--------|------------|-----------|----------------|
| **Avg Pipeline FPS** | **398.5** | **381.4** | **346.1** |
| Total Detections | 42,802 | 45,243 | **77,585** |
| Avg Faces / Frame | 23.78 | 25.14 | 43.10 |
| GPU util avg / peak | 18.3% / 23% | 30.0% / 37% | 74.6% / 77% |
| GPU memory peak | 4158 MB | 4216 MB | 4398 MB |

### Latency per frame (ms) — mean / max

| Stage | SCRFD 2.5G | SCRFD 10G | RetinaFace-R50 |
|-------|------------|-----------|----------------|
| Detect | 0.42 / 3.09 | 0.68 / 1.76 | 1.82 / 6.14 |
| Post (decode + NMS) | 0.62 / 2.19 | 0.61 / 1.97 | 0.80 / 2.42 |
| Blur (CUDA) | 0.15 / 0.60 | 0.37 / 0.84 | 1.36 / 5.81 |
| Encode (NVENC) | 1.02 / 2.43 | 1.15 / 2.04 | 2.21 / 4.68 |

Full results → [docs/RESULTS.md](docs/RESULTS.md)

---

## 6. How to run

### Install first

See → [docs/INSTALL_DEEPSTREAM.md](docs/INSTALL_DEEPSTREAM.md)

### SCRFD (2.5G or 10G)

```bash
# Build engine once (from ONNX)
python3 scrfd_face_detection.py \
  --source video.mp4 \
  --engine scrfd_2.5g.engine \
  --onnx scrfd_2.5g_bnkps_shape640x640.onnx

# Run
python3 scrfd_face_detection.py \
  --source video.mp4 \
  --engine scrfd_2.5g.engine \
  --output output_blurred.mp4
```

### RetinaFace-R50

```bash
# 1. Convert .pth → ONNX
python3 pytorch_to_onnx_retinaface.py

# 2. Build engine once
python3 retinaface_face_detection.py \
  --source video.mp4 \
  --engine retinaface_r50.engine \
  --onnx retinaface_r50.onnx

# 3. Run
python3 retinaface_face_detection.py \
  --source video.mp4 \
  --engine retinaface_r50.engine \
  --output output_blurred.mp4
```

### Live / RTSP

```bash
python3 scrfd_face_detection.py \
  --source rtsp://127.0.0.1:8554/stream \
  --engine scrfd_2.5g.engine \
  --output live_blurred.mp4 \
  --max-seconds 60 \
  --width 1280 \
  --height 720 \
  --leaky
```

Full run guide → [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)

---

## 7. High-FPS RTSP testing

Real live streams only send ~25–30 FPS. To measure true pipeline speed:

**Terminal 1 – RTSP server (no `-re`)**

```bash
ffmpeg -stream_loop -1 -i video.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/stream
```

**Terminal 2 – pipeline**

```bash
python3 scrfd_face_detection.py \
  --source rtsp://127.0.0.1:8554/stream \
  --engine scrfd_2.5g.engine \
  --output live_blurred.mp4 \
  --max-seconds 60 \
  --leaky
```

Details → [docs/HIGH_FPS_RTSP.md](docs/HIGH_FPS_RTSP.md)

---

## 8. Documentation

| Document | Description |
|----------|-------------|
| [docs/INSTALL_DEEPSTREAM.md](docs/INSTALL_DEEPSTREAM.md) | Install DeepStream + Python packages |
| [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | How to run SCRFD and RetinaFace |
| [docs/MODEL.md](docs/MODEL.md) | Model details + ONNX download links |
| [docs/RESULTS.md](docs/RESULTS.md) | Full performance comparison |
| [docs/MODULES_GPU.md](docs/MODULES_GPU.md) | Which modules run on GPU vs CPU |
| [docs/HIGH_FPS_RTSP.md](docs/HIGH_FPS_RTSP.md) | High-rate RTSP testing method |

---

## 9. Folder guide

| Path | Purpose |
|------|---------|
| `scrfd_face_detection.py` | SCRFD pipeline (2.5G / 10G) |
| `retinaface_face_detection.py` | RetinaFace-R50 pipeline |
| `pytorch_to_onnx_retinaface.py` | Convert RetinaFace `.pth` → ONNX |
| `README.md` | This document |
| `requirements.txt` | Python dependencies |
| `docs/` | All documentation |

---

## 10. Requirements

- Ubuntu 22.04 (or similar)
- NVIDIA GPU (**x86 only** — Jetson not supported)
- DeepStream 7.x or 8.x with Python bindings (`pyds`)
- CUDA 12.x (or 11.x)
- Python 3.8+

```bash
pip install -r requirements.txt
```

```
numpy
cupy-cuda12x          # or cupy-cuda11x for CUDA 11
opencv-python-headless
nvidia-ml-py
requests
```

---

## 11. Known limitations

- Jetson is not supported (x86-only GPU surface API).
- Real live streams show ~25–30 FPS — limited by the source, not the pipeline.
- Engine files are TensorRT-version specific; pass `--onnx` to rebuild if needed.

---

## License

Internal use – AI Research / Real-time Video Privacy
