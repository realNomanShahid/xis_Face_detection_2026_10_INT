# Face Detection + GPU Blur (DeepStream + TensorRT)

Real-time face detection and blurring pipeline using **NVIDIA DeepStream**, **TensorRT**, and custom **CUDA** blur kernels.

**Goal:** privacy-preserving video processing — detect faces and blur them on the GPU with low latency.

**Supported detector:** **RetinaFace ResNet-50** only.

SCRFD-2.5G and SCRFD-10G were evaluated during research (strong speed) and **discarded** because accuracy was not accepted versus RetinaFace. They are not part of the supported run path.

| Item | Value |
|------|--------|
| Device (evidence) | NVIDIA GeForce RTX 5080 |
| Runtime | DeepStream + TensorRT (FP16) |
| Detector | **RetinaFace-R50** (1280×736) |
| Pipeline | Almost fully on GPU |
| Typical live source | ~30 FPS (camera / stream limited) |
| File / high-rate RTSP | Hundreds of FPS possible (pipeline headroom) |

---

## Contents

1. [Supported model](#1-supported-model)
2. [Project outcome](#2-project-outcome)
3. [Pipeline overview](#3-pipeline-overview)
4. [Module execution placement](#4-module-execution-placement)
5. [Performance (RetinaFace)](#5-performance-retinaface)
6. [How to run](#6-how-to-run)
7. [Configuration](#7-configuration)
8. [High-FPS RTSP testing](#8-high-fps-rtsp-testing)
9. [Documentation](#9-documentation)
10. [Repository structure](#10-repository-structure)
11. [Requirements](#11-requirements)
12. [Known limitations](#12-known-limitations)
13. [Downlaod Model](#13-retinaface-resnet-50)

---

## 1. Supported model

| Model | Input size | Role |
|-------|------------|------|
| **RetinaFace-R50** | 1280×736 | **Production path** — accuracy priority, then speed |

| Discarded (not supported) | Reason |
|---------------------------|--------|
| SCRFD-2.5G | Speed OK; accuracy not accepted vs RetinaFace |
| SCRFD-10G | Speed OK; accuracy not accepted vs RetinaFace |

Weights: pretrained official RetinaFace-R50 → ONNX → TensorRT engine (no company fine-tune in this phase).

Model notes → [docs/md_files/MODEL.md](docs/md_files/MODEL.md)

---

## 2. Project outcome

1. RetinaFace-R50 ONNX and TensorRT FP16 engine prepared for DeepStream `nvinfer`.
2. Pipeline: decode → detect → GPU ellipse blur → encode (MKV/MP4).
3. Tested on **video files**, **office RTSP**, **Rutube HLS**, and **local high-rate RTSP**.
4. Live camera soak documented (see handbook artifacts / soak notes).
5. Defaults externalized under `configs/*.yaml`.

### Final decision

| Decision area | Selected result | Reason |
|---------------|-----------------|--------|
| Detector | **RetinaFace-R50** | Higher detection accuracy on observed streams |
| SCRFD family | **Discarded** | Not accepted for accuracy |
| Blur | Custom CUDA kernels (GPU) | Elliptical blur; multi-face in two launches |
| Runtime | DeepStream + TensorRT | Zero-copy GPU buffers, NVDEC / NVENC |
| Shareable output | Prefer **`.mkv`** | Better player compatibility than fragmented live MP4 |

---

## 3. Pipeline overview

```text
nvurisrcbin (NVDEC decode, RTSP / HLS / file)
  → nvstreammux
  → nvinfer (TensorRT RetinaFace-R50)
  → nvvideoconvert (RGBA)
  → Python probe (decode boxes + CUDA blur [+ optional preview])
  → nvvideoconvert (NV12)
  → optional steady timestamps
  → nvv4l2h264enc / h265 (NVENC)
  → matroskamux or qtmux
  → filesink
```

Frames stay in GPU memory for almost the entire path.  
Only small detector output tensors are copied to CPU for box decoding and NMS.

---

## 4. Module execution placement

| Module | Device | Notes |
|--------|--------|-------|
| Stream decode (NVDEC) | **GPU** | Hardware decode |
| Face detection (TensorRT) | **GPU** | RetinaFace-R50 engine |
| Post-process (decode + NMS) | **GPU** | tensors|
| Face blur (CUDA kernels) | **GPU** | Separable blur inside face ellipse |
| Video encode (NVENC) | **GPU** | H.264 / H.265 |
| Mux / queues (DeepStream) | **GPU** | Zero-copy buffers |

Details → [docs/md_files/MODULES_GPU.md](docs/md_files/MODULES_GPU.md)

---

## 5. Performance (RetinaFace)

### File-mode example (high throughput)

**Device:** NVIDIA GeForce RTX 5080  
**Example archived run:** `results/RetinaFace_resnet50/` (see `metrics.json` there for exact numbers)

| Metric | Typical order (file, GPU not source-limited) |
|--------|-----------------------------------------------|
| Pipeline FPS | Hundreds of FPS on file / high-rate RTSP |
| Detect latency | ~1–2 ms mean class on this GPU |
| Blur | Sub-ms to low-ms depending on face count |
| GPU util | Higher than lighter detectors (expected) |

### Live office camera (source-limited)

Live RTSP is limited by what the camera delivers (~30 FPS), not by the pipeline when headroom is large.

Example soak-class behaviour (see project soak / A11 notes):

| Metric | Example |
|--------|---------|
| Delivered FPS | ~30 |
| Keeping up with source | Yes, 0 frames lost |
| Privacy audit | PASS when blur ops succeed on detections |

Real live streams will **not** show file-mode hundreds of FPS. That is expected.

Full write-up → [docs/md_files/RESULTS.md](docs/md_files/RESULTS.md)

---

## 6. How to run

### Install first

→ [docs/md_files/INSTALL_DEEPSTREAM.md](docs/md_files/INSTALL_DEEPSTREAM.md)

### RetinaFace-R50

```bash
# 1. Convert .pth → ONNX (once)
python3 src/pytorch_to_onnx_retinaface.py

# 2. Build engine once (or let nvinfer build on first run)
python3 src/retinaface_face_detection.py \
  --source video.mp4 \
  --onnx retinaface_r50_1280x736.onnx \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine

# 3. Run
python3 src/retinaface_face_detection.py \
  --source video.mp4 \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --output output_blurred.mkv \
  --conf 0.15
```

### Live / RTSP

```bash
python3 src/retinaface_face_detection.py \
  --source "rtsp://user:ENCODED_PASS@host:554/path" \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --width 1536 --height 864 \
  --rtsp-protocol tcp --latency 500 \
  --preview \
  --output camera_blurred.mkv
```

Defaults can come from `configs/*.yaml` (CLI overrides YAML).

Full guide → [docs/md_files/HOW_TO_RUN.md](docs/md_files/HOW_TO_RUN.md)

---

## 7. Configuration

| Path | Role |
|------|------|
| `configs/pipeline.yaml` | Resolution, output, GPU, retime |
| `configs/detection.yaml` | ONNX / engine, conf, NMS |
| `configs/blur.yaml` | pad, blur strength, hold |
| `configs/camera.yaml` | RTSP / ONVIF |
| `configs/encode.yaml` | codec, bitrate |
| `configs/preview.yaml` | live window |
| `configs/logging.yaml` | metrics, samples, detected instances |
| `src/config_loader.py` | `load_all()` |

See [README_CONFIG.md](README_CONFIG.md).

---

## 8. High-FPS RTSP testing

Real live streams only send ~25–30 FPS. To measure true pipeline capacity:

**Terminal 1 – RTSP server (no `-re`)**

```bash
ffmpeg -stream_loop -1 -i video.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/stream
```

Optional MediaMTX assets under `tools/`.

**Terminal 2 – pipeline**

```bash
python3 src/retinaface_face_detection.py \
  --source rtsp://127.0.0.1:8554/stream \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --output live_blurred.mkv \
  --max-seconds 60
```

Details → [docs/md_files/HIGH_FPS_RTSP.md](docs/md_files/HIGH_FPS_RTSP.md)

---

## 9. Documentation

| Document | Description |
|----------|-------------|
| [docs/md_files/INSTALL_DEEPSTREAM.md](docs/md_files/INSTALL_DEEPSTREAM.md) | Install DeepStream + Python (`pyds`) |
| [docs/md_files/HOW_TO_RUN.md](docs/md_files/HOW_TO_RUN.md) | Run RetinaFace (file / RTSP / Rutube / preview) |
| [docs/md_files/MODEL.md](docs/md_files/MODEL.md) | Model / ONNX notes |
| [docs/md_files/RESULTS.md](docs/md_files/RESULTS.md) | Performance notes |
| [docs/md_files/MODULES_GPU.md](docs/md_files/MODULES_GPU.md) | GPU vs CPU placement |
| [docs/md_files/HIGH_FPS_RTSP.md](docs/md_files/HIGH_FPS_RTSP.md) | High-rate RTSP method |
| [docs/md_files/Soak_Test.md](docs/md_files/Soak_Test.md) | Long live run notes |
| [docs/artifacts/](docs/artifacts/) | Handbook A01–A17 pack |
| [README_CONFIG.md](README_CONFIG.md) | YAML config layout |

---

## 10. Repository structure

```text
face/
├── README.md
├── README_CONFIG.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── pipeline.yaml
│   ├── detection.yaml
│   ├── blur.yaml
│   ├── camera.yaml
│   ├── encode.yaml
│   ├── preview.yaml
│   └── logging.yaml
│
├── src/
│   ├── config_loader.py              # load configs/*.yaml
│   ├── retinaface_face_detection.py  # main DeepStream pipeline
│   ├── pytorch_to_onnx_retinaface.py # .pth → ONNX
│   └── GPU_Profilig/
│       └── retinaface_face_blur_gpu_profiling.py  # same pipeline + CPU/GPU profiling report (no YAML)
│
├── docs/
│   ├── RetinaFace-R50_Performance_Report.pdf
│   ├── SCRFD-10G_Performance_Report.pdf      # historical only
│   ├── SCRFD-2.5G_Performance_Report.pdf     # historical only
│   ├── artifacts/                            # A00–A17 handbook
│   └── md_files/
│       ├── HOW_TO_RUN.md
│       ├── INSTALL_DEEPSTREAM.md
│       ├── MODEL.md
│       ├── MODULES_GPU.md
│       ├── RESULTS.md
│       ├── HIGH_FPS_RTSP.md
│       └── Soak_Test.md
│       └── GPU_PROFILING.md
├── results/
│   └── RetinaFace_resnet50/          # example metrics + samples
│       ├── metrics.json
│       ├── RetinaFace_RESNET_50.mp4
│       └── Sample_frames/
│
└── tools/                            # optional high-FPS RTSP helpers
    ├── mediamtx.exe
    ├── mediamtx.yml
    └── LICENSE
```

Historical SCRFD result folders or PDFs may exist for archive comparison only. **Do not use SCRFD scripts for new work.**

---

## 11. Requirements

- Ubuntu 22.04 (or similar)
- NVIDIA GPU (**x86 only** — Jetson not supported for this design)
- DeepStream 7.x with Python bindings (`pyds`)
- CUDA 12.x (or 11.x with matching CuPy)
- **Python 3.10** (official `pyds` wheel)

```bash
pip install -r requirements.txt
```

Typical packages:

```text
numpy
cupy-cuda12x
opencv-python-headless
nvidia-ml-py
requests
PyYAML
```

---

## 12. Known limitations

- Jetson is not supported (`pyds.get_nvds_buf_surface_gpu` path is x86-oriented in this project).
- Real live streams show ~25–30 FPS — limited by the **source**, not necessarily the pipeline.
- Very small / distant faces on wide cameras may be missed (detector limit).
- Engine files are TensorRT-version specific; pass `--onnx` to rebuild if deserialize fails.
- Prefer **MKV** for shareable live outputs; some MP4 paths play poorly outside VLC.

---
## 13. RetinaFace Resnet-50
 - RetinaFace Resnet-50 | [Download](https://drive.google.com/uc?export=download&id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW) |

## License

Internal use – AI Research / Real-time Video Privacy
