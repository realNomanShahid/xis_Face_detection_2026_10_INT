# How to Run

Simple guide for the **RetinaFace-R50** face detection + GPU blur pipeline.

SCRFD-2.5G and SCRFD-10G were evaluated and **discarded** (accuracy). They are not part of the supported run path.

Before you start, install DeepStream and Python packages → see [INSTALL_DEEPSTREAM.md](INSTALL_DEEPSTREAM.md)

Model details and weight / ONNX notes → see [MODEL.md](MODEL.md)

---

## Supported model

| Item | Value |
|------|--------|
| Model | **RetinaFace ResNet-50** (pretrained → ONNX → TensorRT) |
| Typical input | 1280 × 736 |
| Script | `src/retinaface_face_detection.py` |
| Defaults | `configs/*.yaml` via `src/config_loader.py` |

---

## One-time setup

### 1. Environment

```bash
# example: conda env with pyds / DeepStream bindings
conda activate ds_face   # or your env name

export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0
export GIO_MODULE_DIR=/usr/lib/x86_64-linux-gnu/gio/modules
```

Install YAML support if needed:

```bash
pip install PyYAML
```

### 2. Convert PyTorch weights → ONNX (once)

```bash
python3 src/pytorch_to_onnx_retinaface.py
```

This produces the ONNX file used by DeepStream (name must match `configs/detection.yaml` → `onnx_path`).

### 3. Build TensorRT engine (once)

First run with `--onnx` builds the engine if it is missing:

```bash
python3 src/retinaface_face_detection.py \
  --source path/to/video.mp4 \
  --onnx retinaface_r50_1280x736.onnx \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine
```

Or rely on paths already set in `configs/detection.yaml` and only pass `--source`.

DeepStream / nvinfer creates the `.engine` file automatically. Later runs reuse it.

---

## Config files (preferred)

All important paths and thresholds live under `configs/`:

| File | Controls |
|------|----------|
| `configs/pipeline.yaml` | resolution, `out_dir`, output name, GPU, retime |
| `configs/detection.yaml` | ONNX / engine paths, conf, NMS, input size |
| `configs/blur.yaml` | pad, blur strength, hold frames |
| `configs/camera.yaml` | RTSP / ONVIF camera |
| `configs/encode.yaml` | codec, bitrate |
| `configs/preview.yaml` | live window |
| `configs/logging.yaml` | metrics, samples, `detected_instances` |

CLI flags **override** YAML for a single run.

Loader: `src/config_loader.py` → `load_all()`.

---

## A. Run on a video file

```bash
python3 src/retinaface_face_detection.py \
  --source video.mp4 \
  --onnx retinaface_r50_1280x736.onnx \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --output output_blurred.mkv \
  --width 1280 --height 720 \
  --conf 0.15
```

With defaults already in YAML (minimal CLI):

```bash
python3 src/retinaface_face_detection.py --source video.mp4
```

---

## B. Run on office RTSP camera

Set password in the environment (do not commit secrets):

```bash
export CAM_ONVIF_PASSWORD='your_password'
```

Password characters must be URL-encoded in a hand-built URL (`#` → `%23`).

**Option 1 – full URL on CLI**

```bash
python3 src/retinaface_face_detection.py \
  --source "rtsp://dev_use:Pass123%23@10.11.4.254:554/rtsp_tunnel?p=1&line=1&inst=1&vcd=2" \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --onnx retinaface_r50_1280x736.onnx \
  --width 1536 --height 864 \
  --conf 0.15 \
  --rtsp-protocol tcp \
  --latency 500 \
  --preview \
  --output camera_blurred.mkv
```

**Option 2 – `configs/camera.yaml` + env password**

```bash
python3 src/retinaface_face_detection.py --preview
```

(Requires `source` / camera fields configured so the script can build the RTSP URI.)

---

## C. Run on Rutube (HLS)

```bash
python3 src/retinaface_face_detection.py \
  --rutube-page "https://rutube.ru/live/video/<id>/" \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --onnx retinaface_r50_1280x736.onnx \
  --width 1280 --height 720 \
  --max-height 720 \
  --conf 0.15 \
  --out-fps 12 \
  --max-seconds 60 \
  --preview
```

Use `--out-fps` close to the delivered live rate so the saved video duration matches wall time.

---

## D. Live preview

```bash
python3 src/retinaface_face_detection.py \
  --source "rtsp://..." \
  --preview \
  --preview-size 1000x1000
```

- Needs a desktop session (`DISPLAY`) and GStreamer X plugin (`gstreamer1.0-x`).
- Overlay shows **Persons** (faces in current frame) and **FPS**.
- Boxes are preview-only; they are **not** burned into the saved video.
- Disable boxes: `--no-preview-boxes`
- Force off if YAML has preview enabled: `--no-preview`

---

## E. High-FPS test (optional)

Measures maximum pipeline speed. Not the normal camera rate.

**Terminal 1 – local RTSP (no `-re`)**

```bash
ffmpeg -stream_loop -1 -i video.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/stream
```

Optional: MediaMTX under `tools/` (see [HIGH_FPS_RTSP.md](HIGH_FPS_RTSP.md)).

**Terminal 2 – pipeline**

```bash
python3 src/retinaface_face_detection.py \
  --source rtsp://127.0.0.1:8554/stream \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --output live_blurred.mkv \
  --max-seconds 60 \
  --width 1280 --height 720
```

More detail → [HIGH_FPS_RTSP.md](HIGH_FPS_RTSP.md)

---

## Useful arguments

| Argument | Config key (typical) | Description |
|----------|----------------------|-------------|
| `--source` | `pipeline.source` / `camera.*` | File, RTSP, or HLS URL |
| `--rutube-page` | `pipeline.rutube_page` | Resolve Rutube page → m3u8 |
| `--onnx` | `detection.onnx_path` | ONNX for engine build |
| `--engine` | `detection.engine_path` | TensorRT engine |
| `--conf` | `detection.conf` | Face score threshold |
| `--nms` | `detection.nms` | NMS IoU |
| `--pad` | `blur.pad` | Box padding |
| `--blur-frac` | `blur.blur_frac` | Blur strength |
| `--hold` | `blur.hold_frames` | Reuse last boxes on short misses |
| `--output` | `pipeline.output` | Prefer `.mkv` for shareable files |
| `--out-dir` | `pipeline.out_dir` | Logs / video / detection images |
| `--width` / `--height` | `pipeline.width` / `height` | nvstreammux size |
| `--rtsp-protocol` | `camera.rtsp_protocol` | `tcp` recommended |
| `--latency` | `camera.latency_ms` | RTSP jitter buffer (ms) |
| `--preview` | `preview.enabled` | Live window |
| `--max-seconds` | `pipeline.max_seconds` | 0 = until end / Ctrl+C |
| `--out-fps` | `pipeline.out_fps` | Steady timestamps for saved video |
| `--sample-every` | `logging.sample_every` | Periodic before/after samples |
| `--no-save-detections` | inverse of `logging.save_detections` | Skip `detected_instances/` |
| `--gpu` | `pipeline.gpu_id` | GPU index |

---

## Output files

Typical location: under `logs/` when `out_dir: logs` (or `results/` if you change config).

| File / folder | Description |
|---------------|-------------|
| `*.mkv` / `*.mp4` | Blurred video (MKV recommended) |
| `metrics.json` | Performance + privacy-audit summary |
| `detection_logs.jsonl` | Per-frame detections |
| `privacy_audit.jsonl` | Per-frame blur audit |
| `detected_instances/` | Frames where a face was detected and blurred (on by default) |
| `sample_frames/` | Optional periodic before/after pairs |

Example archived results for this model: `results/RetinaFace_resnet50/`.

---

## Stop and report

Press **Ctrl+C** at any time. The pipeline sends EOS, finalises the video, and prints the performance report.

---

## Quick checklist

1. Install DeepStream → [INSTALL_DEEPSTREAM.md](INSTALL_DEEPSTREAM.md)
2. Convert weights → `python3 src/pytorch_to_onnx_retinaface.py`
3. Set `configs/detection.yaml` paths (ONNX / engine)
4. Build engine once (first run with ONNX)
5. Run file / RTSP / Rutube with `src/retinaface_face_detection.py`
6. Prefer **`.mkv`** for saved live video
7. See results → [RESULTS.md](RESULTS.md) and `results/RetinaFace_resnet50/`

---

## Not supported (discarded)

| Model | Status |
|-------|--------|
| SCRFD-2.5G | Evaluated; **not** the production path |
| SCRFD-10G | Evaluated; **not** the production path |

Historical comparison PDFs may remain under `docs/` for the archive only. Do not use SCRFD scripts for new runs.
