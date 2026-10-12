# How to Run

Simple guide for SCRFD-2.5G, SCRFD-10G, and RetinaFace-R50.

Before you start, install DeepStream and Python packages → see [INSTALL_DEEPSTREAM.md](INSTALL_DEEPSTREAM.md)

Model details and ONNX download links → see [MODEL.md](MODEL.md)

---

## A. SCRFD-2.5G and SCRFD-10G

### Step 1 – Get the ONNX file

Download from [MODEL.md](MODEL.md), or use your own:

- SCRFD 2.5G → `scrfd_2.5g_bnkps_shape640x640.onnx`
- SCRFD 10G → `scrfd_10g_bnkps.onnx`

### Step 2 – Build TensorRT engine (only once)

```bash
# SCRFD 2.5G
python3 scrfd_face_detection.py \
  --source video.mp4 \
  --engine scrfd_2.5g.engine \
  --onnx scrfd_2.5g_bnkps_shape640x640.onnx

# SCRFD 10G
python3 scrfd_face_detection.py \
  --source video.mp4 \
  --engine scrfd_10g.engine \
  --onnx scrfd_10g_bnkps.onnx
```

DeepStream will create the `.engine` file automatically.

### Step 3 – Run on a video file

```bash
# SCRFD 2.5G
python3 scrfd_face_detection.py \
  --source video.mp4 \
  --engine scrfd_2.5g.engine \
  --output output_blurred.mp4

# SCRFD 10G
python3 scrfd_face_detection.py \
  --source video.mp4 \
  --engine scrfd_10g.engine \
  --output output_blurred.mp4
```

### Step 4 – Run on RTSP / live stream

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

---

## B. RetinaFace-R50

RetinaFace needs one extra step: convert the PyTorch `.pth` weights to ONNX first.

### Step 1 – Convert .pth → ONNX

```bash
python3 pytorch_to_onnx_retinaface.py
```

This creates the ONNX file (for example `retinaface_r50.onnx`).

### Step 2 – Build TensorRT engine (only once)

```bash
python3 retinaface_face_detection.py \
  --source video.mp4 \
  --engine retinaface_r50.engine \
  --onnx retinaface_r50.onnx
```

### Step 3 – Run on a video file

```bash
python3 retinaface_face_detection.py \
  --source video.mp4 \
  --engine retinaface_r50.engine \
  --output output_blurred.mp4
```

### Step 4 – Run on RTSP / live stream

```bash
python3 retinaface_face_detection.py \
  --source rtsp://127.0.0.1:8554/stream \
  --engine retinaface_r50.engine \
  --output live_blurred.mp4 \
  --max-seconds 60 \
  --width 1280 \
  --height 720 \
  --leaky
```

---

## C. High-FPS Test (optional)

To measure maximum pipeline speed, use a local high-rate RTSP server.

**Terminal 1 – start RTSP server**

```bash
ffmpeg -stream_loop -1 -i video.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/stream
```

> Do **not** use `-re`. That forces real-time (~25 FPS).

**Terminal 2 – run pipeline** (SCRFD or RetinaFace)

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

More detail → [HIGH_FPS_RTSP.md](HIGH_FPS_RTSP.md)

---

## Useful Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | – | Video file or RTSP/HLS URL |
| `--engine` | – | TensorRT engine file (required) |
| `--onnx` | – | Optional ONNX to rebuild engine |
| `--output` | `output_blurred.mp4` | Output video |
| `--det-size` | `640` | Model input size (SCRFD) |
| `--conf` | `0.35` | Detection confidence |
| `--pad` | `0.20` | Face box padding |
| `--blur-frac` | `0.25` | Blur strength |
| `--max-seconds` | `0` | Stop after N seconds (0 = full) |
| `--width` / `--height` | `1920` / `1080` | Streammux size |
| `--leaky` | off | Drop old frames when slow (good for live) |
| `--sample-every` | `100` | Save before/after images every N frames |
| `--gpu` | `0` | GPU index |

---

## Output Files

| File | Description |
|------|-------------|
| `output_blurred.mp4` | Blurred video |
| `metrics.json` | Full performance metrics |
| `sample_frames/` | Before/after images (if enabled) |

---

## Quick Checklist

1. Install DeepStream → [INSTALL_DEEPSTREAM.md](INSTALL_DEEPSTREAM.md)
2. Download ONNX → [MODEL.md](MODEL.md)
3. **SCRFD:** run `scrfd_face_detection.py` with `--onnx` once to build engine, then run normally
4. **RetinaFace:** run `pytorch_to_onnx_retinaface.py` first, then `retinaface_face_detection.py`
5. See results → [RESULTS.md](RESULTS.md)
