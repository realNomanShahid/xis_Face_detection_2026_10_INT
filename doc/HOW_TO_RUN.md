# How to Run

---

## 1. Prepare the TensorRT Engine

You need a TensorRT engine file for SCRFD-2.5G.

If you already have the `.engine` file:

```bash
--engine scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine
```

If you only have the ONNX file, the script can rebuild the engine:

```bash
python3 face_blur_deepstream.py \
  --source video.mp4 \
  --engine scrfd_2.5g.engine \
  --onnx scrfd_2.5g_bnkps.onnx
```

---

## 2. Run on a Video File (fastest)

```bash
python3 face_blur_deepstream.py \
  --source video.mp4 \
  --engine scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine \
  --output output_blurred.mp4
```

This runs as fast as possible (hundreds of FPS).

---

## 3. Run on RTSP / Live Stream

```bash
python3 face_blur_deepstream.py \
  --source rtsp://127.0.0.1:8554/stream \
  --engine scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine \
  --output live_blurred.mp4 \
  --max-seconds 60 \
  --width 1280 \
  --height 720 \
  --leaky
```

---

## 4. High-FPS Test with Your Own RTSP Server

To measure true maximum performance, create a local high-speed RTSP source.

### Terminal 1 – Start RTSP server

```bash
ffmpeg -stream_loop -1 -i video.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/stream
```

> **Important:** Do **not** use `-re`.  
> `-re` forces real-time speed (~25 FPS).  
> Without `-re`, FFmpeg pushes frames as fast as possible.

### Terminal 2 – Run the pipeline

```bash
python3 face_blur_deepstream.py \
  --source rtsp://127.0.0.1:8554/stream \
  --engine scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine \
  --output live_blurred.mp4 \
  --max-seconds 60 \
  --width 1280 \
  --height 720 \
  --leaky
```

You should see ~400+ FPS.

---

## 5. Useful Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | – | Video file or RTSP/HLS URL |
| `--engine` | – | TensorRT engine file (required) |
| `--onnx` | – | Optional ONNX to rebuild engine |
| `--output` | `output_blurred.mp4` | Output video |
| `--det-size` | `640` | Model input size |
| `--conf` | `0.35` | Detection confidence |
| `--pad` | `0.20` | Face box padding |
| `--blur-frac` | `0.25` | Blur strength |
| `--max-seconds` | `0` | Stop after N seconds (0 = full video) |
| `--width` / `--height` | `1920` / `1080` | Streammux size |
| `--leaky` | off | Drop old frames when slow (good for live) |
| `--sample-every` | `100` | Save before/after images every N frames |
| `--gpu` | `0` | GPU index |

---

## 6. Output Files

| File | Description |
|------|-------------|
| `output_blurred.mp4` | Blurred video |
| `metrics.json` | Full performance metrics |
| `sample_frames/` | Before/after images (if enabled) |

---

## Next Step

See performance → [RESULTS.md](RESULTS.md)
