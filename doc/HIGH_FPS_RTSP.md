# High-FPS RTSP Testing (How We Measured 400+ FPS)

---

## Why We Did This

Real live streams (cameras, Rutube, etc.) only send **25–30 FPS**.  
Our pipeline can process much more. To measure the real maximum speed, we created our own high-speed RTSP source.

---

## How It Works

1. FFmpeg reads a video file.
2. It re-encodes and pushes frames **as fast as possible** over RTSP.
3. DeepStream connects to this local RTSP stream.
4. Pipeline processes frames at full speed.

---

## Step-by-Step

### Terminal 1 – Start local RTSP server

```bash
ffmpeg -stream_loop -1 -i video.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/stream
```

**Important:**  
Do **not** add `-re`.  
`-re` forces real-time speed (~25 FPS).  
Without `-re`, FFmpeg sends frames as fast as it can.

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

---

## Expected Result

| Source | FPS |
|--------|-----|
| Normal live stream | ~25 |
| Our high-rate RTSP (no `-re`) | **~400+** |

This proves the pipeline is not the bottleneck.

---

## Simple Explanation for Others

> We created a local RTSP server with FFmpeg (without the real-time flag) so it pushes frames much faster than a normal camera. This lets us measure the true maximum performance of our face detection + blur pipeline.
