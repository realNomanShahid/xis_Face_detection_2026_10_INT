# High-FPS RTSP Testing (How We Measured 400+ FPS)

---

## Why We Did This

Real live streams (cameras, Rutube, etc.) usually send **25–30 FPS**.

Our pipeline can process frames much faster than that. To measure its maximum processing speed, we created our own high-speed RTSP source using **FFmpeg and MediaMTX**.

MediaMTX acts as the RTSP server, FFmpeg pushes video frames into it as fast as possible, and DeepStream reads the stream for processing.

---

## How It Works

1. **MediaMTX** runs as a local RTSP server and accepts incoming video streams.
2. **FFmpeg** reads a video file and re-encodes it using H.264.
3. FFmpeg publishes the video to MediaMTX over RTSP without real-time throttling.
4. **DeepStream** connects to the local RTSP stream.
5. The pipeline performs face detection and blurring at the maximum achievable processing speed.
6. We measure the FPS to evaluate the pipeline's performance independently of a normal camera's frame rate.

---

## Prerequisites

Make sure you have the following installed:

* FFmpeg
* MediaMTX
* Python 3 and the dependencies required by your DeepStream pipeline
* Your face detection model and TensorRT engine (if required by your pipeline)

---

## Step-by-Step Setup

### Terminal 1 – Start the MediaMTX RTSP Server

MediaMTX provides the RTSP endpoint that FFmpeg publishes to and DeepStream reads from.

Download MediaMTX from its official release page:

https://github.com/bluenviron/mediamtx/releases

Extract the downloaded archive and navigate to the directory containing the MediaMTX executable.

For example:

```bash
cd mediamtx
```

Start the server:

```bash
./mediamtx
```

On the first run, MediaMTX uses its default configuration. By default, it listens for RTSP connections on port `8554`.

Keep this terminal running.

The RTSP server endpoint will be:

```text
rtsp://127.0.0.1:8554/stream
```

**Important:** MediaMTX is the RTSP server; it does not generate or encode the video itself. FFmpeg will publish the video to it in the next step.

If MediaMTX is installed elsewhere, use the actual path to its executable.

---

### Terminal 2 – Publish the High-Speed Video Using FFmpeg

Open a second terminal on the same machine.

Run FFmpeg to read a video file, encode it, and publish it to MediaMTX:

```bash
ffmpeg -stream_loop -1 -i video.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/stream
```

Replace `video.mp4` with the path to your input video.

**Important: Do not add `-re`.**

* With `-re`, FFmpeg reads the input at approximately its original playback rate.
* Without `-re`, FFmpeg can read and publish frames as fast as the encoding, network, and receiving pipeline allow.
* `-stream_loop -1` continuously loops the input video.
* `-preset ultrafast` reduces H.264 encoding overhead.
* `-tune zerolatency` configures the encoder for low-latency operation.
* `-an` disables audio.
* `-rtsp_transport tcp` uses TCP for RTSP media transport.

Keep this terminal running while the test is in progress.

**Note:** FFmpeg still has to encode every frame. Its output rate depends on CPU performance, video resolution, encoder settings, and the RTSP server and client.

---

### Terminal 3 – Run the DeepStream Pipeline

Open a third terminal and run your face detection and blurring pipeline:

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

This starts the pipeline with the local RTSP stream as its input.

The pipeline reads frames from MediaMTX, performs face detection and blurring, and writes the output to `live_blurred.mp4`.

The `--leaky` option is pipeline-specific; in configurations where it enables dropping queued frames, it can help prevent latency from growing when processing falls behind.

---

## Expected Results

| Source                     |                          Expected FPS |
| -------------------------- | ------------------------------------: |
| Normal live camera stream  |                            ~25–30 FPS |
| High-rate RTSP test source |                  Potentially 400+ FPS |
| DeepStream pipeline        | Depends on actual processing capacity |

**Important:** 400+ FPS is a target or observed result, not a guaranteed outcome. FFmpeg must publish frames quickly enough, and MediaMTX and DeepStream must handle the resulting stream without becoming bottlenecks.

To establish the pipeline's actual maximum processing speed, compare the measured processing FPS with the incoming frame rate and check whether frames are being dropped.

---

## How to Verify the Setup

1. Confirm that MediaMTX starts successfully in Terminal 1.
2. Confirm that FFmpeg connects to MediaMTX and begins publishing in Terminal 2.
3. Confirm that DeepStream connects to `rtsp://127.0.0.1:8554/stream` in Terminal 3.
4. Check the pipeline's FPS logs or performance metrics.
5. Monitor CPU and GPU utilization to identify possible bottlenecks.

If FFmpeg cannot publish the stream, verify that MediaMTX is running and that port `8554` is available.

If DeepStream cannot connect, check the RTSP URL and confirm that the publisher is active.

---

## Simple Explanation for Others

> We used MediaMTX as a local RTSP server and FFmpeg as a high-speed video publisher. FFmpeg continuously reads and encodes a video file without the real-time `-re` flag, allowing it to publish frames faster than a normal camera. DeepStream connects to this stream and runs face detection and blurring, allowing us to measure the pipeline's maximum achievable processing performance.

---

## Architecture Overview

```text
                 Terminal 1
              ┌──────────────┐
              │   MediaMTX   │
              │  RTSP Server │
              │  Port: 8554  │
              └──────┬───────┘
                     │
             RTSP video stream
                     │
              ┌──────┴───────┐
              │    FFmpeg    │
              │ Video Input  │
              │ H.264 Encode │
              │ No -re flag  │
              └──────────────┘
                 Terminal 2

                     │
             Published RTSP stream
                     │
              ┌──────▼───────┐
              │  DeepStream  │
              │ Face Detect  │
              │ Face Blur    │
              │ FPS Measure  │
              └──────┬───────┘
                     │
              live_blurred.mp4
                 Terminal 3
```

**Key takeaway:** MediaMTX handles RTSP streaming, FFmpeg generates and publishes the high-rate video, and DeepStream processes the incoming frames. This setup helps benchmark the pipeline beyond the normal 25–30 FPS limits of live cameras.
