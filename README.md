# Face Detection + GPU Blur (DeepStream + TensorRT)

This repository contains a real-time face detection and blurring pipeline built with **NVIDIA DeepStream**, **TensorRT**, and custom **CUDA** blur kernels.

The goal is privacy-preserving video processing: detect faces and blur them on the GPU with low latency and high throughput.

The final measured result on a high-rate RTSP source was approximately **408.50 processed FPS** on an NVIDIA GeForce RTX 5080, with almost the entire pipeline running on GPU.

---

## Contents

1. Project outcome  
2. Model identity  
3. Pipeline overview  
4. Module execution placement  
5. Performance results  
6. High-FPS RTSP testing method  
7. How to run  
8. Folder guide  
9. Requirements  
10. Known limitations  

---

## 1. Project outcome

The work followed this sequence:

1. A SCRFD-2.5G TensorRT engine was prepared for face detection.
2. A DeepStream pipeline was built: decode → detect → GPU blur → encode.
3. The pipeline was first tested on a normal video file (hundreds of FPS).
4. The same pipeline was tested on real live streams (Rutube / RTSP), which delivered only ~25 FPS because the source itself was limited.
5. A local high-rate RTSP server was created with FFmpeg (without the real-time flag) to push frames as fast as possible.
6. On that high-rate source the pipeline reached **~408.50 FPS** with very low latency and almost zero dropped frames.

### Final decision

| Decision area              | Selected result                          | Reason                                                                 |
|----------------------------|------------------------------------------|------------------------------------------------------------------------|
| Face detector              | SCRFD-2.5G (TensorRT FP16)               | Good speed / accuracy balance for real-time use                        |
| Blur method                | Custom CUDA kernels (GPU)                | Fast elliptical box blur, all faces processed in two kernel launches   |
| Runtime                    | DeepStream + TensorRT                    | Full GPU pipeline with zero-copy buffers                               |
| Measured high-rate FPS     | **408.50**                               | Pipeline easily keeps up; GPU still has headroom                       |
| Real live-stream behaviour | Limited by source (~25–30 FPS)           | Normal cameras / Rutube only send frames at their native rate          |

---

## 2. Model identity

| Item              | Value                                      |
|-------------------|--------------------------------------------|
| Model             | SCRFD-2.5G                                 |
| Task              | Face detection                             |
| Input size        | 640 × 640                                  |
| Precision         | FP16 (TensorRT)                            |
| Backend           | TensorRT via DeepStream `nvinfer`          |
| Typical engine    | `scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine`|

SCRFD (Sample and Computation Redistribution for efficient Face Detection) is a fast face detector.  
The **2.5G** variant is the lighter version chosen for real-time performance.

The engine produces score maps, bounding-box maps, and optional keypoint maps.  
A lightweight Python probe decodes the outputs, applies NMS, and generates face boxes for the blur stage.

---

## 3. Pipeline overview

```
nvurisrcbin (NVDEC decode)
    → nvstreammux
    → nvinfer (TensorRT SCRFD-2.5G)
    → nvvideoconvert (to RGBA)
    → Python probe (CUDA blur kernels)
    → nvvideoconvert (to NV12)
    → nvv4l2h264enc / nvv4l2h265enc (NVENC)
    → filesink
```

Frames stay in GPU memory for almost the entire path.  
Only the small SCRFD output tensors are copied to CPU for box decoding and NMS.

---

## 4. Module execution placement

| Module                                      | Device              | Notes                                      |
|---------------------------------------------|---------------------|--------------------------------------------|
| Stream Decode (NVDEC)                       | **GPU**             | Hardware video decode                      |
| Face Detection (TensorRT / SCRFD-2.5G)      | **GPU**             | TensorRT engine, FP16                      |
| Post-processing (decode + NMS)              | CPU (lightweight)   | Tiny tensors only                          |
| Face Blur (Custom CUDA kernels)             | **GPU**             | Separable box blur inside face ellipse     |
| Video Encode (NVENC)                        | **GPU**             | Hardware H.264 / H.265 encode              |
| Stream Mux / Queue (DeepStream)             | **GPU**             | Zero-copy GPU buffers                      |

Almost the entire pipeline runs on GPU.  
Typical measured CPU usage is only ~2.17 cores while processing 400+ FPS.

---

## 5. Performance results

**Device:** NVIDIA GeForce RTX 5080  
**Model:** SCRFD-2.5G  
**Detection size:** 640 × 640  

### High-rate RTSP test

| Metric                  | Value     |
|-------------------------|-----------|
| Avg Pipeline FPS        | **408.50**|
| Source FPS              | 428.60    |
| Frames Processed        | 14332     |
| Frames Dropped          | 6         |
| Headroom                | 2.58×     |
| Estimated Max FPS       | 1052.89   |

### Latency per frame (ms)

| Stage                         | mean  | max   |
|-------------------------------|-------|-------|
| Detect (nvinfer / TensorRT)   | 0.58  | 2.53  |
| Post (decode + NMS)           | 0.78  | 3.94  |
| Blur (CUDA kernels)           | 0.17  | 2.06  |
| Encode (NVENC)                | 0.77  | 5.42  |

### Resource usage

| Metric                        | Value          |
|-------------------------------|----------------|
| CPU cores used (avg)          | 2.17           |
| GPU utilisation avg / peak    | 15.03% / 21%   |
| GPU memory peak               | 4050 MB        |

### Key takeaways

- The pipeline easily keeps up with high-rate sources.
- The GPU still has significant headroom at 400+ FPS.
- Real live streams (cameras, Rutube, etc.) are limited by their own frame rate (usually 25–30 FPS). The pipeline itself is much faster.

---

## 6. High-FPS RTSP testing method

Real live streams only send 25–30 frames per second.  
To measure the true maximum speed of the pipeline, a local high-rate RTSP server was created with FFmpeg.

### Why this was needed

| Source                        | Observed FPS | Reason                              |
|-------------------------------|--------------|-------------------------------------|
| Normal live stream (Rutube)   | ~25          | Source only sends 25 FPS            |
| Local RTSP **with** `-re`     | ~25          | `-re` forces real-time speed        |
| Local RTSP **without** `-re`  | **~408**     | FFmpeg pushes frames as fast as possible |

### Command used for high-rate source

```bash
ffmpeg -stream_loop -1 -i video.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/stream
```

**Important:** Do not use the `-re` flag.  
`-re` forces real-time playback. Removing it allows FFmpeg to send frames at maximum speed.

---

## 7. How to run

### Quick start (video file)

```bash
python3 face_blur_deepstream.py \
  --source video.mp4 \
  --engine scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine \
  --output output_blurred.mp4
```

### Live / RTSP

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

### Useful arguments

| Argument          | Default               | Description                              |
|-------------------|-----------------------|------------------------------------------|
| `--source`        | –                     | Video file or RTSP / HLS URL             |
| `--engine`        | –                     | TensorRT engine file (required)          |
| `--onnx`          | –                     | Optional ONNX to rebuild engine          |
| `--output`        | `output_blurred.mp4`  | Output video                             |
| `--det-size`      | `640`                 | Model input size                         |
| `--conf`          | `0.35`                | Detection confidence                     |
| `--max-seconds`   | `0`                   | Stop after N seconds (0 = full)          |
| `--leaky`         | off                   | Drop old frames when slow (good for live)|
| `--gpu`           | `0`                   | GPU index                                |

Full install and run instructions are in:

- [docs/INSTALL_DEEPSTREAM.md](docs/INSTALL_DEEPSTREAM.md)
- [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)

---

## 8. Folder guide

| Path / file                  | Purpose                                      |
|------------------------------|----------------------------------------------|
| `face_blur_deepstream.py`    | Main pipeline script                         |
| `README.md`                  | This document                                |
| `requirements.txt`           | Python dependencies                          |
| `docs/INSTALL_DEEPSTREAM.md` | How to install DeepStream and packages       |
| `docs/HOW_TO_RUN.md`         | How to run on file / RTSP / high-FPS test    |
| `docs/MODULES_GPU.md`        | Which modules run on GPU vs CPU              |
| `docs/RESULTS.md`            | Performance results                          |
| `docs/MODEL.md`              | SCRFD-2.5G model details                     |
| `docs/HIGH_FPS_RTSP.md`      | How the high-rate RTSP test was performed    |

---

## 9. Requirements

- Ubuntu 22.04 (or similar)
- NVIDIA GPU (x86 only — **Jetson is not supported**)
- DeepStream 7.x or 8.x with Python bindings (`pyds`)
- CUDA 12.x (or 11.x)
- Python 3.8+

Python packages (see `requirements.txt`):

```
numpy
cupy-cuda12x
opencv-python-headless
nvidia-ml-py
requests
```

For CUDA 11.x use `cupy-cuda11x` instead of `cupy-cuda12x`.

---

## 10. Known limitations

- Jetson devices are not supported (the code uses an x86-only GPU surface API).
- Real live streams (cameras, Rutube, etc.) will show ~25–30 FPS. This is limited by the source, not by the pipeline.
- The high-rate RTSP test is intended only for measuring maximum pipeline throughput.
- Engine files are TensorRT-version specific. If the engine was built with a different TensorRT version, pass the ONNX file so `nvinfer` can rebuild it.

---

## License

Internal use – AI Research / Real-time Video Privacy
