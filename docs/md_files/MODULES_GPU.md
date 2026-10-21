# Module Execution Placement

Almost the entire pipeline runs on **GPU**.

---

## Pipeline Overview

```
nvurisrcbin (NVDEC decode)
    → nvstreammux
    → nvinfer (TensorRT SCRFD)
    → nvvideoconvert (to RGBA)
    → Python probe (CUDA blur kernels)
    → nvvideoconvert (to NV12)
    → nvv4l2h264enc (NVENC)
    → filesink
```

---

## What Runs Where

| Module | Device | Notes |
|--------|--------|-------|
| Stream Decode (NVDEC) | **GPU** | Hardware video decode |
| Face Detection (TensorRT / .Engine) | **GPU** | TensorRT engine, FP16 |
| Post-processing (decode + NMS) | **GPU** | tensors |
| Face Blur (Custom CUDA kernels) | **GPU** | Separable box blur inside ellipse |
| Video Encode (NVENC) | **GPU** | Hardware H.264 / H.265 encode |
| Stream Mux / Queue (DeepStream) | **GPU** | Zero-copy GPU buffers |

---

## Why CPU usage is low

- Only the SCRFD **output tensors** (~1 MB) are copied to CPU for decoding boxes.
- All heavy work (decode, inference, blur, encode) stays on GPU.
- Frames never leave GPU memory except for that small tensor copy.

Typical result:
- CPU cores used: **~0.17**
- GPU utilisation: **~35%** 

---

## CUDA Blur Kernels

Two custom CUDA kernels:

1. **Horizontal blur** – average along X inside each face box
2. **Vertical blur** – average along Y, only inside the face ellipse

Both run fully on GPU. All faces in a frame are processed in two kernel launches.
