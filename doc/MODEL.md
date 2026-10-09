# Model: SCRFD-2.5G

---

## Overview

| Item | Value |
|------|-------|
| Model | SCRFD-2.5G |
| Task | Face Detection |
| Input size | 640 × 640 |
| Precision | FP16 (TensorRT) |
| Backend | TensorRT via DeepStream `nvinfer` |

---

## What is SCRFD?

SCRFD (Sample and Computation Redistribution for efficient Face Detection) is a fast and accurate face detector.

- **2.5G** = lighter version (good speed / accuracy balance)
- **10G** = heavier version (higher accuracy, slower)

We use **SCRFD-2.5G** for real-time performance.

---

## Outputs

The engine produces:

- Score maps (3 strides: 8, 16, 32)
- Bounding box maps
- Keypoint maps (5 facial landmarks)

The Python probe decodes these outputs, applies NMS, and generates face boxes for the blur stage.

---

## Engine File

Typical engine name:

```
scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine
```

Meaning:
- `bnkps` → with landmarks (keypoints)
- `b1` → batch size 1
- `gpu0` → built for GPU 0
- `fp16` → half precision

---

## How to Rebuild Engine

If the engine was built with a different TensorRT version:

```bash
python3 face_blur_deepstream.py \
  --source video.mp4 \
  --engine scrfd_2.5g.engine \
  --onnx scrfd_2.5g_bnkps.onnx
```

DeepStream `nvinfer` will rebuild the engine automatically.
