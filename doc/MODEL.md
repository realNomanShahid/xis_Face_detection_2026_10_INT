# Models

This project supports three face detection models, all running as TensorRT engines via DeepStream `nvinfer`.

---

## 1. SCRFD-2.5G

| Item | Value |
|------|-------|
| Model | SCRFD-2.5G |
| Task | Face Detection |
| Input size | 640 × 640 |
| Precision | FP16 (TensorRT) |
| Backend | TensorRT via DeepStream `nvinfer` |
| Speed | Fastest (~398.5 FPS) |

**What is SCRFD?**  
SCRFD (Sample and Computation Redistribution for efficient Face Detection) is a fast and accurate face detector.

- **2.5G** = lighter version (good speed / accuracy balance)
- **10G** = heavier version (higher accuracy, slightly slower)

**ONNX download:**  
[scrfd_2.5g_bnkps_shape640x640.onnx](https://media.githubusercontent.com/media/cysin/scrfd_onnx/main/scrfd_2.5g_bnkps_shape640x640.onnx)

**Typical engine name:**
```
scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine
```

---

## 2. SCRFD-10G

| Item | Value |
|------|-------|
| Model | SCRFD-10G |
| Task | Face Detection |
| Input size | 640 × 640 |
| Precision | FP16 (TensorRT) |
| Backend | TensorRT via DeepStream `nvinfer` |
| Speed | Fast (~381.4 FPS) |

Heavier SCRFD variant. Slightly more detections than 2.5G, still very fast.

**ONNX download:**  
[scrfd_10g_bnkps.onnx](https://media.githubusercontent.com/media/cysin/scrfd_onnx/main/scrfd_10g_bnkps.onnx)

**Typical engine name:**
```
scrfd_10g_bnkps.onnx_b1_gpu0_fp16.engine
```

---

## 3. RetinaFace-R50

| Item | Value |
|------|-------|
| Model | RetinaFace-R50 (ResNet-50) |
| Task | Face Detection |
| Input size | 1280 × 736 |
| Precision | FP16 (TensorRT) |
| Backend | TensorRT via DeepStream `nvinfer` |
| Speed | Good (~346.1 FPS) |

Detects the most faces among the three models, but uses more GPU and is slower than SCRFD.

**.pth download:**  
[RetinaFace-R50 .pth](https://drive.google.com/uc?export=download&id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW)

---

## Outputs (SCRFD)

The SCRFD engines produce:

- Score maps (3 strides: 8, 16, 32)
- Bounding box maps
- Keypoint maps (5 facial landmarks)

The Python probe decodes these outputs, applies NMS, and generates face boxes for the blur stage.

---

## How to Rebuild Engine

If the engine was built with a different TensorRT version, pass the ONNX so DeepStream `nvinfer` rebuilds it:

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

---

## Quick Comparison

| Model | Input Size | Avg FPS | Total Detections | GPU util |
|-------|------------|---------|------------------|----------|
| SCRFD 2.5G | 640×640 | **398.5** | 42,802 | 18.3% |
| SCRFD 10G | 640×640 | 381.4 | 45,243 | 30.0% |
| RetinaFace-R50 | 1280×736 | 346.1 | **77,585** | 74.6% |

**Recommendation:** Use **SCRFD-2.5G** for real-time performance. Use **RetinaFace-R50** when maximum face recall is more important than speed.
