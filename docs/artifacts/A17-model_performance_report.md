# A17 — Model Performance Report

**ID:** `Inter-69`-MPR-v1  
**Engineer:** Noman  
**Related:** A06, A07, A08, A11  
**Model:** RetinaFace-R50 (pretrained → TensorRT FP16)  

---

## Purpose

Summarises detection runtime performance and resource use for the locked engine.  
Not a training-curve report (no company train).

---

## Model identity

| Field | Value |
| ----- | ----- |
| Architecture | RetinaFace ResNet-50 |
| Input | 1280 × 736 |
| Engine | `retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine` |
| conf (soak) | 0.15 |
| NMS | 0.4 |

---

## Throughput and latency (live soak 2026-10-15 — A11)

| Metric | Value |
| ------ | ----- |
| Source delivery | ~30 FPS |
| Steady / whole-run FPS | 30.0 / 30.0 |
| Frames processed | 393,511 |
| Frames lost | 0 |
| Est. max FPS (pipeline) | ~520 (nvinfer-limited in estimate) |
| Detect mean latency | 1.9 ms |
| Post+NMS mean | 0.2 ms |
| Blur mean | ~0.0 ms (reported round) |
| Encode mean | 0.8 ms |
| E2E mux→file mean | 6.6 ms |

---

## Detection volume (same soak)

| Metric | Value |
| ------ | ----- |
| Total detections | 4,900 |
| Avg faces / frame | 0.01 |
| Max faces / frame | 3 |
| Zero-detect frames | 98.8% |
| Hold frames | 1,412 |

Interpretation: scene often empty or faces too small for reliable fire; not evidence of 30 FPS failure.

---

## Resources (RTX 5080)

| Metric | Value |
| ------ | ----- |
| GPU util avg / peak | 6% / 13% |
| GPU mem peak | 4,889 MB |
| CPU cores avg | 0.24 |
| NVENC / NVDEC avg | 2% / 2% |

---

## Accuracy vs SCRFD (qualitative)

| Model | Speed (observed) | Accuracy vs project need |
| ----- | ---------------- | ------------------------ |
| SCRFD-2.5G | Strong | Not accepted |
| SCRFD-10G | Strong | Not accepted |
| RetinaFace-R50 | Adequate on RTX 5080 at 30 FPS live | **Selected** |


---

## Privacy / blur ops (soak)

| Metric | Value |
| ------ | ----- |
| Audit | PASS |
| Failed | 0 |
| Partial | 0 |

---

## Comparison to production target hardware

| Item | Value |
| ---- | ----- |
| Evidence GPU | RTX 5080 |

---

## Status

Performance numbers above are from the **supplied A11 soak report** only.  
No fabricated accuracy percentages.
