# A06 — Dataset / Model Version (locked runtime identity)

**ID:** `Inter-69`-DSV-v1  
**Engineer:** Noman  
**Date:** `2026-10-9`  
**Related:** A03, A07, A13, A17  

---

## Purpose

Locks the **version identity** of what runs in evaluation and soak.  
For this project that is the **ONNX + TensorRT engine**, not a labelled dataset version .

---

## Locked artifacts

| Asset | Identifier |
| ----- | ---------- |
| Detector family | RetinaFace ResNet-50 |
| ONNX file | `retinaface_r50_1280x736.onnx` |
| Engine file | `retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine` |
| Input size | 1280 × 736 |
| Precision | FP16 (engine name) |
| Batch | 1 |
| GPU index used in name | 0 |

**Checksums / git commit of export scripts:** `[TO FILL]`

---

## Runtime defaults used with this version (evidence runs)

| Parameter | Value used in documented live soak (A11) |
| --------- | ---------------------------------------- |
| conf | 0.15 |
| NMS | 0.4 |
| Source resolution (example) | 1536 × 864 |
| Transport | RTSP TCP, latency 500 ms |

Other experiments may use different conf; **soak A11** is the long-run reference.

---

## Explicitly not locked here

- SCRFD-2.5G / SCRFD-10G engines (discarded; do not ship)  
- Any fine-tuned company checkpoint (none in this version)  

---

## Status

| Field | Value |
| ----- | ----- |
| Issued for | RetinaFace-R50 inference path |
| Parent data version | N/A (pretrained) |
| Engineer | Noman |
