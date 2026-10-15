# A03 — Dataset Inventory

**ID:** `Inter-69`-DS-INV-v1  
**Engineer:** Noman  
**Date:** `2026-10-9`  
**Related:** A04, A06, A07  

---

## Purpose

Inventory of **inputs used for this project**.  
This is an **inference deployment** project: no project training set was collected for RetinaFace.

---

## Model assets (not a labelled train set)

| Item | Description | Location / note |
| ---- | ----------- | --------------- |
| Pretrained PyTorch weights | Official RetinaFace ResNet-50 (from public RetinaFace / ResNet repo as used by engineer) |
| ONNX | Exported for nvinfer | `retinaface_r50_1280x736.onnx` (project tree) |
| TensorRT engine | FP16, batch 1, GPU 0 | `retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine` |

---

## Runtime / evaluation sources (streams & files)

| Source | Type | Notes |
| ------ | ---- | ----- |
| Office IP camera | RTSP via ONVIF | IP `10.11.4.254`; profiles include 3840×2160 and 1536×864; soak used 1536×864 class stream |
| Rutube live / VOD | HLS (page → m3u8) | Used during integration tests |
| Local video files | File URI | e.g. `video.mp4` style tests |
| Local RTSP stress | RTSP | High-rate FFmpeg/MediaMTX for max-FPS measurement (not production FPS) |

**Counts / hours of labelled faces for training:** N/A (pretrained).

---

## Outputs produced by runs (not inputs)

| Output | Purpose |
| ------ | ------- |
| `*.mkv` / `*.mp4` | Blurred video |
| `metrics.json` | Performance report |
| `detection_logs.jsonl` | Per-frame detection summary |
| `privacy_audit.jsonl` | Per-frame blur audit |
| `detected_instances/` | JPEG frames where detection+blur occurred |

Store location in runs: typically under `logs/` (`--out-dir`).

---

## Coverage statement

| Question | Answer |
| -------- | ------ |
| Project-owned face annotation dataset? | **No** |
| SCRFD training data in inventory? | **No** — SCRFD discarded |


---

## Status

Inventory for **runtime assets and test sources**.  
Training corpus inventory: **N/A**.
