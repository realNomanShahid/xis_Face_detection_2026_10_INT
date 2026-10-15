# A01 — Question Set

**ID:** `Inter-69`-QS  
**Engineer:** Noman  
**Date:** 2026-10-07 (project start)  
**Lead context:** Abdul Moiz  
**Related:** A02, A14  

---

## Purpose of this artifact

Frames the problem before the detector and delivery path are locked.  
Records what decision the system supports, what is observed, what data enters the system, and under what operating conditions it must run.

---

## PURPOSE

### Decision the system supports

- Detect human faces in live and file video.
- Apply GPU ellipse blur on each detected face so identity is not visible in the **saved** output and in the optional live preview policy.
- Prefer **detection and blur accuracy** over maximum throughput (speed still considered).

### What happens to frames with faces

- Faces above the confidence threshold are blurred in the pipeline before encode.
- Frames with no detection may still use a short **hold** of previous boxes (configurable) so brief misses do not flash unblurred faces.
- Privacy audit log records applied / hold / skipped / failed per frame.

### Why this exists

Unblurred faces in office / public streams create privacy risk.  
Manual editing does not scale to live RTSP or continuous recording.  
The package provides a repeatable, logged, GPU-accelerated blur path.

---

## MEASURED

| Quantity | Meaning |
| -------- | ------- |
| Face presence | RetinaFace-R50 boxes above `--conf` after NMS |
| Blur application | CUDA separable box blur inside ellipse ROIs |
| Stream health | FPS delivered, frames lost, gaps, reconnects |
| Resources | GPU util, NVENC/NVDEC, memory, CPU cores |
| Privacy audit | applied / from_previous_frame / partial / skipped / failed |

---

## DATA IN

| Source type | Examples used in this project |
| ----------- | ----------------------------- |
| Live RTSP | Office IP camera (`10.11.4.254`, ONVIF → RTSP tunnel) |
| HLS / web live | Rutube live pages resolved to `.m3u8` |
| File | Local `.mp4` clips |
| Local RTSP stress | FFmpeg / MediaMTX high-rate push (benchmark, not production rate) |

**Model weights:** Pretrained RetinaFace ResNet-50 (official PyTorch → ONNX → TensorRT FP16). **No project training or fine-tune.**

---

## SCALE

| Item | Known value |
| ---- | ----------- |
| Typical live delivery | ~30 FPS from office camera profile used in soak |
| Engine input | 1280 × 736 |
| Example stream resolution | 1536 × 864 (camera profile 1) |
| Hardware used | NVIDIA GeForce RTX 5080 (16 GB class) |
| Stack | DeepStream 7.1, TensorRT via nvinfer, CUDA blur, NVENC |

---

## OUT OF SCOPE (this package)

- Training or fine-tuning RetinaFace on company data  
- SCRFD-2.5G / SCRFD-10G production path (discarded; see A14)  
- Face recognition / identity matching  
- PLC / physical access control  

---

## Early model research (context only)

| Candidate | Outcome |
| --------- | ------- |
| SCRFD-2.5G | Good speed; **accuracy not accepted** vs RetinaFace on tested streams |
| SCRFD-10G | Good speed; **accuracy not accepted** vs RetinaFace |
| RetinaFace-R50 | **Selected** — accuracy priority; speed still adequate on RTX 5080 |

Formal discard decision: **A14**.

---

  
