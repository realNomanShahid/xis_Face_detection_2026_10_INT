# A02 — Objective

**ID:** `Inter-69`-OBJ-v1  
**Parent:** `Inter-69`-QS (A01)  
**Engineer:** Noman  
**Confirmed by:** Abdul Moiz —   
**Related:** A01, A08, A09, A11, A12, A13, A17  

---

## Purpose

Defines what the Face Detection and Blur system must do, how success is observed, and under which conditions evaluation is valid.  
Parent target for evaluation, validation, soak, and release evidence.

---

## Confirmation

| Field | Value |
| ----- | ----- |
| Engineer | Noman |
| Confirmed by | Abdul Moiz |
| Parent question set | A01 |
| Confirmation date | `---` |

---

## Seven elements

| Element | Content |
| ------- | ------- |
| **Subject** | Live and file video face detection + GPU face blur pipeline (DeepStream, TensorRT RetinaFace-R50, CUDA ellipse blur, NVENC). |
| **Observable** | Faces above threshold are blurred in the output stream/file; optional live preview; metrics and privacy-audit logs written. |
| **Measure** | Detection conf threshold; NMS; faces per frame; blur applied / hold / failed; steady FPS vs source; frame loss; GPU memory; stage latency. |
| **Target** | **Priority 1 — accuracy:** reliable face detection and blur on in-scope streams (office RTSP, file, agreed live tests). **Priority 2 — speed:** sustain source FPS with headroom on target GPU; no systematic frame loss under normal network conditions. Privacy audit **PASS** on recorded runs (zero failed blur applications when faces detected). |
| **Method** | Pretrained RetinaFace-R50 → ONNX → TensorRT FP16 engine; DeepStream `nvinfer`; post-decode NMS; CUDA blur kernels; hold of last boxes for short misses; RTSP TCP + latency buffer for live. |
| **Conditions** | x86 Linux + NVIDIA GPU (project evidence on RTX 5080); DeepStream 7.x; live RTSP/HLS and file sources as tested; conf and resolution as recorded in A06/A11. |
| **Done looks like** | Pipeline runs sustained live; faces that are detected are blurred; audit PASS; output video finalised; performance report (A17/A11) archived. |

---

## Explicit priorities (Lead direction)

1. **Face detection and blur accuracy** (highest).  
2. Speed / throughput (important, secondary).  

SCRFD paths are **out of objective** after discard (A14).

---

## Evidence under this objective

| Artifact | Role |
| -------- | ---- |
| A06 / A07 | Locked ONNX/engine identity |
| A08 | Evaluation against these targets |
| A09 | Validation method for blur/privacy checks |
| A11 | Long live soak |
| A12 | Fallback / hold / no-face behaviour |
| A17 | Latency, FPS headroom, detection counts |

---

## Notes

- Pretrained weights only — no company fine-tune in this objective version.  
- Audit PASS means recorded blur operations succeeded when faces were present; it does **not** prove every possible face in every frame was detected (detector miss is separate).  
