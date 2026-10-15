# A14 — Decision Record

**Engineer:** Noman  
**Lead:** Abdul Moiz  
**Related:** A01, A02, A06, A10  

---

## DEC-001 — Discard SCRFD; adopt RetinaFace-R50

| Field | Content |
| ----- | ------- |
| Decision | SCRFD-2.5G and SCRFD-10G are **not** the production detector. **RetinaFace ResNet-50** is the only detector path in scope. |
| Decider | Abdul Moiz (Lead), implemented by Noman |
| Date | `[TO FILL — on or after 2026-10-07 research]` |
| Context | Engineer evaluated SCRFD 2.5G and 10G (strong speed) and RetinaFace-R50 on live office camera, Rutube, file, and local RTSP. Accuracy of SCRFD was not accepted relative to RetinaFace. Project priority is **face detection and blur accuracy**, then speed. |
| Alternatives rejected | SCRFD-2.5G production; SCRFD-10G production |
| Why | Accuracy priority; Lead direction to discard SCRFD |
| Consequences | All A06+ artifacts lock RetinaFace engine only; SCRFD retained only as historical research note |

---

## DEC-002 — Pretrained weights only (no fine-tune in this phase)

| Field | Content |
| ----- | ------- |
| Decision | Use official/public pretrained RetinaFace-R50 → ONNX → TensorRT. No company training run.|
| Decider | Noman |
| Date | 2026-10-12 |
| Consequences | A03–A05 largely N/A; A07 is engine-build record |

---

## DEC-003 — Output container preference MKV for shareable saves

| Field | Content |
| ----- | ------- |
| Decision | Prefer **matroskamux / .mkv** for saved live outputs intended for sharing |
| Decider | Engineer (from playback issues) |
| Date | `[TO FILL]` |
| Why | Fragmented / progressive MP4 issues with players other than VLC; corruption reports |
| Alternatives | MP4 + ffmpeg faststart/re-encode |
| Consequences | Default `--output` .mkv in pipeline |

---

## DEC-004 — Operating point for office live soak

| Field | Content |
| ----- | ------- |
| Decision | conf **0.15**, NMS **0.4**, source ~1536×864, RTSP TCP latency 500 ms for documented soak |
| Decider | Engineer |
| Date | 2026-10-15 (soak) |
| Why | Improve recall on difficult wide camera vs higher conf |
| Note | Higher conf may be used for cleaner false-positive trade-off in other scenes |

---

## Status

DEC-001 is the governing product decision for model family.  
Dates marked `----` should be completed by engineer/Lead when exact calendar dates are confirmed.
