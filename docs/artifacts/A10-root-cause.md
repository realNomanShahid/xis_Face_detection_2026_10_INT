# A10 — Root-Cause Note

**ID:** `Inter-69`-RCA-001  
**Engineer:** Noman  
**Related:** A08, A12, A14, A15  

---

## Purpose

Records disagreements or defects, root cause, and fix status.

---

## RCA entries

### RCA-1 — SCRFD accuracy insufficient for Lead acceptance

| Field | Content |
| ----- | ------- |
| Symptom | SCRFD 2.5G and 10G acceptable speed; accuracy weaker than RetinaFace on observed live/office/Rutube/file tests |
| Shape | Model selection / quality |
| Stages checked | Live stream, file, RTSP |
| Root cause | Detector family choice; project priority is detection/blur **accuracy** |
| Fix | Discard SCRFD production path; lock RetinaFace-R50 (A14, A06) |
| Re-measured | RetinaFace used for subsequent integration and soak |
| Status | Closed |

### RCA-2 — Saved MP4 poor compatibility / artifacts

| Field | Content |
| ----- | ------- |
| Symptom | Output played poorly outside VLC; occasional blocky frames; fragmented MP4 behaviour |
| Shape | Muxer / encoder / live EOS |
| Stages checked | qtmux + live fragment settings; player tests |
| Root cause | Live fragmented MP4 and hard EOS/finalisation behaviour not ideal for shareable files |
| Fix | Prefer **MKV (matroskamux)**; avoid fragment-duration for shareable files; optional ffmpeg remux/re-encode |
| Status | Mitigated in pipeline defaults |

### RCA-3 — Sparse detections on office wide camera

| Field | Content |
| ----- | ------- |
| Symptom | Long stretches of `faces now: 0`; low avg faces/frame on soak |
| Shape | Content / geometry / threshold |
| Stages checked | conf 0.15, 1536×864, fisheye-style view, distant subjects |
| Root cause | Face pixel size often below reliable detectability; not a TRT crash |
| Fix options | Closer framing, higher-res profile, lower conf (trade false positives), future model/ROI — **product decision** |
| Status | Accepted limitation unless Lead requires GT campaign |

### RCA-4 — RTSP 401 Unauthorized during first GPU-machine tests

| Field | Content |
| ----- | ------- |
| Symptom | GStreamer/ffplay 401 with wrong password encoding |
| Root cause | Special characters in password; ONVIF password required in URL (`Pass123#` → `Pass123%23`) |
| Fix | Document encoded RTSP URL; use ONVIF credentials that OpenCV path accepted |
| Status | Closed |

---

## Status

Known issues documented; no open crash-class defect on 2026-10-15 soak summary.
