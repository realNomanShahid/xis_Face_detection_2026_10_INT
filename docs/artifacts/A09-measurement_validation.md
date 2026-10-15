# A09 — Measurement Validation

**ID:** `Inter-69`-MVR-v1  
**Engineer:** Noman  
**Related:** A02, A08, A11, A12  

---

## Purpose

Validates that the **measurements and checks** the system claims (detection, blur applied, audit, FPS) are produced correctly and are usable for release decisions.

---

## What is “measured” in this system

| Measure | How obtained |
| ------- | ------------ |
| Face boxes / scores | RetinaFace decode + NMS in probe |
| Blur applied | CUDA kernels on ROIs; audit JSONL |
| Hold reuse | Counter when previous boxes used |
| FPS / latency | Pipeline timers + NVML |
| Frame loss | mux received vs processed |

---

## Validation activities performed

| Activity | Done? | Evidence |
| -------- | ----- | -------- |
| Live RTSP with known person in view | Yes | Preview / detections / detected_instances |
| Privacy audit PASS on long run | Yes | A11 |
| Saved video plays (MKV path) | Yes | Preferred container after MP4 issues |
| Side-by-side sample frames | Yes (optional flag) | `sample_frames/` when enabled |


---

## Ground-truth method

| Item | Value |
| Method used in practice | Visual confirmation on live preview and saved detection JPEGs; audit counters |

---

## Known measurement limits

- Detector **misses** (far / tiny / fisheye faces) are not counted as audit “failed”; audit only covers blur ops when detections exist.  
- Blur latency may round to 0.0 ms in reports for very fast kernels.  
- “Faces now” console line samples every N frames and can show 0 even when other frames had detections.

---

## Verdict

Instrumentation for FPS, loss, and privacy audit is **validated operationally** on the soak run.  
