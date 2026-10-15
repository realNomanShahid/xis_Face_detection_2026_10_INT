# A08 — Evaluation

**ID:** `Inter-69`-EVAL-v1  
**Engineer:** Noman  
**Related:** A02, A06, A09, A11, A17  
**Parent objective:** A02  

---

## Purpose

Evaluate the RetinaFace-R50 DeepStream blur pipeline against A02 targets (accuracy priority, then speed).

---

## Scope evaluated

| Source class | Evaluated? | Notes |
| ------------ | ---------- | ----- |
| Office RTSP camera | Yes | Live; soak A11 |
| Rutube HLS | Yes | Integration / short runs |
| Local video file | Yes | High throughput possible |
| Local high-FPS RTSP | Yes | Benchmark only |
| SCRFD engines | Historical only | Discarded; not scored as release candidate |

---

## Method

- DeepStream pipeline with locked engine (A06).  
- Metrics from pipeline report (`metrics.json`) and privacy_audit.  
- Qualitative checks on preview / saved video / `detected_instances` images.  


---

## Results against A02 (known)

| Target theme | Evidence | Result |
| ------------ | -------- | ------ |
| Blur when faces detected | Privacy audit on soak: failed = 0, partial = 0 | PASS on recorded audit |
| Sustain live FPS | A11: 30.0 FPS, 0 frames lost, keeping up YES | PASS for that source rate |
| Accuracy priority | SCRFD discarded for weaker accuracy; RetinaFace retained | Process decision A14 |
| Tiny / distant faces | Office wide-angle: many frames 0 detect; sparse detections when person far | Limitation — not a pipeline crash |

---

## Quantitative snapshot (live soak reference — full detail A11)

| Metric | Value |
| ------ | ----- |
| Frames | 393,511 |
| Whole-run FPS | 30.0 |
| Total face detections | 4,900 |
| Frames with 0 detections | 98.8% |
| Privacy audit | PASS |


---

## Verdict (engineer)

**Conditionally PASS** for live stability and audit on the 2026-10-15 soak source.  
Accuracy claims beyond “RetinaFace preferred over SCRFD on observed streams” need labelled eval if Lead requires numeric mAP.

**Lead sign-off:** `---`
