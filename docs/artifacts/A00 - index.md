# Face Blur Pipeline — Artifact Index

**Engineer:** Noman  
**Lead:** Abdul Moiz  
**Project start:** 2026-10-07  
**Model in scope:** RetinaFace ResNet-50 (pretrained → ONNX → TensorRT) only  
**Discarded (not in scope):** SCRFD-2.5G, SCRFD-10G  

**Task / project ID:** `Inter-69`  

---

## Purpose of this folder

This folder holds the A01–A17 handbook artifacts for the **live face detection and GPU blur** pipeline (DeepStream + TensorRT + CUDA blur).  
Use this index for timeline, ownership, and which file answers which review question.

SCRFD variants were evaluated early and **discarded by Lead direction** in favour of RetinaFace-R50 (accuracy priority). Artifacts below describe the **RetinaFace path only**.

---

## People

| Role | Name |
| ---- | ---- |
| Engineer | Noman |
| AI Lead | Abdul Moiz |
| Annotation QA reviewer | N/A (pretrained model; no project annotation set) |

---

## Timeline (known)

| Date | Activity |
| ---- | -------- |
| 2026-10-07 | Project start; model research (SCRFD 2.5G / 10G vs RetinaFace-R50) |
| 2026-10-07 → mid Oct | SCRFD trials on video / Rutube / RTSP; accuracy judged insufficient vs RetinaFace |
| 2026-10-12 | Lead decision: discard SCRFD; adopt RetinaFace-R50 only |
| 2026-10-12 | Official RetinaFace PyTorch weights → ONNX → TensorRT FP16 engine |
| 2026-10-9 | DeepStream pipeline integration (nvurisrcbin, nvinfer, CUDA ellipse blur, NVENC) |
| 2026-10-9 | Live tests: office RTSP camera, Rutube HLS, local file, local high-FPS RTSP |
| 2026-10-15 | A11 soak: live camera ~3 h 38 min (see A11) |

---

## Artifact map (A01–A17)

| ID | File | What it records |
| -- | ---- | --------------- |
| A01 | A01-question_set.md | Problem framing; why face blur |
| A02 | A02-objective.md | Success criteria (accuracy first, then speed) |
| A03 | A03-dataset_inventory.md | Weights / streams / test sources (no train set) |
| A04 | A04-annotation_schema.md | N/A for pretrained detector; blur policy notes |
| A05 | A05-annotation_qa.md | N/A — no project annotation batch |
| A06 | A06-dataset_version.md | Locked engine / ONNX identity |
| A07 | A07-training_run.md | Not a train run — engine build from pretrained ONNX |
| A08 | A08-evaluation.md | Evaluation vs A02 (live + file sources) |
| A09 | A09-measurement_validation.md | How blur/privacy checks were validated |
| A10 | A10-root-cause.md | Issues found (e.g. tiny faces, MP4 playback) |
| A11 | A11-soak_test.md | Live soak 2026-10-15 (~3h38m) |
| A12 | A12-fallback_audit.md | Hold / no-face / reconnect behaviour |
| A13 | A13-release_manifest.md | What is eligible for release |
| A14 | A14-decision_record.md | SCRFD discard; RetinaFace adopt; conf/output choices |
| A15 | A15-re-entry_record.md | Re-entry after fixes (if any) |
| A16 | A16-exception_record.md | Handbook exceptions (if any) |
| A17 | A17-model_performance_report.md | Latency, FPS headroom, detection stats |

---

## Reading order for reviewers

1. A01 → A02 (problem and target)  
2. A14 (why SCRFD discarded, why RetinaFace)  
3. A06 → A07 (what binary is locked)  
4. A08 → A09 → A11 → A12 (evidence)  
5. A13 (release)  
6. A10 / A15 / A16 only if issues or exceptions apply  

---

## Status of this pack

| Item | Status |
| ---- | ------ |
| A11 soak numbers | Filled from supplied performance report (2026-10-15) |
| Task ID / formal names | `Inter-69` |

