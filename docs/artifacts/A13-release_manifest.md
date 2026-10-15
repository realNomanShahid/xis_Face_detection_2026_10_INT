# A13 — Release Manifest

**ID:** `Inter-69`-REL-v1  
**Lead:** Abdul Moiz  
**Engineer:** Noman  
**Related:** A02, A06, A08, A11, A12  

---

## Purpose

Lists what is included in a release candidate and what is explicitly excluded.

---

## Release candidate contents (intended)

| Component | Identity |
| --------- | -------- |
| Pipeline script | Face blur DeepStream RetinaFace app (project `retinaface_face_detection.py` / equivalent) |
| ONNX | `retinaface_r50_1280x736.onnx` |
| Engine | `retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine` |
| Stack | DeepStream 7.x, pyds, CUDA blur, NVENC |
| Default output | MKV recommended for shareable files |
| Docs | A01–A17 pack; install notes as applicable |

---

## Explicitly excluded

| Item | Reason |
| ---- | ------ |
| SCRFD-2.5G / SCRFD-10G engines and scripts as production path | Discarded for accuracy (A14) |
| Fine-tuned company face weights | None produced |
| Guaranteed detection of arbitrarily small / distant faces | Limitation documented |

---

## Evidence attached to this manifest

| Artifact | Status |
| -------- | ------ |
| A06 locked engine | Required |
| A11 soak | Completed 2026-10-15 |
| A12 fallback | Recorded |
| A08 evaluation | Draft / conditional |
| Lead approval on main | `[TO FILL: commit / tag / date]` |

---

## Approval

| Role | Name | Date | Sign |
| ---- | ---- | ---- | ---- |
| Engineer | Noman | `Noman` | |
| Lead | Abdul Moiz | `----` | |

---

## Status

**DRAFT** until Lead fills approval and task ID.
