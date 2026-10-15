# A04 — Annotation Schema

**ID:** `Inter-69`-SCHEMA-v1  
**Engineer:** Noman  
**Lead visibility:** Abdul Moiz  
**Related:** A03, A05  

---

## Purpose

In a training project this file locks label definitions.  
For this Face Blur project **no company annotation schema was defined**, because the detector is **pretrained RetinaFace-R50** with no fine-tune.

---

## Status

| Item | Value |
| ---- | ----- |
| Project annotation schema | **N/A** |
| Reason | Pretrained model; no label campaign for this task |
| SCRFD labels | **N/A** (path discarded) |

---

## Runtime “schema” of pipeline outputs (informational)

Not an annotation schema — describes **machine outputs**:

| Field / concept | Meaning |
| --------------- | ------- |
| Box | `x1,y1,x2,y2` in frame pixels after letterbox inverse |
| Score | Face confidence after NMS |
| ROI + blur radius | Box expanded by `--pad`; radius from `--blur-frac` |
| Hold | Reuse last boxes up to `--hold` frames when current detect is empty |
| Audit statuses | applied, applied_from_previous_frame, partial, skipped_no_faces, failed |

---

## Change log

| Date | Change | By |
| ---- | ------ | -- |
| 2026-10-07+ | No annotation schema required for pretrained path | Noman |

---

## Lead note

If a future fine-tune is approved, a new A04 must be issued **before** annotation starts.
