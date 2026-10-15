# A15 — Re-entry Record

**ID:** `Inter-69`-RE-1  
**Engineer:** Noman  
**Related:** A10, A14  

---

## Purpose

Handbook re-entry after root-cause work or rejected evaluation.

---

## Re-entries logged

### RE-1 — After model family decision (SCRFD → RetinaFace)

| Field | Content |
| ----- | ------- |
| Trigger | Lead discard of SCRFD path |
| Re-entry point | Integration continues on RetinaFace-R50 only |
| Cause sentence | Accuracy priority not met by SCRFD on observed streams |
| Follow-on artifacts | A06/A07 engine lock; A08/A11 on RetinaFace |

### RE-2 — After output container / playback issues

| Field | Content |
| ----- | ------- |
| Trigger | Shareable MP4 playback failures |
| Re-entry point | Default MKV; muxer selection by extension |
| Cause sentence | Live MP4 fragmentation and player compatibility |


---

## If no further re-entry

Additional formal re-entries: **`---`**.

---

## Status

Template complete; fill dates when calendar is confirmed.
