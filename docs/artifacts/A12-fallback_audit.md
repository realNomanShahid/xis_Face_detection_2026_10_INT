# A12 — Fallback Audit

**ID:** `Inter-69`-FBA-001  
**Engineer:** Noman  
**Related:** A02, A10, A11  

---

## Purpose

Confirm failure and empty-detection paths do **not** invent unsafe output (e.g. must not claim blur success when blur failed).

---

## Paths examined

| Path | Behaviour | Safe? |
| ---- | --------- | ----- |
| No faces in frame | Audit `skipped_no_faces`; no blur kernels required | Yes |
| Brief detect miss | `--hold` reuses last boxes up to N frames; audit `applied_from_previous_frame` | Yes (by design) |
| Blur kernel exception | Logged as failed; run can error — must not silent-success | Yes if failed counted |
| RTSP disconnect | nvurisrcbin reconnect settings; may gap frames | Network-dependent |
| Unauthorized RTSP | No frames; not a false PASS audit | Yes |
| EOS / Ctrl+C | EOS finalises container | Preferred MKV path |

---

## Soak evidence (A11)

| Outcome | Count |
| ------- | ----: |
| Failed blur | 0 |
| Partial | 0 |
| Applied | 4,593 |
| From previous frame | 1,412 |
| Skipped no faces | 387,506 |

---

## Verdict

**PASS** for audit accounting on the 2026-10-15 soak.  
Hold behaviour is intentional fallback, not a hidden skip of blur when boxes are held.

**Lead review:** `---`
