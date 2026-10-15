# A11 — Soak Test

**Date:** 2026-10-15

---

**Engineer:** Noman

## Purpose

This artifact records a sustained live run of the Face Detection and Blur pipeline. It documents runtime, frame throughput, resource utilization, detection statistics, output generation, and privacy-audit results.

---

## Run identity

| **Field** | **Value** |
|---|---|
| Start time | 3:00 PM |
| Final timestamp | 6:38 PM |
| Elapsed | 13,112.4 seconds (3 hours, 38 minutes, 32.4 seconds) |
| Engineer | Noman |
| Mode | Live camera stream with DeepStream, TensorRT, and GPU blur |
| Model | RetinaFace-R50 |
| Input size | 1280 × 736 |
| Source resolution | 1536 × 864 |

---

## Environment

- **Pipeline:** DeepStream → TensorRT inference → post-processing/NMS → GPU blur (CUDA) → NVENC encoding
- **Hardware:** NVIDIA GeForce RTX 5080
- **Reported GPU memory total:** 16,303 MB
- **Source:** Live stream delivering approximately 30 FPS
- **Detection confidence:** 0.15
- **NMS threshold:** 0.4
- **Video timestamps:** Original timestamps preserved
- **Monitoring:** FPS, frame counts, latency, CPU/GPU utilization, GPU memory, and privacy-audit metrics

---

## Duration and performance

| **Requirement / Metric** | **Result** |
|---|---|
| Reported runtime | 3 hours, 38 minutes, 32.4 seconds |
| Frames processed | 393,511 |
| Steady-state FPS | 30.0 |
| Whole-run FPS | 30.0 |
| Keeping up with source | **YES** |
| Frames lost | 0 |
| Seconds with no delivered frames | 0 |
| Longest gap between frames | 175 ms |
| Estimated maximum FPS | 520.0 |
| Pipeline bottleneck | Detection (`nvinfer`) |
| End-of-stream handling | EOS handled successfully for source 0 |

**Performance observation:** The pipeline maintained the approximately 30 FPS source rate. The report indicates that the pipeline was not the throughput limit; the observed FPS was primarily determined by the source delivery rate.

---

## Resource utilization

| **Metric** | **Result** |
|---|---:|
| Average CPU cores used | 0.24 |
| Average GPU utilization | 6% |
| Peak GPU utilization | 13% |
| Average NVENC utilization | 2% |
| Average NVDEC utilization | 2% |
| Peak whole-GPU memory usage | 4,889 MB |

---

## Latency per frame

| **Processing stage** | **Mean (ms)** | **p50 (ms)** | **p95 (ms)** | **Max (ms)** |
|---|---:|---:|---:|---:|
| Detection (`nvinfer`) | 1.9 | 1.9 | 2.1 | 7.7 |
| Post-processing: map + decode + NMS | 0.2 | 0.2 | 0.3 | 2.4 |
| GPU blur kernels | 0.0 | 0.0 | 0.0 | 3.3 |
| Encoding (NVENC) | 0.8 | 0.8 | 1.1 | 5.5 |
| End-to-end mux → file | 6.6 | 5.1 | 9.7 | 142.5 |
| Frame interval | 33.3 | 33.1 | 39.1 | 175.0 |

*Note: The blur latency of 0.0 ms is the reported rounded value, not necessarily literal zero processing time.*

---

## Detection statistics

| **Metric** | **Result** |
|---|---:|
| Frames processed | 393,511 |
| Frames with zero detections | 388,918 (98.8%) |
| Total face detections | 4,900 |
| Average faces per frame | 0.01 |
| Maximum faces per frame | 3 |
| Frames handled by the hold mechanism | 1,412 |
| Detected-frame images saved | 4,593 |

---

## Privacy audit

| **Audit metric** | **Count / Result** |
|---|---:|
| Blur applied | 4,593 |
| Blur applied from previous frame | 1,412 |
| Partial outcomes | 0 |
| Skipped because no faces were detected | 387,506 |
| Failed | 0 |
| Privacy audit result | **PASS** |

The privacy audit reported no partial or failed blur applications. The hold mechanism reused previous-frame detections in 1,412 cases.

The audit PASS applies to the recorded run and does not independently guarantee that every face was detected in every frame.

---

## Stability and continuity

- **Frame loss:** 0 frames reported lost.
- **Stream continuity:** The pipeline maintained the source-delivered rate of approximately 30 FPS.
- **Output finalization:** EOS was sent, and `nvstreammux` reported successful EOS handling for source 0.
- **GPU memory peak:** 4,889 MB.
- **Crash / out-of-memory stop:** Not confirmed by the supplied performance summary; verify the complete process log before recording these as absent.

---

## What this soak test demonstrates

- Sustained live processing over a reported duration of approximately 3 hours and 38 minutes.
- Processing of 393,511 frames at a whole-run rate of 30 FPS.
- Zero reported frame loss while keeping up with the live source.
- Recorded CPU/GPU utilization, memory usage, and per-stage latency.
- Successful finalization of the output video.
- A **PASS** result from the recorded privacy audit, with zero partial and zero failed outcomes.

---

## Verdict

**PASS** for reported throughput and privacy-audit results.

The pipeline processed 393,511 frames at the source-delivered rate of approximately 30 FPS, with no reported frame loss. The privacy audit returned PASS.


---

## Status

**RUN COMPLETED — results recorded from the supplied performance report.**