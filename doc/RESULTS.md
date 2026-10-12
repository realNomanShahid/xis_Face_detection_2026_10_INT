# Performance Results

Work Item: **INTER-69**  
Device: **NVIDIA GeForce RTX 5080**  
Runtime: **DeepStream + TensorRT (FP16)**

---

## Summary Comparison (Video File Input)

Input: 1920×1080 video file | 1,800 frames processed | 0 frames dropped

| Metric | SCRFD 2.5G | SCRFD 10G | RetinaFace-R50 |
|--------|------------|-----------|----------------|
| **Avg Pipeline FPS** | **398.5** | **381.4** | **346.1** |
| Frames Processed | 1800 | 1800 | 1800 |
| Frames Dropped | 0 | 0 | 0 |
| Headroom | 1.73× | 1.62× | 1.13× |
| Estimated Max FPS | 983.70 | 871.20 | 452.21 |
| **Total Detections** | 42,802 | 45,243 | **77,585** |
| Avg Faces / Frame | 23.78 | 25.14 | 43.10 |
| Max Faces / Frame | 35 | 37 | 55 |

**Highest FPS:** SCRFD 2.5G (398.5)  
**Most detections:** RetinaFace-R50 (77,585)

---

## Model-specific Settings

| Setting | SCRFD 2.5G | SCRFD 10G | RetinaFace-R50 |
|---------|------------|-----------|----------------|
| Detection Size | 640 × 640 | 640 × 640 | 1280 × 736 |
| Confidence | 0.35 | 0.35 | 0.40 |
| NMS | 0.4 | 0.4 | 0.4 |

---

## Latency per Frame (ms) — mean / max

| Stage | SCRFD 2.5G | SCRFD 10G | RetinaFace-R50 |
|-------|------------|-----------|----------------|
| Detect (nvinfer / TensorRT) | 0.42 / 3.09 | 0.68 / 1.76 | 1.82 / 6.14 |
| Post (decode + NMS) | 0.62 / 2.19 | 0.61 / 1.97 | 0.80 / 2.42 |
| Blur (CUDA kernels) | 0.15 / 0.60 | 0.37 / 0.84 | 1.36 / 5.81 |
| Encode (NVENC) | 1.02 / 2.43 | 1.15 / 2.04 | 2.21 / 4.68 |

---

## Resource Usage

| Metric | SCRFD 2.5G | SCRFD 10G | RetinaFace-R50 |
|--------|------------|-----------|----------------|
| CPU cores used (avg) | 1.77 | 1.75 | 1.56 |
| GPU util avg / peak | 18.3% / 23% | 30.0% / 37% | 74.6% / 77% |
| GPU memory peak | 4158 MB | 4216 MB | 4398 MB |

---

## High-Rate RTSP Test (SCRFD-2.5G only)

| Metric | Value |
|--------|-------|
| Avg Pipeline FPS | **408.50** |
| Source FPS | 428.60 |
| Frames Processed | 14332 |
| Frames Dropped | 6 |
| Headroom | 2.58× |
| Estimated Max FPS | 1052.89 |

### Latency (High-Rate RTSP)

| Stage | mean | max |
|-------|------|-----|
| Detect (nvinfer / TensorRT) | 0.58 | 2.53 |
| Post (decode + NMS) | 0.78 | 3.94 |
| Blur (CUDA kernels) | 0.17 | 2.06 |
| Encode (NVENC) | 0.77 | 5.42 |

### Resource Usage (High-Rate RTSP)

| Metric | Value |
|--------|-------|
| CPU cores used (avg) | 2.17 |
| GPU utilisation avg / peak | 15.03% / 21% |
| GPU memory peak | 4050 MB |

---

## Key Takeaways

1. **SCRFD 2.5G** is the fastest (398.5 FPS) and lightest on GPU (18.3%).
2. **SCRFD 10G** is close behind (381.4 FPS) with slightly more detections.
3. **RetinaFace-R50** detects the most faces (77,585) but is slower (346.1 FPS) and uses significantly more GPU (74.6%).
4. All three models far exceed real-time on a 60 FPS source.
5. Real live streams (cameras / Rutube) are limited by their own FPS (usually 25–30). The pipeline itself is much faster.

---

## Note on Live Streams

When using a normal live stream (Rutube, camera, etc.), you will see ~25–30 FPS.  
This is **normal** — the source only sends frames at that rate.  
To measure true pipeline speed, use a high-rate local RTSP server (see [HOW_TO_RUN.md](HOW_TO_RUN.md)).
