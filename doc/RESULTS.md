# Performance Results

Work Item: **INTER-69**  
Model: **SCRFD-2.5G**  
Device: **NVIDIA GeForce RTX 5080**

---

## High-Rate RTSP Test

| Metric | Value |
|--------|-------|
| Avg Pipeline FPS | **408.50** |
| Source FPS | 428.60 |
| Frames Processed | 14332 |
| Frames Dropped | 6 |
| Headroom | 2.58× |
| Estimated Max FPS | 1052.89 |

---

## Latency per Frame (ms)

| Stage | mean | max |
|-------|------|-----|
| Detect (nvinfer / TensorRT) | 0.58 | 2.53 |
| Post (decode + NMS) | 0.78 | 3.94 |
| Blur (CUDA kernels) | 0.17 | 2.06 |
| Encode (NVENC) | 0.77 | 5.42 |

---

## Resource Usage

| Metric | Value |
|--------|-------|
| CPU cores used (avg) | 2.17 |
| GPU utilisation avg / peak | 15.03% / 21% |
| GPU memory peak | 4050 MB |

---

## Key Takeaways

1. Pipeline easily keeps up with high-rate sources.
2. Bottleneck is not the model or blur — GPU still has headroom.
3. Real live streams (cameras / Rutube) are limited by their own FPS (usually 25–30). The pipeline itself is much faster.

---

## Note on Live Streams

When using a normal live stream (Rutube, camera, etc.), you will see ~25–30 FPS.  
This is **normal** — the source only sends frames at that rate.  
To measure true pipeline speed, use a high-rate local RTSP server (see [HOW_TO_RUN.md](HOW_TO_RUN.md)).
