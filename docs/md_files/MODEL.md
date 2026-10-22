# Model

This project uses **one face detection model: RetinaFace-R50 (ResNet-50)**. It runs as a TensorRT FP16 engine through DeepStream `nvinfer`, and the boxes are decoded and filtered on the GPU before the blur stage.

SCRFD-2.5G and SCRFD-10G were evaluated earlier and are **not supported** (see [section 6](#6-evaluated-and-not-used)).

---

## 1. RetinaFace-R50

| Item | Value |
|------|-------|
| Model | RetinaFace-R50 (ResNet-50 backbone) |
| Task | Face detection |
| Weights | Official pretrained `Resnet50_Final.pth` (no training or fine-tuning in this project) |
| Input size | 1280 × 736 (must be a multiple of 32) |
| Precision | FP16 (TensorRT) |
| Batch size | 1 (GPU 0) |
| Backend | TensorRT via DeepStream `nvinfer` |
| ONNX file | `retinaface_r50_1280x736.onnx` |
| Engine file | `retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine` |

**Pipeline:** `.pth` weights → ONNX (once) → TensorRT engine (built by `nvinfer` on the first run) → GPU decode + NMS → CUDA blur.

**Pretrained weights (.pth):**
[RetinaFace-R50 .pth](https://drive.google.com/uc?export=download&id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW). Save the file as `Resnet50_Final.pth`.

---

## 2. Convert the weights to ONNX (once)

The converter needs `--weights`, and its default size is 640 × 640, so pass the size explicitly:

```bash
pip install -r requirements-export.txt
python3 src/pytorch_to_onnx_retinaface.py \
  --weights Resnet50_Final.pth \
  --width 1280 --height 736
# writes retinaface_r50_1280x736.onnx
```

- The script clones `biubug6/Pytorch_Retinaface` into `./Pytorch_Retinaface` (needs `git` and internet).
- CPU PyTorch is enough. This only runs once.
- If `onnxruntime` is installed, the script compares the PyTorch and ONNX outputs as a self-check.

**ONNX outputs** (two tensors, landmarks are not exported):

| Output | Shape | Meaning |
|--------|-------|---------|
| `loc` | N × 4 | Box regression (dcx, dcy, dw, dh) |
| `conf` | N × 2 | Scores: [background, face] |

For 1280 × 736 there are N = 38,640 anchors.

---

## 3. Post-processing (on the GPU)

The `nvinfer` output tensors are decoded with CuPy kernels using fixed-size arrays, then filtered by score and NMS. There is one small host readback per frame.

| Setting | Value | Where |
|---------|-------|-------|
| Strides | 8, 16, 32 | code |
| Anchor min sizes | (16, 32), (64, 128), (256, 512) | code |
| Variances | (0.1, 0.2) | code |
| Score threshold (`conf`) | 0.15 (default) | `configs/detection.yaml`, or `--conf` |
| NMS IoU | 0.4 | `configs/detection.yaml`, or `--nms` |
| Max candidates before NMS | 1000 | `configs/detection.yaml` |

Lower `conf` gives more detections and more false positives. Raise it for cleaner scenes.

---

## 4. Rebuilding the engine

A TensorRT engine only works with the TensorRT version, CUDA version and GPU it was built on. After changing TensorRT, CUDA or the driver, delete the old engine and pass the ONNX so `nvinfer` rebuilds it:

```bash
rm -f retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine
python3 src/retinaface_face_detection.py \
  --source video.mp4 \
  --onnx retinaface_r50_1280x736.onnx \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine
```

The first run takes longer while the engine is built. Later runs load the saved engine. The versions this project was tested with are in [TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md).

---

## 5. Performance (RTX 5080)

Video file input, 1920 × 1080, 1,800 frames, 0 frames dropped. Numbers from [RESULTS.md](RESULTS.md); that run used `conf` 0.40, NMS 0.4.

| Metric | Value |
|--------|-------|
| Average pipeline FPS | 346.1 |
| Headroom | 1.13× |
| Estimated max FPS | 452.21 |
| Total detections | 77,585 |
| Faces per frame (avg / max) | 43.10 / 55 |
| GPU utilisation (avg / peak) | 74.6% / 77% |
| GPU memory (peak) | 4,398 MB |
| CPU cores used (avg) | 1.56 |

| Stage | Latency per frame, mean / max (ms) |
|-------|-------------------------------------|
| Detect (`nvinfer` / TensorRT) | 1.82 / 6.14 |
| Post (decode + NMS) | 0.80 / 2.42 |
| Blur (CUDA kernels) | 1.36 / 5.81 |
| Encode (NVENC) | 2.21 / 4.68 |

---

## 6. Evaluated and not used

SCRFD-2.5G and SCRFD-10G (640 × 640) were tested on the same input before RetinaFace-R50 was selected. The SCRFD code path was removed from this repository, so the earlier commands for it no longer work.

| Model | Input | Avg FPS | Total detections | GPU util (avg) |
|-------|-------|---------|------------------|----------------|
| SCRFD-2.5G | 640 × 640 | 398.5 | 42,802 | 18.3% |
| SCRFD-10G | 640 × 640 | 381.4 | 45,243 | 30.0% |
| **RetinaFace-R50** | 1280 × 736 | 346.1 | **77,585** | 74.6% |

**Why RetinaFace-R50:** it finds the most faces (about 1.8× the detections of SCRFD-2.5G on the same video), and its speed is still well above real time. The SCRFD models are faster and use less GPU, but missed many more faces.