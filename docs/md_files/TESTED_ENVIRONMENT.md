# Tested Environment

Exact versions this project was checked against. If something behaves differently on your machine, compare it with this page first.

- **Collected:** 2026-10-22
- **Machine:** native Ubuntu 22.04 (not WSL), single NVIDIA GPU
- **Scope:** RetinaFace-R50 only (see [MODEL.md](MODEL.md))

---

## 1. System

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04.5 LTS (x86_64) |
| Kernel | 6.8.0-138-generic |
| GPU | NVIDIA GeForce RTX 5080, 16 GB, compute capability 12.0 (`sm_120`) |
| NVIDIA driver | 595.99.02 (reports CUDA 13.2) |
| GStreamer | 1.20.3 |

## 2. CUDA, TensorRT, DeepStream

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA runtime used | **12.9** | `nvcc` 12.9.41. DeepStream and CuPy both report 12.9. |
| CUDA toolkits installed | 12.6 and 12.9 | `/usr/local/cuda` points to **12.6**; 12.9 is used only because `PATH` and `LD_LIBRARY_PATH` point to it (section 4). |
| CuPy NVRTC | 12.9 | Compiled and ran a test kernel on `sm_120`. |
| TensorRT used by `nvinfer` | `libnvinfer10` **10.16.1.11** (+cuda13.2) | Apt packages `libnvinfer10`, `libnvinfer-plugin10`, `libnvonnxparsers10`. The plugin links `libnvinfer.so.10`. |
| DeepStream SDK | **7.1.0** (`deepstream-7.1 7.1.0-1`) | All needed plugins load (`nvurisrcbin`, `nvstreammux`, `nvinfer`, `nvvideoconvert`, `nvv4l2h264enc`, `nvv4l2h265enc`). |
| cuDNN | not installed | Not needed by this project. |

Also installed but **not used** by DeepStream 7.1: `libnvinfer11` / `libnvonnxparsers11` 11.4.0.106 and the `-dev` packages (11.4). The plain `apt install libnvinfer-dev` now resolves to TensorRT 11.

> **RTX 50-series note.** DeepStream 7.1 ships with an older TensorRT (10.3) that does not support Blackwell GPUs. This setup works because a newer TensorRT 10.x is installed from the NVIDIA apt repository. NVIDIA's supported route for RTX 50-series is DeepStream 8.0, which has **not** been tested here.

> **Engine files are tied to the TensorRT version.** `retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine` was built with the TensorRT above. After changing TensorRT, CUDA or the driver, pass `--onnx` so it is rebuilt.

## 3. Python

| Package | Version |
|---------|---------|
| Python | 3.10.21 (conda env `ds_face`) |
| pyds | 1.2.0 (`pyds-1.2.0-cp310-cp310-linux_x86_64.whl`) |
| PyGObject | 3.54.5 |
| numpy | 2.2.6 |
| cupy-cuda12x | 14.2.0 |
| opencv-python-headless | 5.0.0.93 |
| nvidia-ml-py (`pynvml`) | 13.615.71 |
| nvidia-nvimgcodec-cu12 | 0.9.0.20 |
| PyYAML | 6.0.3 |
| requests | 2.34.2 |

`pip check` reported no conflicts. For the one-time ONNX export only (not needed to run): `torch`, `torchvision`, `onnx`, `onnxruntime`, see `requirements.txt`.

Install with `pip install -r requirements.txt`. Exact versions above are in `requirements.lock`.

## 4. Environment variables

Add these to `~/.bashrc` **once** (repeating them creates duplicates in `LD_LIBRARY_PATH`):

```bash
export PATH=/usr/local/cuda-12.9/bin:/opt/nvidia/deepstream/deepstream/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0
export GIO_MODULE_DIR=/usr/lib/x86_64-linux-gnu/gio/modules
```

The `cuda-12.9` entries are required. Without them the system falls back to CUDA 12.6, which cannot be relied on for `sm_120`.

## 5. Known behaviour

- **Import order matters.** `import gi` / `Gst` must come **before** `import pyds`. Importing `pyds` first fails with `libgobject-2.0.so.0: undefined symbol: g_pointer_bit_unlock_and_set` (conda glib vs system glib). `src/retinaface_face_detection.py` already imports in the safe order.
- **Harmless messages:** GStreamer warnings about `libtritonserver`, `librivermax`, `libucs`; nvImageCodec errors about `nvjpeg2k` / `nvtiff` (JPEG2000 / TIFF are not used here).
- **PyYAML is required** (not optional): without it the `configs/*.yaml` files are silently skipped and built-in defaults are used.


