# DeepStream Face Blur – Complete Installation & Run Guide

**RetinaFace-R50 only** (SCRFD paths discarded)

**Simple step-by-step guide** for Ubuntu 22.04 + NVIDIA GPU  
(Tested with RTX 5080, Driver 595.99, DeepStream 7.1)

SCRFD-2.5G / SCRFD-10G were evaluated for speed but **not accepted for accuracy**.  
This guide covers the **supported** stack only: DeepStream + TensorRT + **RetinaFace ResNet-50**.

Project layout (reference):

```text
face/
  configs/          # YAML defaults
  src/
    config_loader.py
    pytorch_to_onnx_retinaface.py
    retinaface_face_detection.py
  docs/md_files/
  results/RetinaFace_resnet50/
```

How to run after install → [HOW_TO_RUN.md](HOW_TO_RUN.md)

For exact version installation read this → [TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT) 
---

## 1. System Requirements

| Item | Required |
|------|----------|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA GPU (RTX series preferred) |
| NVIDIA Driver | Working (`nvidia-smi` must work) |
| CUDA (from driver) | 12.x or higher |
| Python | **3.10** (required for official `pyds` wheel) |
| Model | RetinaFace-R50 ONNX → TensorRT engine |

---

## 2. Fix NVIDIA Driver (if `nvidia-smi` fails)

If you see **"Driver/library version mismatch"**:

```bash
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

You must see a clean table (GPU name + driver version).  
**Do not continue until `nvidia-smi` works.**

On company machines, prefer reboot over forced `rmmod` when modules are in use.

---

## 3. Install System Dependencies

```bash
sudo apt update

sudo apt install -y \
  libssl3 libssl-dev \
  libgles2-mesa-dev \
  libgstreamer1.0-0 gstreamer1.0-tools \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libgstrtspserver-1.0-0 \
  libjansson4 libyaml-cpp-dev libjsoncpp-dev \
  protobuf-compiler gcc make git python3 \
  python3-gi python3-dev python3-gst-1.0 python-gi-dev \
  libgirepository1.0-dev libcairo2-dev \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  gstreamer1.0-x \
  build-essential cmake
```

---

## 4. Install CUDA Toolkit 12.6 (toolkit only, no new driver)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
```

---

## 5. Install TensorRT packages (needed by DeepStream)

```bash
sudo apt install -y \
  libnvinfer10 libnvinfer-dev \
  libnvinfer-plugin10 libnvinfer-plugin-dev \
  libnvonnxparsers10 libnvonnxparsers-dev \
  libnvinfer-headers-dev libnvinfer-headers-plugin-dev
```

Package names can vary slightly by Ubuntu mirror; if a name is missing, search with:

```bash
apt-cache search nvinfer | head
```

---

## 6. Download & Install DeepStream 7.1

### Official download

**https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=7.1**

Download:

```text
deepstream-7.1_7.1.0-1_amd64.deb
```

(Free NVIDIA developer account required.)

### Install

```bash
cd ~/Downloads
sudo dpkg -i deepstream-7.1_7.1.0-1_amd64.deb
sudo apt --fix-broken install -y
```

If `dpkg` reports missing CUDA/GStreamer deps, `apt --fix-broken install -y` should pull them. Re-run the `.deb` install if needed.

### Environment variables (permanent)

```bash
echo 'export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=/opt/nvidia/deepstream/deepstream/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Also useful for Python GI / TLS on some machines:

```bash
echo 'export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0' >> ~/.bashrc
echo 'export GIO_MODULE_DIR=/usr/lib/x86_64-linux-gnu/gio/modules' >> ~/.bashrc
source ~/.bashrc
```

### Verify

```bash
deepstream-app --version
```

Expected:

```text
deepstream-app version 7.1.0
DeepStreamSDK 7.1.0
```

---

## 7. Create Python Environment (Python 3.10)

The official `pyds` wheel targets **cp310**.

```bash
conda create -n ds_face python=3.10 -y
conda activate ds_face
```

---

## 8. Install Python packages

### `requirements.txt` (minimum for this project)

```txt
numpy
cupy-cuda12x
opencv-python-headless
nvidia-ml-py
requests
PyYAML
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Use `cupy-cuda11x` only if your CUDA major version is 11.

---

## 9. Install DeepStream Python bindings (`pyds`)

```bash
wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl
pip install pyds-1.2.0-cp310-cp310-linux_x86_64.whl
```

### Test

```bash
python -c "import pyds; print('pyds OK')"
python -c "import cupy; print('cupy OK')"
python -c "from gi.repository import Gst; print('Gst OK')"
```

All should succeed inside the `ds_face` env.

---

## 10. Prepare RetinaFace model files

| File | Role |
|------|------|
| Pretrained PyTorch weights | Official RetinaFace-R50 (see [MODEL.md](MODEL.md)) |
| ONNX | e.g. `retinaface_r50_1280x736.onnx` |
| TensorRT engine | e.g. `retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine` |

### Convert PyTorch → ONNX (once)

```bash
python3 src/pytorch_to_onnx_retinaface.py
```

### Build engine (once)

First DeepStream run with `--onnx` builds a compatible engine if the old `.engine` / `.trt` does not match your TensorRT version:

```bash
python3 src/retinaface_face_detection.py \
  --source video.mp4 \
  --onnx retinaface_r50_1280x736.onnx \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine
```

Set the same paths in `configs/detection.yaml` so later runs need fewer flags.

**Do not** use SCRFD ONNX/engines for this project path.

---

## 11. Run the Face Blur Script (RetinaFace)

```bash
conda activate ds_face

export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
export PATH=/opt/nvidia/deepstream/deepstream/bin:$PATH
export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0
```

### Video file

```bash
python3 src/retinaface_face_detection.py \
  --source video.mp4 \
  --onnx retinaface_r50_1280x736.onnx \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --output output_blurred.mkv \
  --conf 0.15
```

### Live RTSP (example)

```bash
python3 src/retinaface_face_detection.py \
  --source "rtsp://user:ENCODED_PASS@host:554/path" \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --onnx retinaface_r50_1280x736.onnx \
  --width 1536 --height 864 \
  --rtsp-protocol tcp --latency 500 \
  --preview \
  --output camera_blurred.mkv
```

Prefer **`.mkv`** for shareable live recordings.

Defaults (conf, paths, out_dir, etc.) can come from `configs/*.yaml` — see [HOW_TO_RUN.md](HOW_TO_RUN.md) and [README_CONFIG.md](../../README_CONFIG.md).

---

## 12. Useful optional flags

| Flag | Meaning | Example |
|------|---------|---------|
| `--max-seconds` | Stop after N seconds | `--max-seconds 60` |
| `--width` / `--height` | nvstreammux size | `--width 1536 --height 864` |
| `--conf` | Detection threshold | `--conf 0.15` |
| `--preview` | Live window | `--preview` |
| `--out-fps` | Steady timestamps for saved video | `--out-fps 12` |
| `--sample-every` | Periodic samples | `--sample-every 100` |
| `--no-save-detections` | Skip `detected_instances/` | |
| `--rtsp-protocol tcp` | Reliable RTSP | |
| `--latency` | RTSP jitter buffer (ms) | `--latency 500` |

---

## 13. Output files

After a successful run (often under `logs/` when `out_dir: logs`):

| Output | Description |
|--------|-------------|
| `*.mkv` / `*.mp4` | Blurred video |
| `metrics.json` | Performance + privacy audit summary |
| `detection_logs.jsonl` | Per-frame detections |
| `privacy_audit.jsonl` | Per-frame blur audit |
| `detected_instances/` | Frames with detection + blur (default on) |
| `sample_frames/` | Optional before/after pairs |
| `*.engine` | TensorRT engine built/reused by nvinfer |

Archived example results: `results/RetinaFace_resnet50/`.

---

## 14. Common problems & solutions

| Problem | Solution |
|---------|----------|
| `nvidia-smi` driver/library mismatch | Reboot |
| `No module named 'pyds'` | Python **3.10** + official wheel |
| `Namespace Gst not available` | Install `gir1.2-gstreamer-1.0` and set `GI_TYPELIB_PATH` |
| Engine serialize version mismatch | Pass `--onnx` so DeepStream rebuilds the engine |
| `Could not create 'nvurisrcbin'` | Check `GST_PLUGIN_PATH` and `LD_LIBRARY_PATH` |
| `cupy` CUDA error | Match `cupy-cuda12x` to CUDA 12 |
| RTSP `401 Unauthorized` | URL-encode password; use correct ONVIF/RTSP user |
| MP4 only plays in VLC / artifacts | Save as **`.mkv`** or re-encode with ffmpeg |
| GStreamer warnings (`libtritonserver`, `librivermax`) | Harmless if you do not use those plugins |

---

## 15. Quick checklist

- [ ] `nvidia-smi` works  
- [ ] `deepstream-app --version` shows 7.1.0  
- [ ] Conda env with **Python 3.10**  
- [ ] `import pyds`, `import cupy`, `Gst` OK  
- [ ] RetinaFace ONNX available  
- [ ] Engine built once for this machine  
- [ ] `configs/detection.yaml` paths set  
- [ ] Environment variables set  

---

## 16. Download links summary

| Item | Link |
|------|------|
| DeepStream 7.1 `.deb` | https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=7.1 |
| `pyds` wheel (Python 3.10) | https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl |
| CUDA keyring | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb |

---

## 17. Out of scope

| Item | Status |
|------|--------|
| SCRFD-2.5G / SCRFD-10G install or run | **Not supported** (discarded) |
| Jetson | Not supported (`get_nvds_buf_surface_gpu` is x86-only in this design) |
| Fine-tuning RetinaFace on company data | Not part of this install guide |

Historical SCRFD PDFs under `docs/` are archive only.

---

**Done.**  
After install, follow [HOW_TO_RUN.md](HOW_TO_RUN.md) for file / RTSP / Rutube / preview runs.
