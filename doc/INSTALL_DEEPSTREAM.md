# Install DeepStream + Dependencies

Simple step-by-step guide.

---

## 1. System Requirements

- Ubuntu 22.04 (recommended)
- NVIDIA GPU (x86 only – **Jetson is not supported**)
- NVIDIA driver installed
- CUDA 12.x (or 11.x)

Check GPU:

```bash
nvidia-smi
```

---

## 2. Install DeepStream

### Option A – Official NVIDIA package (recommended)

1. Go to NVIDIA DeepStream download page:  
   https://developer.nvidia.com/deepstream-sdk

2. Download **DeepStream 7.x or 8.x** for x86.

3. Install:

```bash
sudo tar -xvf deepstream_sdk_v*.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream
sudo ./install.sh
sudo ldconfig
```

4. Check Python bindings:

```bash
python3 -c "import pyds; print('pyds OK')"
```

If `pyds` is missing, make sure you installed the Python bindings that come with DeepStream.

---

## 3. Create Python Environment

```bash
conda create -n ds_face python=3.10 -y
conda activate ds_face
```

---

## 4. Install Python Packages

```bash
# For CUDA 12.x
pip install numpy cupy-cuda12x opencv-python-headless nvidia-ml-py requests

# If you have CUDA 11.x instead:
# pip install numpy cupy-cuda11x opencv-python-headless nvidia-ml-py requests
```

---

## 5. Verify Installation

```bash
python3 -c "
import pyds
import cupy as cp
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
print('All packages OK')
print('CuPy device:', cp.cuda.Device().id)
"
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| `No module named pyds` | DeepStream Python bindings not installed |
| `Could not create nvinfer` | DeepStream plugins not in `GST_PLUGIN_PATH` |
| CuPy CUDA version mismatch | Install matching `cupy-cuda11x` or `cupy-cuda12x` |
| Jetson | **Not supported** (this code uses x86-only GPU surface API) |

---

## Next Step

Go to → [HOW_TO_RUN.md](HOW_TO_RUN.md)
