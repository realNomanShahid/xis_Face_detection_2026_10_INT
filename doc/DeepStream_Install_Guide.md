# DeepStream Face Blur – Complete Installation & Run Guide

**Simple step-by-step guide** for Ubuntu 22.04 + NVIDIA GPU  
(Tested with RTX 5080, Driver 595.99, DeepStream 7.1)

---

## 1. System Requirements

| Item              | Required                          |
|-------------------|-----------------------------------|
| OS                | Ubuntu 22.04 LTS                  |
| GPU               | NVIDIA GPU (RTX series preferred) |
| NVIDIA Driver     | Working (`nvidia-smi` must work)  |
| CUDA (from driver)| 12.x or higher                    |
| Python            | 3.10 (important!)                 |

---

## 2. Fix NVIDIA Driver (if `nvidia-smi` fails)

If you see **"Driver/library version mismatch"**:

```bash
# Just reboot the machine
sudo reboot
```

After reboot, check:

```bash
nvidia-smi
```

You must see a clean table with GPU name and driver version.  
**Do not continue until `nvidia-smi` works.**

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
  build-essential cmake
```

---

## 4. Install CUDA Toolkit 12.6 (only toolkit, no new driver)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
```

---

## 5. Install TensorRT (needed by DeepStream)

```bash
sudo apt install -y \
  libnvinfer10 libnvinfer-dev \
  libnvinfer-plugin10 libnvinfer-plugin-dev \
  libnvonnxparsers10 libnvonnxparsers-dev \
  libnvinfer-headers-dev libnvinfer-headers-plugin-dev
```

---

## 6. Download & Install DeepStream 7.1

### Download link (official):

**https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=7.1**

Download this file only:

```
deepstream-7.1_7.1.0-1_amd64.deb   (~742 MB)
```

You need a free NVIDIA account to download.

### Install:

```bash
cd ~/Downloads
sudo dpkg -i deepstream-7.1_7.1.0-1_amd64.deb
sudo apt --fix-broken install -y
```

### Set environment variables (permanent):

```bash
echo 'export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=/opt/nvidia/deepstream/deepstream/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify:

```bash
deepstream-app --version
```

You should see:

```
deepstream-app version 7.1.0
DeepStreamSDK 7.1.0
```

---

## 7. Create Python Environment (Python 3.10 is required)

The official `pyds` wheel only works with **Python 3.10**.

```bash
# Using conda (recommended)
conda create -n ds_face python=3.10 -y
conda activate ds_face
```

---

## 8. Install Python packages

### Create `requirements.txt`:

```txt
numpy
cupy-cuda12x
opencv-python-headless
nvidia-ml-py
requests
```

### Install:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Install DeepStream Python bindings (`pyds`)

### Download official wheel (Python 3.10 + x86_64):

```bash
wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl
```

### Install:

```bash
pip install pyds-1.2.0-cp310-cp310-linux_x86_64.whl
```

### Test:

```bash
python -c "import pyds; print('pyds OK')"
python -c "import cupy; print('cupy OK')"
```

Both must print OK.

---

## 10. Prepare model files

You need:

| File                        | Description                          |
|-----------------------------|--------------------------------------|
| `scrfd_2.5g_bnkps.onnx`     | Original ONNX model (required once)  |
| `scrfd_2.5g_fp16.trt`       | Your old engine (may not work)       |

**Important:**  
If your `.trt` / `.engine` was built with a different TensorRT version, DeepStream cannot load it.  
In that case you **must** pass the ONNX file so DeepStream can rebuild a compatible engine.

---

## 11. Run the Face Blur Script

### First run (rebuilds engine if needed):

```bash
conda activate ds_face

export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
export PATH=/opt/nvidia/deepstream/deepstream/bin:$PATH

python face_blur_deepstream.py \
  --source video.mp4 \
  --engine scrfd_2.5g_fp16.trt \
  --onnx scrfd_2.5g_bnkps.onnx \
  --output output_blurred.mp4
```

DeepStream will create a new engine file, for example:

```
scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine
```

### Later runs (use the new engine only):

```bash
python face_blur_deepstream.py \
  --source video.mp4 \
  --engine scrfd_2.5g_bnkps.onnx_b1_gpu0_fp16.engine \
  --output output_blurred.mp4
```

---

## 12. Useful optional flags

| Flag              | Meaning                              | Example        |
|-------------------|--------------------------------------|----------------|
| `--max-seconds`   | Stop after N seconds                 | `--max-seconds 60` |
| `--width`         | Mux width                            | `--width 1920` |
| `--height`        | Mux height                           | `--height 1080`|
| `--sample-every`  | Save before/after images every N frames | `--sample-every 50` |
| `--leaky`         | Drop old frames on live streams      | `--leaky`      |
| `--conf`          | Detection confidence                 | `--conf 0.35`  |
| `--blur-frac`     | Blur strength                        | `--blur-frac 0.25` |

---

## 13. Output files

After a successful run you will get:

- `output_blurred.mp4`          → final video with faces blurred
- `metrics.json`                → performance report
- `sample_frames/`              → before/after images (if `--sample-every` used)
- `*.engine`                    → new TensorRT engine built by DeepStream

---

## 14. Common Problems & Solutions

| Problem                                      | Solution |
|----------------------------------------------|----------|
| `nvidia-smi` fails with version mismatch     | Reboot the machine |
| `No module named 'pyds'`                     | Use Python 3.10 + install the official wheel |
| Engine version mismatch (243 vs 240)         | Pass `--onnx` so DeepStream rebuilds the engine |
| `Could not create 'nvurisrcbin'`             | Check `GST_PLUGIN_PATH` and `LD_LIBRARY_PATH` |
| `cupy` CUDA error                            | Use `cupy-cuda12x` (not cuda11x) |

---

## 15. Quick Checklist

- [ ] `nvidia-smi` works
- [ ] `deepstream-app --version` shows 7.1.0
- [ ] Conda env with **Python 3.10**
- [ ] `python -c "import pyds"` works
- [ ] `python -c "import cupy"` works
- [ ] ONNX file available (for first run)
- [ ] Environment variables set

---

## 16. Download Links Summary

| Item                          | Link |
|-------------------------------|------|
| DeepStream 7.1 `.deb`         | https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=7.1 |
| `pyds` Python wheel (3.10)    | https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl |
| CUDA keyring                  | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb |

---

**Done!**  
You now have a working DeepStream + TensorRT face-blur pipeline.  
Save this file and follow the steps in order on any similar machine.
