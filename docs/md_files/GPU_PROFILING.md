# GPU / CPU Profiling

This is the **same face-blur pipeline** as the main file, with built-in **CPU and GPU measurement**. You run it the same way. When it finishes it saves a report that shows how much CPU and GPU the pipeline used, and where the time goes.

| | Main pipeline | Profiling version |
|---|---|---|
| File | `src/retinaface_face_detection.py` | `src/GPU_Profilig/retinaface_face_blur_gpu_profiling.py` |
| Detects and blurs faces | Yes | Yes (same model, same blur) |
| Reads `configs/*.yaml` | Yes | **No**, settings come from the command line only |
| CPU / GPU report | No | **Yes** |

Use the main pipeline for normal runs. Use the profiling version when you want to know how heavy the pipeline is.

---

## 1. What it gives you

Everything the main pipeline writes (blurred video, `metrics.json`, logs), plus three profiling files, all saved in `logs/`:

| File | What it is |
|------|------------|
| `optimization_report.txt` | The report in plain text (also printed when the run ends) |
| `optimization_report.json` | The same numbers, for scripts |
| `optimization_report_timeline.csv` | CPU and GPU usage every 0.5 s (open in Excel to plot) |

A sampler thread starts with the program and records system CPU %, this process's CPU cores, RAM, GPU busy %, NVENC %, NVDEC %, VRAM, power, temperature and clock speed. Each stage of the per-frame path is also timed.

---

## 2. How to run

Same setup as the main pipeline (see [INSTALL_DEEPSTREAM.md](INSTALL_DEEPSTREAM.md) and [TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md)). No extra packages are needed. Run from the **project root**.

**Video file, first 60 seconds:**

```bash
python3 src/GPU_Profilig/retinaface_face_blur_gpu_profiling.py \
  --source video.mp4 \
  --onnx retinaface_r50_1280x736.onnx \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --conf 0.15 \
  --max-seconds 60
```

**Live camera with the preview window:**

```bash
python3 src/GPU_Profilig/retinaface_face_blur_gpu_profiling.py \
  --source "rtsp://<user>:<password>@<camera-ip>:554/<path>" \
  --onnx retinaface_r50_1280x736.onnx \
  --engine retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine \
  --conf 0.15 --width 1280 --height 720 \
  --preview
```

`--preview` needs a desktop session. Press **Ctrl+C** at any time: the video is saved and the report is printed.

**Pure speed test** (switches off the per-stage timing, images and logs):

```bash
... --no-profile --no-save-detections --no-logs
```

---

## 3. Important: no config files, different defaults

This script does **not** read `configs/*.yaml`. Its built-in defaults differ from the main pipeline, so set these yourself:

| Setting | Profiling default | Main pipeline (from YAML) | What to do |
|---------|-------------------|---------------------------|------------|
| `--onnx` | `retinaface_r50_640x640.onnx` | `retinaface_r50_1280x736.onnx` | **Always pass `--onnx`** (the model size is read from the file name) |
| `--conf` | 0.5 | 0.15 | Pass `--conf 0.15` to match the main pipeline |
| `--width` / `--height` | 1920 × 1080 | 1536 × 864 | Match your source |
| `--source` | required | falls back to `camera.yaml` | Always pass `--source` or `--rutube-page` |

Output files go in `logs/` by default (`--out-dir`).

---

## 4. Profiling options

| Option | Default | Meaning |
|--------|---------|---------|
| `--no-profile` | off | Turn off per-stage timing (the CPU/GPU usage sampler stays on) |
| `--prof-period` | 0.5 | Seconds between CPU/GPU samples |
| `--profile-report` | `optimization_report.txt` | Report file name (`.json` and `_timeline.csv` are written next to it) |
| `--warmup` | 30 | Frames left out of the timings |
| `--max-seconds` | 0 | Stop after N seconds (0 = run to the end) |

All other options (`--preview`, `--output`, `--codec`, `--out-fps`, ...) work as in the main pipeline. See `--help`.

---

## 5. How to read the report

The report has 8 sections:

| Section | What it shows |
|---------|---------------|
| 1. Headline numbers | FPS, latency, GPU and CPU use, GPU/CPU split. Start here. |
| 2. Usage over time | CPU and GPU at start, steady state, peak and finish |
| 3. Function map | Time per stage (GPU ms, CPU ms, WAIT ms) and which function runs on GPU or CPU |
| 4. Workload split | GPU % and CPU % of the measured busy time per frame |
| 5. Code split | GPU % and CPU % of the executable lines in the file |
| 6. CPU time accounting | Where the CPU time goes |
| 7. Bottlenecks | Automatic hints |
| 8. Definitions | What each number means, and caveats |

**In section 3:** *GPU ms* is time on the GPU, *CPU ms* is time the CPU spends working, and *WAIT ms* is time the CPU waits for the GPU.

**Workload split and code split are different on purpose.** Most of the *lines* of code are CPU code (setup, reports, arguments), but most of the *time per frame* is spent on the GPU.

### Key numbers

| Number | Meaning |
|--------|---------|
| Steady-state FPS | Speed the run actually had. A live camera is limited by its own frame rate. |
| Estimated max FPS | What the pipeline could handle, and which stage limits it |
| Headroom | Estimated max FPS divided by actual FPS. Above 1 means spare capacity. |
| GPU busy % | How busy the GPU was |
| CPU cores | How many CPU cores this process used |
| Privacy audit | `PASS` if every frame was blurred and checked |

### Example

From `results/GPU_Profiling/` (RTX 5080, live RTSP camera at 30 FPS, preview on, `conf` 0.15, 169 s, 5,058 frames, 0 lost):

| Number | Value |
|--------|-------|
| Steady-state FPS | 30.0 (camera limit) |
| Estimated max FPS | 518.2 (limited by `nvinfer`) |
| Headroom | 17.3× |
| GPU busy (avg / peak) | 6% / 8% |
| CPU used by this process | 0.16 cores |
| VRAM peak | 4,956 MB |
| Workload split | GPU 71.2% / CPU 28.8% |
| Privacy audit | PASS |

---

## 6. Notes

- The first run builds the TensorRT engine, which makes the startup phase slow and CPU/GPU heavy. Run twice and read the second report.
- The report prints the stream URL on its `Source` line. If the RTSP URL contains a password, **do not commit or share the report** without removing it.
- Jetson is not supported (x86 Linux + NVIDIA GPU only).