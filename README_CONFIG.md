# Face Blur — config layout

```text
project_root/
  configs/
    pipeline.yaml      # mux size, out_dir, source, retime
    detection.yaml     # RetinaFace ONNX/engine, conf, NMS
    blur.yaml          # pad, blur_frac, hold
    camera.yaml        # RTSP / ONVIF
    encode.yaml        # NVENC + codec
    preview.yaml       # live window
    logging.yaml       # metrics, samples, detected_instances
  src/
    config_loader.py   # load_yaml / load_all
```

## Usage

```python
from src.config_loader import load_all

cfg = load_all()
conf = cfg["detection"]["conf"]
engine = cfg["detection"]["engine_path"]
```

CLI flags can still override config for one-off experiments; **defaults should come from YAML**.

## Secrets

Do not commit real passwords. In `camera.yaml` set `onvif_password_env: CAM_ONVIF_PASSWORD` and:

```bash
export CAM_ONVIF_PASSWORD='Pass123#'
```

Build RTSP URI with URL-encoding (`#` → `%23`).

## Model scope

Only **RetinaFace-R50**. SCRFD configs are intentionally absent.
