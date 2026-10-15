# A07 — Training Run (Engine build record)

**ID:** `Inter-69`-RUN-RF-0001  
**Engineer:** Noman  
**Related:** A06, A17  

---

## Important classification

This is **not** a neural-network training run.  
RetinaFace-R50 was **not trained or fine-tuned** on company data.

This A07 records the **export / engine build** step that produced the locked runtime binary (handbook slot A07 reused for “run that produced the model artifact”).

---

## Run identity

| Field | Value |
| ----- | ----- |
| Type | Pretrained weights → ONNX → TensorRT engine |
| Model | RetinaFace ResNet-50 |
| Input | 1280 × 736 |
| Engine | `retinaface_r50_1280x736.onnx_b1_gpu0_fp16.engine` |
| Engineer | Noman |
| Date of build | `2026-10-12` |


---

## Source of weights

| Step | Detail |
| ---- | ------ |
| PyTorch | Official / public RetinaFace ResNet-50 pretrained (no company train) |
| ONNX export | Converted for DeepStream nvinfer |
| TensorRT | Built by nvinfer on first run or offline; FP16, batch 1 |



---

## What was not done

| Activity | Status |
| -------- | ------ |
| Train from scratch | No |
| Fine-tune on office faces | No |
| SCRFD training | N/A (discarded path) |

---

## Outputs

| File | Role |
| ---- | ---- |
| `retinaface_r50_1280x736.onnx` | Portable graph |
| `*.engine` | Locked TRT plan for project GPU/driver stack |

---

## Metrics from “training”

N/A — no train loss/mAP from a company train loop.  
Runtime quality is reported in **A08 / A11 / A17**.

---

## Status

**ENGINE BUILD RECORDED** — training metrics not applicable.
