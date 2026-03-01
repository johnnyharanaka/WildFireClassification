# DINOv2 — Fork for WSOD (Weakly Supervised Object Detection)

This is a fork of [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) adapted for use as a submodule within the [SimpleML](https://github.com/johhny-haranaka/SimpleML) framework, with a focus on **Weakly Supervised Object Detection (WSOD)** using DINOv2 attention maps as localization signals.

---

## Why this fork?

The original DINOv2 repository is structured as a standalone package (`dinov2/` nested inside `dinov2/`), with many extras (Cell-DINO, Channel-Adaptive-DINO, XRay-DINO, notebooks, conda configs, etc.) that are not needed here.

This fork:

- **Flattens the package structure** — modules previously under `dinov2/dinov2/` were moved to the top level so they can be imported directly as a submodule.
- **Removes unused extensions** — Cell-DINO, Channel-Adaptive-DINO, XRay-DINO, notebooks, conda files, and unrelated scripts were stripped out to keep the codebase lean.
- **Adds the `app/` module** — new files built on top of the DINOv2 backbone for WSOD and fine-tuning workflows, integrated with the SimpleML registry.

---

## What was added (`app/`)

| File | Description |
|---|---|
| `dino_classifier.py` | `DinoClassifier` — image-level classification head over DINOv2. Supports ImageNet and Remote Sensing pretrained weights via `DinoModelSize` and `PretrainType` enums. |
| `dino_detector.py` | `DinoDetector` — WSOD detector. Uses the CLS token's attention over spatial patches as a localization signal to generate bounding boxes without instance-level annotations. Registered as `@MODELS.register`. |
| `detection_utils.py` | Utility functions: attention map extraction from the last transformer layer, dynamic thresholding, and bounding box generation via connected components. |
| `multi_start.py` | Multi-start optimization strategy to improve localization stability in the WSOD pipeline. |
| `LoRA.py` | LoRA (Low-Rank Adaptation) for efficient fine-tuning of DINOv2 attention layers (Q and V projections). |

---

## WSOD approach

Detection is performed without bounding box annotations at training time:

1. The DINOv2 backbone extracts patch-level features.
2. The CLS token's attention weights over spatial patches are used as a heatmap.
3. A dynamic threshold is applied to the heatmap.
4. Connected components on the thresholded map produce bounding box proposals.
5. Image-level classification loss provides the only supervision signal during training.

---

## Usage within SimpleML

`DinoDetector` is registered in the SimpleML model registry and can be used directly via the `API` builder or a config file:

```python
from simpleml import API

results = (
    API()
    .model("DinoDetector", model_name="DinoV2_Base", num_classes=2, pretrained=True)
    .loss("CrossEntropyLoss")
    .optimizer("AdamW", lr=1e-4)
    .data(train="data/train", val="data/val")
    .train_config(epochs=20, device="mps")
    .fit()
)
```

---

## Original project

- **Repository:** https://github.com/facebookresearch/dinov2
- **Paper:** [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **License:** Apache 2.0 (see `LICENSE`)

All original model weights, training code, and evaluation pipelines remain under their respective licenses.
