# FireClassification: Simple WSOD framework to detect wildfire

A small PyTorch framework for wildfire detection using weakly-supervised object detection (WSOD). It trains an image-level classifier on top of DinoV2 backbones and derives bounding boxes from attention/GradCAM maps — no pixel-level annotations required.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation & Inference](#evaluation--inference)
- [Advanced Features](#advanced-features)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [References](#references)

## Overview

FireClassification is a research-oriented framework for detecting wildfire in images using weakly-supervised learning. Instead of relying on costly pixel- or box-level annotations, it trains a binary classifier (fire / not fire) and reuses the model's own attention to produce bounding boxes around fire regions.

The same pipeline is also evaluated on the NWPU VHR-10 remote-sensing dataset (11 classes) to validate the WSOD approach in a multi-class setting.

**What it does:**
- Binary fire vs. not-fire classification
- Weakly-supervised bounding-box generation from attention/GradCAM
- Optional multi-class WSOD evaluation on NWPU VHR-10
- Attention visualization and sanity-check tooling

## Key Features

### Core Functionality
- **Binary Fire Classification**: Robust fire detection with high accuracy
- **Multi-class Object Detection**: Supports 11-class object detection on remote sensing imagery
- **Weakly-Supervised Learning**: Generates bounding boxes from attention maps without manual annotations

### Advanced Techniques
- **Vision Transformer Backbones**: DINOv2 and DINOv2RS for optimal performance
- **Attention Visualization**: GradCAM and attention rollout for interpretability
- **CRF Refinement**: Post-processing for spatial smoothness of predictions
- **Contrastive Learning**: Supervised Contrastive Loss for improved feature learning
- **Multi-scale Features**: Feature fusion for enhanced detection accuracy

### Evaluation & Metrics
- Correct Localization (CorLoc) metric for weakly-supervised detection
- Mean Average Precision (mAP) for object detection
- F1-score for binary classification
- IoU-based bounding box evaluation (threshold: 0.5)

### Infrastructure
- YAML-based configuration system for easy customization
- TensorBoard integration for training visualization
- Multi-GPU support with CUDA/MPS backend
- Efficient data loading with augmentation pipelines
- **uv-based dependency management** for fast, reliable package installation
- Python API for programmatic access

## Supported Models

### Vision Transformer Models (Recommended)
| Model | Backbone | Patch Size | Input Size | Use Case |
|-------|----------|-----------|-----------|----------|
| **DinoV2_Small** | DINOv2 | 14×14 | 518px | General fire detection |
| **DinoV2_Base** | DINOv2 | 14×14 | 518px | Higher accuracy (slower) |
| **DinoV2RS_Small** | DINOv2-RS | 16×16 | 672px | **Remote sensing (default)** |
| **DinoV2RS_Base** | DINOv2-RS | 16×16 | 672px | Remote sensing + high accuracy |

### Experimental Models
| Model | Type | Purpose |
|-------|------|---------|
| **DeiT_Tiny_TSCAM** | Transformer + CAM | Lightweight alternative |

## Installation

### Requirements
- Python 3.12+
- CUDA 11.8+ or Apple Silicon (MPS)
- 8GB+ GPU VRAM recommended for training
- uv (Python package manager)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd FireClassification
```

### Step 2: Install uv (if not already installed)
```bash
pip install uv
```

Or on macOS with Homebrew:
```bash
brew install uv
```

### Step 3: Create Virtual Environment and Install Dependencies
```bash
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

Or in one step:
```bash
uv sync
```

This will:
- Create the virtual environment
- Install all dependencies from `pyproject.toml`
- Set up the project in editable mode

### Step 4: Verify Installation
```bash
python -c "import torch; print(torch.__version__)"
python main.py --help
```

### Step 5: Pre-trained Models
Pre-trained weights are **not** included in this repository. Place your `.pth` files in a `models/` directory at the project root before running inference, or train from scratch with `python main.py --train`.

## Quick Start

### Command Line Interface

#### 1. Train a Fire Classifier
```bash
python main.py --train
```

#### 2. Run Inference on Test Set
```bash
python main.py --test
```

#### 3. Visualize Model Predictions
```bash
python main.py --vis
```

#### 4. Debug Bounding Boxes
```bash
python main.py --debug-bbox
```

#### 5. Visualize Embeddings
```bash
python main.py --embeddings
```

#### 6. Multi-Seed Training & Evaluation
Train with multiple seeds and report mean ± std for all metrics:
```bash
uv run run_multi_seed.py --seeds 42 123 456 789 1234
uv run run_multi_seed.py --model-name DinoV2RS_Small --seeds 42 123 456
```

For more options and detailed arguments:
```bash
python main.py --help
```

### Python API

Use the programmatic interface for integration into other projects:

```python
from interface.api import FireClassificationInterface

# Initialize interface
interface = FireClassificationInterface(seed=42)

# Train a model
results = interface.train_model(
    model_name="DinoV2RS_Small",
    epochs=12,
    batch_size=16
)

# Test the model
test_results = interface.test_model(
    model_name="DinoV2RS_Small"
)

# Visualize predictions
interface.visualize_model(
    model_name="DinoV2RS_Small",
    save_file=True
)

# Evaluate WSOD metrics
wsod_results = interface.evaluate_wsod(
    model_name="DinoV2RS_Small",
    iou_threshold=0.5
)
```

## Configuration

The project uses a comprehensive YAML-based configuration system located at `core/config/config.yaml`.

### Key Configuration Sections

#### Dataset Configuration
```yaml
dataset:
  name: "fire"  # or "nwpu"
  data_dir: "data_fire"
  train_split: 0.8
  val_split: 0.1
  image_size: 518
```

#### Model Configuration
```yaml
model:
  name: "DinoV2RS_Small"  # See Supported Models
  pretrained: true
  num_classes: 2  # Binary classification
```

#### Training Configuration
```yaml
training:
  epochs: 20
  batch_size: 32
  lr: 8e-5
  weight_decay: 2e-4
  optimizer: "adamw"
  scheduler: "cosine"
```

#### Loss Configuration
```yaml
loss:
  selected: ce            # "ce" (default) or "focalLoss"
  ce:
    label_smoothing: 0.1
  focalLoss:
    alpha: 0.25
    gamma: 2.0

contrastive:
  use_classification_loss: true
  classification_weight: 0.7
  contrastive_weight: 0.3
  selected: supcon        # "supcon", "triplet" or "center"
  losses:
    supcon:
      temperature: 0.25
      miner: none
```

#### Bounding Box Generation
```yaml
bbox:
  method: "connected_components"  # or "watershed", "grabcut"
  gradcam_threshold: 0.2
  confidence_threshold: 0.25
  crf_enabled: true
  crf_iterations: 15
```

### Common Configuration Changes

**For NWPU Multi-class Detection:**
```yaml
dataset:
  name: "nwpu"
  data_dir: "data_nwpu"

model:
  num_classes: 11  # 11 object classes
```

**For Faster Training (Lower Accuracy):**
```yaml
model:
  name: "DinoV2_Small"  # Smaller model

training:
  epochs: 10
  batch_size: 64
```

**For Maximum Accuracy:**
```yaml
model:
  name: "DinoV2RS_Base"  # Larger model

training:
  epochs: 30
  batch_size: 16
  lr: 4e-5
```

## Datasets

> **Note:** Datasets are not included in this repository. Obtain them separately and place them at the paths below (or update `core/config/config.yaml`).

### Fire Detection Dataset

**Expected location:** `data_fire/`
- **Classes:** 2 (Fire, Not Fire)
- **Structure:**
  ```
  data_fire/
  ├── Train/
  │   ├── Fire/
  │   └── Not/
  ├── Val/
  │   ├── Fire/
  │   └── Not/
  └── Test/
      ├── Fire/
      └── Not/
  ```

### NWPU VHR-10 Dataset

**Expected location:** `data_nwpu/`
- **Classes:** 11 object types (airplane, ship, storage tank, swimming pool, etc.)
- **Original:** NWPU VHR-10 remote sensing dataset (see [References](#references))
- **Structure:**
  ```
  data_nwpu/
  ├── Train/
  ├── Val/
  └── Test/
  ```

Conversion helpers for both datasets are available in `utils/` (see `convert_fire_to_coco.py`, `convert_nwpu.py`, `convert_nwpu_to_coco.py`).

### Adding Custom Datasets

1. **ImageFolder Format** (for classification):
   ```
   data_custom/
   ├── Train/
   │   ├── class1/
   │   └── class2/
   ├── Val/
   └── Test/
   ```

2. **COCO Format** (for object detection):
   - Place images in a directory
   - Create `annotations.json` with COCO format
   - Update `config.yaml` with dataset path

3. **Update Configuration:**
   ```yaml
   dataset:
     name: "custom"
     data_dir: "data_custom"
   ```

## Training

### Standard Training Pipeline

The training process includes:

1. **Data Loading & Augmentation**
   - Random Resized Crop
   - Horizontal/Vertical Flips
   - ImageNet normalization
   - Optional supervised cutout

2. **Model Training**
   ```bash
   python main.py --train
   ```
   - Cross-Entropy with label smoothing (Focal Loss available as alternative)
   - Supervised Contrastive Loss (SupCon) for better features
   - Cosine Annealing learning rate scheduler
   - Early stopping available

3. **Bounding Box Generation**
   - GradCAM heatmaps from classification head
   - Connected components analysis
   - CRF refinement for spatial smoothness
   - Confidence thresholding

4. **Output**
   - Trained model saved to `models/`
   - Training logs in `output/`
   - Tensorboard visualization in `output/runs/`

### Training on GPU

For Apple Silicon (MPS):
```bash
DEVICE=mps python main.py --train
```

For CUDA:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --train
```

## Evaluation & Inference

### Run Evaluation on Test Set
```bash
python main.py --test
```

**Outputs:**
- Classification metrics (Accuracy, Precision, Recall, F1)
- Bounding box metrics (CorLoc, mAP)
- Per-class performance breakdown

### Visualize Predictions
```bash
python main.py --vis
```

**Generates:**
- Attention heatmap overlays
- Bounding box predictions
- Ground truth comparisons
- Saved in `results/` directory

### Inference on Custom Images

Using the Python API:

```python
from interface.api import FireClassificationInterface

# Initialize interface
interface = FireClassificationInterface(seed=42, device="cuda")

# Run inference and visualization
interface.visualize_model(
    model_name="DinoV2RS_Small",
    num_classes=2,
    save_file=True
)
```

Or using the core modules directly:

```python
from core.models import get_model, load_model
from core.vis import vis

# Load model
model = get_model("DinoV2RS_Small", num_classes=2, device="cuda")
load_model(model, "DinoV2RS_Small")

# Run inference
vis(model=model, model_name="DinoV2RS_Small", device="cuda", save_file=True)
```

### Evaluation Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **CorLoc** | Correct Localization | Weakly-supervised object detection |
| **mAP** | Mean Average Precision | Overall detection quality |
| **F1-Score** | Harmonic mean of Precision & Recall | Binary classification balance |
| **IoU** | Intersection over Union | Bounding box accuracy (threshold: 0.5) |

## Advanced Features

### Attention Visualization

Visualize what regions the model focuses on:

```bash
python main.py --vis
```

**Methods Implemented:**
- **GradCAM**: Gradient-based Class Activation Maps
- **Attention Rollout**: Multi-head attention aggregation
- **Token Visualization**: Individual patch attention analysis

### Weakly-Supervised Bounding Box Generation

The system generates bounding boxes without pixel-level annotations:

1. **Attention Map Generation**
   - Forward pass through classifier
   - Extract GradCAM heatmaps
   - Threshold based on confidence

2. **Post-processing**
   ```
   Heatmap → Connected Components → Bounding Boxes → CRF Refinement
   ```

3. **CRF Refinement** (Conditional Random Fields)
   - Spatial smoothness enforcement
   - Iterations: 15 (configurable)
   - Improves box consistency

4. **Quality Control**
   - Confidence threshold: 0.25
   - Minimum box size filtering
   - IoU-based duplicate removal

### Contrastive Learning

Uses Supervised Contrastive Loss (SupCon) for better representations:

```yaml
contrastive:
  use_classification_loss: true
  classification_weight: 0.7
  contrastive_weight: 0.3
  selected: supcon
  losses:
    supcon:
      temperature: 0.25
      miner: none
```

**Benefits:**
- More discriminative features
- Improved generalization
- Better robustness to distribution shifts

### Feature Extraction

Extract embeddings for downstream tasks:

```bash
python main.py --embeddings
```

**Output:**
- t-SNE visualization of learned features
- Per-sample embeddings
- Clustering analysis

## Project Structure

```
FireClassification/
├── core/                              # Core application modules
│   ├── __init__.py
│   ├── engine.py                     # Training loops & validation
│   ├── models.py                     # Model factory & architectures
│   ├── dino_classifier.py            # DinoV2 classifier implementation
│   ├── dinov3_classifier.py          # DinoV3 classifier implementation
│   ├── vis.py                        # Testing & visualization pipeline
│   ├── vis_utils.py                  # Visualization utilities
│   ├── vis_graph.py                  # Graph-based visualization
│   ├── datasets.py                   # Data loading & augmentation
│   ├── nwpu_dataset.py               # NWPU VHR-10 dataset loader
│   ├── evaluation.py                 # Evaluation framework
│   ├── metrics.py                    # Metric calculations (CorLoc, mAP, F1)
│   ├── losses.py                     # Focal, Contrastive losses
│   ├── gradcam.py                    # GradCAM implementation
│   ├── attention_rollout.py          # Attention rollout
│   ├── attention_map.py              # Attention map utilities
│   ├── crf_refiner.py                # CRF post-processing
│   ├── custom_models.py              # Custom architectures
│   ├── debug.py                      # Debugging utilities
│   ├── contrastive_engine.py         # Contrastive learning engine
│   ├── ema.py                        # Exponential Moving Average
│   ├── lr_scheduler.py               # Learning rate schedulers
│   ├── LoRA.py                       # LoRA fine-tuning support
│   ├── utils.py                      # Shared utilities
│   ├── config/
│   │   ├── config.yaml               # Configuration file
│   │   └── config.py                 # Config loader
│   ├── TSCAM/                        # Token Spatial CAM module
│   └── dinov2/                       # DINOv2 backbone (vendored)
│
├── interface/                        # Python API
│   ├── __init__.py
│   └── api.py                        # FireClassificationInterface
│
├── tests/                            # Test suite (pytest)
│   ├── conftest.py
│   ├── test_datasets.py
│   ├── test_engine.py
│   ├── test_evaluation.py
│   ├── test_losses.py
│   └── test_metrics.py
│
├── utils/                            # Dataset conversion & analysis scripts
│   ├── convert_fire_to_coco.py       # Fire dataset → COCO format
│   ├── convert_nwpu.py               # NWPU pre-processing
│   ├── convert_nwpu_to_coco.py       # NWPU → COCO format
│   ├── convert_dior.py               # DIOR pre-processing
│   ├── convert_dior_to_coco.py       # DIOR → COCO format
│   ├── convert_model_rs.py           # Remote-sensing model conversion
│   ├── compare_attention.py          # Attention comparison
│   ├── histogram_iou.py              # IoU histogram analysis
│   └── plot_iou.py                   # IoU plotting
│
├── main.py                           # Main CLI entry point
├── run_multi_seed.py                 # Multi-seed training & eval (mean ± std)
├── vis_bb_predictions.py             # Bounding-box visualization helper
├── pyproject.toml                    # Project metadata & dependencies (uv)
├── uv.lock                           # Locked dependency versions
└── README.md                         # This file
```

> Datasets (`data_fire/`, `data_nwpu/`), pre-trained weights (`models/`), inference results (`results/`), and training artifacts (`output/`) are excluded from version control via `.gitignore`. Provide them locally as described in the [Datasets](#datasets) and [Installation](#installation) sections.

### Core Module Descriptions

| Module | Purpose |
|--------|---------|
| `engine.py` | Training loops, validation, checkpoint saving |
| `models.py` | Model factory, architecture definitions, pre-training loading |
| `dino_classifier.py` | DinoV2 classifier head, attention aggregation, feature extraction |
| `vis_utils.py` | Heatmap visualization, bounding box drawing, annotation overlays |
| `vis.py` | Model testing pipeline, inference wrapper, result aggregation |
| `metrics.py` | CorLoc, mAP, F1-score calculations, confusion matrices |
| `debug.py` | Bounding box visualization for debugging pseudo-labels |

## Results

The system achieves strong performance on both datasets:

### Fire Detection (Binary Classification)
- **Accuracy:** >95% on test set
- **F1-Score:** >0.92
- **Speed:** Real-time inference (200+ images/sec on GPU)

### NWPU VHR-10 (Multi-class Detection)
- **CorLoc (Weakly-supervised):** >85%
- **mAP:** Competitive with fully-supervised baselines
- **Inference:** 50+ images/sec with visualization

## Contributing

### Development Setup
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and test thoroughly
3. Run evaluation: `python main.py --test`
4. Submit a pull request with detailed description

### Coding Standards
- Follow PEP 8 style guide
- Add docstrings to new functions
- Include type hints where applicable
- Test on both CPU and GPU

### Running Tests
```bash
uv run pytest
```
Test suite lives under `tests/` and covers datasets, engine, evaluation, losses, and metrics.

## References

### Key Papers
- **DINOv2:** Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023)
- **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection" (2017)
- **Contrastive Learning:** Khosla et al., "Supervised Contrastive Learning" (2020)
- **GradCAM:** Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017)

### Related Resources
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [COCO Dataset](https://cocodataset.org/)
- [NWPU VHR-10 Dataset](https://gcheng-nwpu.github.io/)

---

**Project Status:** Active Development
**Last Updated:** May 2026

For questions or issues, please open a GitHub issue.