# TB Chest X-ray Detection — Shortcut Learning Study

MobileNetV2 TB-CXR classifier that achieves near-perfect in-domain AUC (0.9995) but collapses on independent hospital cohorts (0.70–0.78 AUC, ~0.25 gap). The central finding is that this generalisation gap resists two standard mitigations — harmonisation/augmentation (+0.02 gain) and lung segmentation (Dice 0.975) which *worsened* unbiased external performance, contradicting prior COVID-CXR work.

Targeting IEEE journal publication.

## Requirements

- macOS with Apple Silicon (M-series)
- Python 3.10+
- ~10 GB free disk space (datasets)
- Kaggle account (free)

## 1 · One-time environment setup

```bash
cd /Users/alvinchow/analysis

python3 -m venv .venv
source .venv/bin/activate

pip install tensorflow-macos tensorflow-metal \
    opencv-python-headless scipy scikit-learn \
    pillow matplotlib seaborn kaggle jupyter ipykernel

python -m ipykernel install --user --name=tb-analysis --display-name "TB Analysis (.venv)"
```

> **Not on Apple Silicon?** Replace `tensorflow-macos tensorflow-metal` with `tensorflow`.

## 2 · Kaggle credentials

The training dataset is downloaded via the Kaggle API.

1. Go to [kaggle.com](https://www.kaggle.com) → your profile → **Settings** → **API** → **Create New Token**
2. Open `kaggle.json` in this folder and replace the placeholders with your real values:

```json
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}
```

The notebook copies it to `~/.kaggle/kaggle.json` automatically on first run.

## 3 · Open the notebook

**In VS Code (recommended)**

First, install the `code` CLI if you haven't already:
1. Open VS Code from Applications
2. `Cmd+Shift+P` → **Shell Command: Install 'code' command in PATH**
3. Restart your terminal

Then:
```bash
source .venv/bin/activate
code analysis.ipynb
```

If the `code` command still isn't available, open directly with:
```bash
open -a "Visual Studio Code" analysis.ipynb
```

**In classic Jupyter**

```bash
source .venv/bin/activate
jupyter notebook analysis.ipynb
```

## 4 · Select the kernel

In VS Code, click **Select Kernel** (top-right of the notebook) → **TB Analysis (.venv)**.

In Jupyter, the kernel menu is at **Kernel → Change kernel → TB Analysis (.venv)**.

## 5 · Run

**Kernel → Restart & Run All** (VS Code: `Cmd+Shift+P` → *Run All Cells*).

On first run the notebook will:
- Download the Kaggle TB dataset (~2 GB) → `data/kaggle/`
- Download NLM Montgomery (~1.5 GB) + Shenzhen (~3.5 GB) datasets → `data/`
- Train the baseline model (~60–90 min on M4)
- Train the robust model (Mitigation A, ~60–90 min)
- Train the U-Net segmenter (~3–5 min) and produce masked datasets (~3–5 min)
- Train the masked classifier (Mitigation B, ~30–60 min)
- Run 5-fold cross-validation (~60–100 min)
- Convert to `tb_detector_dynamic.tflite`
- Total: **several hours** on first run; each training section is skipped if the `.keras` file already exists

**Subsequent runs skip all downloads automatically** — datasets are checked by directory existence before any network request is made.

## Data layout

```
analysis/
├── analysis.ipynb
├── kaggle.json               ← fill in your credentials
├── tb_detector.keras         ← produced on first run
├── tb_detector_robust.keras
├── tb_lung_unet.keras
├── tb_detector_masked.keras
├── tb_cv_fold{0-4}.keras
├── cv_results.json
├── backbone_results.json
├── tb_detector_dynamic.tflite
└── data/
    ├── kaggle/
    │   └── TB_Chest_Radiography_Database/
    ├── montgomery/
    ├── montgomery.zip
    ├── shenzhen/
    └── shenzhen.zip
```

To store datasets elsewhere, change `DATA_DIR` in cell 4 of the notebook.

## What the notebook produces

| Output | Description |
|---|---|
| ROC-AUC + 95% CI | Bootstrapped over 2 000 resamples |
| PR-AUC | Precision-recall curve (more informative under 5:1 class imbalance) |
| Sensitivity / Specificity | At operating threshold selected on validation set (≥ 0.90 sensitivity) |
| External validation AUC | Per-cohort (Montgomery, Shenzhen) and combined |
| Reliability diagrams | ECE before and after temperature scaling (calibration fails — degenerate T=0.12) |
| Grad-CAM / Integrated Gradients | In-domain and external samples |
| `tb_detector.keras` | Main trained classifier |
| `tb_detector_robust.keras` | Mitigation A: CLAHE + augmentation + multi-source |
| `tb_lung_unet.keras` | U-Net lung segmenter (Montgomery masks, Dice 0.975) |
| `tb_detector_masked.keras` | Mitigation B: classifier retrained on segmented images |
| `tb_cv_fold{0-4}.keras` | 5-fold cross-validation models |
| `cv_results.json` | Per-fold AUC metrics (in-domain / Montgomery / Shenzhen / gap) |
| `backbone_results.json` | Backbone study results (MobileNetV2, DenseNet-121, EfficientNet-B0) |
| `tb_detector_dynamic.tflite` | Dynamic-range quantized TFLite model (AUC Δ −0.0007, ~4 ms/image) |

> **Note on TFLite:** dynamic-range quantization is used instead of INT8 because `tensorflow-macos` crashes on INT8 conversion via the MLIR/LLVM path. The conversion runs in an isolated subprocess via `SavedModel` export to survive the crash without killing the kernel.
