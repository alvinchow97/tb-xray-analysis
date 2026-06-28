# TB Chest X-ray Detection

Binary classifier (Tuberculosis vs Normal) using two-stage MobileNetV2 transfer learning, with external validation, calibration, and INT8 TFLite deployment.

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
- Train the model (~60–90 min on M4 CPU/GPU)
- Produce evaluation plots, Grad-CAM visualisations, and a `tb_detector_int8.tflite` file

**Subsequent runs skip all downloads automatically** — datasets are checked by directory existence before any network request is made.

## Data layout

```
analysis/
├── analysis.ipynb
├── kaggle.json          ← fill in your credentials
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
| Reliability diagrams | ECE before and after temperature scaling |
| Grad-CAM overlays | In-domain and external samples — check attention falls inside lungs |
| `tb_detector_int8.tflite` | INT8 quantized model with AUC delta and latency benchmark |
