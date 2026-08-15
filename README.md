# TB Chest X-ray Detection — a Cross-Dataset Study of Shortcut Learning

Code and results for the paper *"Tuberculosis Is Not in the Corner of the Image: A Cross-Dataset
Study of Shortcut Learning and the Limits of Lung Segmentation in Chest-Radiograph Classifiers"*
(under submission). LaTeX sources for the manuscript are in [`tmlr/`](tmlr/), with an IEEEtran
port in [`latex/`](latex/).

## Findings in one paragraph

A MobileNetV2 TB classifier trained on the widely used Kaggle TB Chest Radiography Database
reaches near-perfect in-domain AUC (0.9995) but collapses to 0.70–0.78 on two independent NLM
hospital cohorts (Montgomery, Shenzhen) — a ~0.25 AUC generalisation gap that is stable across
five-fold cross-validation and not attributable to duplicate-image contamination. The gap resists
the standard mitigations: harmonisation + augmentation + multi-source training gains only +0.02,
and lung segmentation with a high-quality U-Net (Dice 0.975) *degrades* the unbiased external
cohort in every fold (−0.057 ± 0.011). The failure is architecture-dependent: DenseNet-121 and
EfficientNet-B0 match MobileNetV2 in-domain yet each collapses on a *different* external cohort,
so the two cohorts rank the three architectures in opposite orders — and a frozen-feature probe
shows these failure profiles pre-exist in the ImageNet-pretrained representations before any
fine-tuning. Masking homogenises all three backbones into a common 0.61–0.77 band. Conclusion:
out-of-source external validation on more than one cohort is mandatory, and neither segmentation
nor backbone choice validated on a single cohort confers robustness.

## Repository layout

| Path | Contents |
|---|---|
| `analysis.ipynb` | Main notebook: data ingestion → training → evaluation → calibration → explainability → mitigations → TFLite. Runs top-to-bottom. |
| `backbone_study.py` | **Canonical** backbone study (MobileNetV2 / DenseNet-121 / EfficientNet-B0 × raw / masked × 5 folds). Split-verified against the notebook, resumable via `backbone_results.json`. Supersedes the notebook's in-notebook backbone cells. |
| `run_backbone_study.sh` | Runs the remaining backbone folds, one process per (backbone, condition). |
| `densenet_probe.py` | Frozen-feature probe: ImageNet features + logistic head, plus the brightness-shortcut analysis. |
| `make_figures.py` | Generates all five paper figures into `figures/` (PNG + PDF). CPU-only. |
| `backbone_results.json` | All 30 backbone-study fold results (AUC per cohort). |
| `cv_results.json` | Baseline 5-fold cross-validation results. |
| `probe_results.json` | Frozen-probe AUCs and score percentiles for all three backbones. |
| `figures/` | Publication figures. |
| `latex/` | IEEEtran port; compiles with `tectonic main.tex`. |
| `tmlr/` | Manuscript source (TMLR style); `build.sh` produces the submission and preprint PDFs. |
| `*.keras`, `*.tflite` | Trained models (baseline, robust, U-Net segmenter, masked, CV folds, TFLite export). |

## Reproducing

### 1. Environment

Python 3.9+. On Apple Silicon:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On other platforms, replace `tensorflow-macos`/`tensorflow-metal` in `requirements.txt` with
`tensorflow==2.16.2`.

### 2. Data (~10 GB free disk)

- **Kaggle TB Chest Radiography Database** (~2 GB): downloaded automatically on first notebook run
  via the Kaggle API. Put your API token in `kaggle.json` (placeholder provided; get one from
  kaggle.com → Settings → API).
- **NLM Montgomery (~1.5 GB) and Shenzhen (~3.5 GB) sets**: downloaded automatically by the
  notebook into `data/`.

All downloads are idempotent — existing files are skipped. Dataset licenses remain with their
providers (Qatar University/University of Dhaka for the Kaggle DB; U.S. National Library of
Medicine for Montgomery/Shenzhen); this repository does not redistribute any image data.

### 3. Run

```bash
jupyter notebook analysis.ipynb   # Kernel → Restart & Run All
```

First full run takes several hours (baseline + robust + segmenter + masked models + 5-fold CV +
TFLite); each training section skips itself if its `.keras` file already exists. Then:

```bash
./run_backbone_study.sh           # backbone study (resumable; skips cached folds)
python densenet_probe.py          # frozen-feature probe + brightness analysis
python make_figures.py            # regenerate the five paper figures
```

`backbone_study.py` verifies its reproduced data split against the notebook's cached mask
directory before training and aborts on mismatch, so script results are guaranteed comparable
with notebook results.

## Key result files

`backbone_results.json` keys are `"<backbone>|<raw|masked>|<fold>"` with per-cohort AUCs
(`in`, `mont`, `shen`). The paper's Table V is the per-condition mean ± std over folds; Fig. 5
combines it with `probe_results.json`.

## Known limitations

Documented in the paper (Section V): no patient identifiers in the Kaggle DB (image-level
splitting), segmenter trained on Montgomery (Shenzhen is the unbiased external test), three
ImageNet CNN backbones only, retrospective public data.

## License

Code is MIT-licensed (see `LICENSE`). Cite via `CITATION.cff`.
