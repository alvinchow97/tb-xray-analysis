#!/bin/zsh
# Builds tmlr-supplement.zip: the anonymised code/results package to attach to the
# TMLR submission on OpenReview. Excludes LICENSE and CITATION.cff (they carry the
# author name) and rewrites any identifying URLs or names inside text files.
#
# Always re-run this after changing code; never hand-edit the zip.
set -e
cd "$(dirname "$0")"

OUT="tmlr-supplement.zip"
STAGE=$(mktemp -d)

cp analysis.ipynb backbone_study.py run_backbone_study.sh densenet_probe.py \
   make_figures.py requirements.txt \
   backbone_results.json cv_results.json probe_results.json "$STAGE/"

# Anonymised README in place of the identifying one.
cat > "$STAGE/README.md" <<'EOF'
# Supplementary code — TB chest X-ray cross-dataset study

Anonymised supplement accompanying the TMLR submission. The public repository is withheld
during review and will be released on acceptance.

## Contents

| File | Purpose |
|---|---|
| `analysis.ipynb` | Main notebook: ingestion, training, evaluation, calibration, explainability, mitigations |
| `backbone_study.py` | Canonical backbone study (3 backbones x raw/masked x 5 folds); split-verified and resumable |
| `run_backbone_study.sh` | Runs remaining folds, one process per (backbone, condition) |
| `densenet_probe.py` | Frozen-feature probe and brightness-shortcut analysis |
| `make_figures.py` | Regenerates all paper figures |
| `backbone_results.json` | All 30 backbone fold results (per-cohort AUC) |
| `cv_results.json` | Baseline 5-fold cross-validation results |
| `probe_results.json` | Frozen-probe AUCs for all three backbones |
| `requirements.txt` | Pinned dependency versions used for the reported results |

## Reproducing

Python 3.9+; `pip install -r requirements.txt` (on non-Apple-Silicon, replace
`tensorflow-macos`/`tensorflow-metal` with `tensorflow==2.16.2`).

Datasets are public and downloaded by the notebook on first run: the TB Chest Radiography
Database via the Kaggle API, and the NLM Montgomery and Shenzhen sets. No image data is
redistributed here.

Then: run the notebook top to bottom, followed by `./run_backbone_study.sh`,
`python densenet_probe.py`, and `python make_figures.py`.

`backbone_study.py` verifies its reproduced data split against the cached mask directory before
training and aborts on mismatch, so script results are guaranteed comparable with notebook results.
Random seeds are fixed; per-fold seeds are derived deterministically.
EOF

# Scrub identifying strings from every text file (incl. notebook JSON).
python3 - "$STAGE" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
pats = [
    (re.compile(r'https?://github\.com/alvinchow97[^\s"\'\\)]*'), 'ANONYMISED_REPOSITORY_URL'),
    (re.compile(r'/Users/alvinchow[^\s"\'\\)]*'), '/path/to/analysis'),
    (re.compile(r'alvinchow97@gmail\.com'), 'anonymous@example.com'),
    (re.compile(r'[Aa]lvin\s+Chow'), 'Anonymous Author'),
    (re.compile(r'alvinchow'), 'anonymous'),
]
hits = 0
for f in root.rglob('*'):
    if not f.is_file():
        continue
    try:
        t = f.read_text(encoding='utf-8')
    except (UnicodeDecodeError, ValueError):
        continue
    o = t
    for p, r in pats:
        t = p.sub(r, t)
    if t != o:
        f.write_text(t, encoding='utf-8')
        hits += 1
print(f"  scrubbed identifying strings in {hits} file(s)")
PY

# Fail loudly rather than shipping a leak.
if grep -rIl -i "alvinchow\|alvin chow" "$STAGE" >/dev/null 2>&1; then
    echo "ABORT: identifying strings still present:"
    grep -rIl -i "alvinchow\|alvin chow" "$STAGE"
    rm -rf "$STAGE"; exit 1
fi

rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OLDPWD/$OUT" .)
rm -rf "$STAGE"
echo "built $OUT ($(du -h "$OUT" | cut -f1)) — verified free of identifying strings"
