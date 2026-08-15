#!/bin/zsh
# Runs the remaining backbone-study folds, one (backbone, condition) per process
# to avoid the tensorflow-metal cross-session memory leak. caffeinate keeps the
# Mac awake for the duration. Resumable: re-run this script after any interruption.
set -e
cd /Users/alvinchow/analysis
source .venv/bin/activate

LOG=backbone_study.log
echo "=== backbone study run started $(date) ===" | tee -a "$LOG"

# densenet121 raw is fully cached and will be skipped; masked folds 1-4 will run.
for spec in "densenet121 masked" "efficientnetb0 raw" "efficientnetb0 masked"; do
    parts=(${=spec})
    echo "--- ${parts[1]} / ${parts[2]} : $(date) ---" | tee -a "$LOG"
    caffeinate -is python backbone_study.py "${parts[1]}" "${parts[2]}" 2>&1 | tee -a "$LOG"
done

echo "=== all backbone runs finished $(date) ===" | tee -a "$LOG"
