#!/usr/bin/env bash
# =============================================================================
#  StructuralAnomaly -- 99.15 (3B) / 99.05 (8B), chance 20.00
#
#  A benchmark INTRODUCED BY THIS REPO -- no external competitor, so this is a
#  capability result rather than a leaderboard win. It is nonetheless the only
#  result here validated by a control that could have failed, and the only one
#  testing ReGraph.md section 2.1's ANCHOR-FREE path (no node is named; every
#  node carries ROLE_NONE), which nothing else in the repo evaluates.
#
#  THE CONTROL IS NOT OPTIONAL. It removes the density contrast and randomises
#  the label, making the task unanswerable, so it MUST score the analytic
#  chance of 20.00. Both arms are evaluated below; 99.15 alone is not an
#  interpretable result.
#
#  Known limits: the structural cue is DEGREE (a first-order local quantity),
#  and at density_ratio=5.0 the main arm is saturated -- the informative
#  quantity is the accuracy-vs-density_ratio curve, not this point.
#
#  DEFAULT ACTION: evaluate the released checkpoint (minutes, no training).
#  TO RETRAIN:     comment out STEP 2, uncomment STEP 3.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh

# Backbone. Default is Llama-3.2-3B-Instruct. There is no separate 3B config in
# this repo -- every 3B number is the base config plus exactly these three
# overrides. For the 8B backbone, set BACKBONE=() instead.
BACKBONE=(
  llm.name=meta-llama/Llama-3.2-3B-Instruct
  llm.d_llm=3072
  llm.num_layers=28
)

MAIN_CFG=configs/synth_anomaly.yaml
CTRL_CFG=configs/synth_anomaly_control.yaml
MAIN_CKPT=runs/synth_anomaly/llama3b             # 8B: runs/synth_anomaly/seed0
CTRL_CKPT=runs/synth_anomaly_control/llama3b     # 8B: runs/synth_anomaly_control/seed0

# --- STEP 1: data (synthetic; no download, seeded, reproducible bit-for-bit) -
for cfg in "$MAIN_CFG" "$CTRL_CFG"; do
  STORE=$(python -c "
from regraph.utils.config import load_config
from regraph.data.preprocess import store_dir_for
print(store_dir_for(load_config('$cfg')))")
  [ -f "$STORE/graphs.pt" ] || python -m regraph.data.preprocess_synth --config "$cfg"
done

# --- STEP 2 (default): evaluate both arms -----------------------------------
python -m regraph.eval --config "$MAIN_CFG" --ckpt "$MAIN_CKPT/best.pt" --split test "${BACKBONE[@]}"
python -m regraph.eval --config "$CTRL_CFG" --ckpt "$CTRL_CKPT/best.pt" --split test "${BACKBONE[@]}"
echo "expected (3B): main 99.15 | control ~20.3 | analytic chance 20.00"
echo "expected (8B): main 99.05 | control ~19.9"
echo "the control is only required to be indistinguishable from chance; it is not"
echo "bit-reproducible across runs (no signal -> near-tied logits), spread 19.45-20.55."

# --- STEP 3: retrain from scratch (~15 min per arm) -------------------------
# python -m regraph.train --config "$MAIN_CFG" run_name=my3b "${BACKBONE[@]}"
# python -m regraph.eval  --config "$MAIN_CFG" --ckpt runs/synth_anomaly/my3b/best.pt --split test "${BACKBONE[@]}"
# python -m regraph.train --config "$CTRL_CFG" run_name=my3b "${BACKBONE[@]}"
# python -m regraph.eval  --config "$CTRL_CFG" --ckpt runs/synth_anomaly_control/my3b/best.pt --split test "${BACKBONE[@]}"
