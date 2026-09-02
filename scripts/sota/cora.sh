#!/usr/bin/env bash
# =============================================================================
#  Cora -- 88.99 +/- 1.26 over 3 seeds (3B)
#
#  Statistically matches LLaGA-HO-7B (89.22) and SAGN (89.19) at well under half
#  the LLM size, and beats LLaGA-ND-7B (88.86) and the other GNN baselines.
#  The claim rests on the 3-seed mean, so all three checkpoints are evaluated.
#
#  DEFAULT ACTION: evaluate the released checkpoint (minutes, no training).
#  TO RETRAIN:     comment out STEP 2, uncomment STEP 3.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh

# Backbone. Llama-3.2-3B-Instruct is now the repo-wide default (configs/default.yaml),
# so no override is needed for it and BACKBONE=() below means 3B, not 8B.
# For the 8B backbone, uncomment BACKBONE_8B and use it instead -- the three
# fields move together, and an 8B checkpoint will not load under a 3B config.
BACKBONE=()
# BACKBONE=(llm.name=meta-llama/Llama-3.1-8B-Instruct llm.d_llm=4096 llm.num_layers=32)

CONFIG=configs/cora.yaml
SEEDS=(llama3b llama3b-s1 llama3b-s2)

# --- STEP 1: data -----------------------------------------------------------
STORE=$(python -c "
from regraph.utils.config import load_config
from regraph.data.preprocess import store_dir_for
print(store_dir_for(load_config('$CONFIG')))")
if [ ! -f "$STORE/graphs.pt" ]; then
  echo "[data] building store (TAPE text + LINQS graph; see scripts/DATASETS.md)"
  python -m regraph.data.preprocess_tag --config "$CONFIG"
fi

# --- STEP 2 (default): evaluate the three released checkpoints --------------
for s in "${SEEDS[@]}"; do
  python -m regraph.eval --config "$CONFIG" --ckpt "runs/cora/$s/best.pt" --split test "${BACKBONE[@]}"
done
python - <<'PY'
import json, statistics as st
a=[json.load(open(f"runs/cora/{s}/metrics_test.json"))["accuracy"]*100
   for s in ("llama3b","llama3b-s1","llama3b-s2")]
print(f"3-seed mean {st.mean(a):.2f} +/- {st.stdev(a):.2f}   (expected 88.99 +/- 1.26)")
PY

# --- STEP 3: retrain from scratch -------------------------------------------
# for i in 0 1 2; do
#   python -m regraph.train --config "$CONFIG" run_name=my3b-s$i seed=$i "${BACKBONE[@]}"
#   python -m regraph.eval  --config "$CONFIG" --ckpt runs/cora/my3b-s$i/best.pt --split test "${BACKBONE[@]}"
# done
