#!/usr/bin/env bash
# =============================================================================
#  ogbn-arxiv -- Top-1 72.28, Top-3 93.23, Top-5 96.90 (3B)
#
#  Best result on GraphTranslator's exact 4,000-node evaluation subset
#  (their zero-shot ChatGLM2-6B: 28.48 / 37.62 / 39.87). Protocols differ --
#  GraphTranslator does not train on downstream labels, ReGraph trains on 20,000
#  nodes -- so this is a capability difference, not a like-for-like win.
#
#  Top-1 72.28 is from generation; Top-3/Top-5 come from a separate likelihood
#  ranking pass, whose own Top-1 is 72.10. Both are printed below.
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

CONFIG=configs/arxiv.yaml
CKPT=runs/arxiv/llama3b

# --- STEP 1: data -----------------------------------------------------------
STORE=$(python -c "
from regraph.utils.config import load_config
from regraph.data.preprocess import store_dir_for
print(store_dir_for(load_config('$CONFIG')))")
if [ ! -f "$STORE/graphs.pt" ]; then
  echo "[data] building store (OGB download + titleabs.tsv, one-off)"
  python -m regraph.data.preprocess_arxiv --config "$CONFIG"
fi

# --- STEP 2 (default): evaluate the released checkpoint ---------------------
python -m regraph.eval      --config "$CONFIG" --ckpt "$CKPT/best.pt" --split test "${BACKBONE[@]}"
python -m regraph.eval_rank --config "$CONFIG" --ckpt "$CKPT/best.pt" --split test "${BACKBONE[@]}"
echo "expected: accuracy 72.28 (generation) | top1 72.10 top3 93.23 top5 96.90 (ranking)"

# --- STEP 3: retrain from scratch -------------------------------------------
# python -m regraph.train     --config "$CONFIG" run_name=my3b "${BACKBONE[@]}"
# python -m regraph.eval      --config "$CONFIG" --ckpt runs/arxiv/my3b/best.pt --split test "${BACKBONE[@]}"
# python -m regraph.eval_rank --config "$CONFIG" --ckpt runs/arxiv/my3b/best.pt --split test "${BACKBONE[@]}"
