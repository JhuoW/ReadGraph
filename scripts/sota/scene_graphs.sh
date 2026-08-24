#!/usr/bin/env bash
# =============================================================================
#  SceneGraphs -- 89.41 (8B, + token channel)
#
#  Best non-GRAFF result: clears G-Retriever w/ LoRA (86.83) by +2.58 (11.9 SE)
#  with the LLM frozen against their tuned one. GRAFF (90.2) is still 0.79 ahead.
#
#  TWO THINGS TO KNOW. (1) This configuration SERIALIZES THE GRAPH INTO THE
#  PROMPT and therefore violates ReGraph.md section 3.2 -- it is the augmented
#  setup, not ReGraph as specified, whose SceneGraphs score is 51.83 (8B) /
#  53.24 (3B). (2) 89.41 is an 8B number; the matched-backbone 3B run is still
#  training at the time of writing, so this script defaults to 8B and falls
#  back with a message if the 3B checkpoint is absent.
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

# This dataset's headline number is 8B, so the default here is 8B, not 3B.
BACKBONE=()
CONFIG=configs/scene_graphs_dual.yaml
CKPT=runs/scene_graphs/dual-full

# To use the matched-backbone 3B run instead, uncomment both lines:
# BACKBONE=(llm.name=meta-llama/Llama-3.2-3B-Instruct llm.d_llm=3072 llm.num_layers=28)
# CKPT=runs/scene_graphs/dual-3b

STORE=$(python -c "
from regraph.utils.config import load_config
from regraph.data.preprocess import store_dir_for
print(store_dir_for(load_config('$CONFIG')))")
if [ ! -f "$STORE/graphs.pt" ]; then
  echo "[data] needs G-Retriever's sceneGraphs.zip + questions.csv under data/raw/scene_graphs/"
  python -m regraph.data.preprocess --config "$CONFIG"
fi

if [ ! -f "$CKPT/best.pt" ]; then
  echo "[error] $CKPT/best.pt not found."
  echo "        The 3B run (dual-3b) takes ~19 h; use CKPT=runs/scene_graphs/dual-full for 8B."
  exit 1
fi

# --- STEP 2 (default): evaluate the released checkpoint ---------------------
python -m regraph.eval --config "$CONFIG" --ckpt "$CKPT/best.pt" --split test "${BACKBONE[@]}"
echo "expected: 89.41 (8B, dual-full) on all 20,025 test examples"

# --- STEP 3: retrain from scratch (~45 h at 8B, ~19 h at 3B) ----------------
# python -m regraph.train --config "$CONFIG" run_name=mydual \
#     train.batch_size=2 train.grad_accum=2 "${BACKBONE[@]}"
# python -m regraph.eval  --config "$CONFIG" --ckpt runs/scene_graphs/mydual/best.pt \
#     --split test "${BACKBONE[@]}"
