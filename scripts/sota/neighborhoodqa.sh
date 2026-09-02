#!/usr/bin/env bash
# =============================================================================
#  NeighborhoodQA -- 83.83 set-F1 (3B) / 83.87 (8B)
#
#  A benchmark INTRODUCED BY THIS REPO, so there is no external competitor and
#  no like-for-like SOTA comparison. Report it as a capability result, not a
#  leaderboard win. What makes the number meaningful is the control ladder:
#      6.98   no graph at all (num_rounds=0)
#     63.81   analytic shortcut: answer the centre paper's own area
#     72.65   0-hop control: centre node only
#     83.87   ReGraph, full 2-hop
#  The 0-hop control is the load-bearing one: same store, ego-subgraphs
#  truncated to the centre node, examples.json byte-for-byte identical.
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

CONFIG=configs/arxiv_nbrqa.yaml
CKPT=runs/arxiv_nbrqa/llama3b            # 3B; the 8B run is runs/arxiv_nbrqa/seed0

STORE=$(python -c "
from regraph.utils.config import load_config
from regraph.data.preprocess import store_dir_for
print(store_dir_for(load_config('$CONFIG')))")
if [ ! -f "$STORE/graphs.pt" ]; then
  echo "[data] generated from the ogbn-arxiv raw files, no extra download"
  python -m regraph.data.preprocess_nbrqa --config "$CONFIG"
fi

# --- STEP 2 (default): evaluate the released checkpoint ---------------------
python -m regraph.eval --config "$CONFIG" --ckpt "$CKPT/best.pt" --split test "${BACKBONE[@]}"
echo "expected: 83.83 set-F1 (3B).  8B is 83.87 with runs/arxiv_nbrqa/seed0"

# The 0-hop control that makes the number interpretable (8B checkpoint):
# python -m regraph.eval --config configs/arxiv_nbrqa_zerohop.yaml \
#     --ckpt runs/arxiv_nbrqa_zerohop/seed0/best.pt --split test    # expect 72.65

# --- STEP 3: retrain from scratch -------------------------------------------
# python -m regraph.train --config "$CONFIG" run_name=my3b "${BACKBONE[@]}"
# python -m regraph.eval  --config "$CONFIG" --ckpt runs/arxiv_nbrqa/my3b/best.pt --split test "${BACKBONE[@]}"
