#!/usr/bin/env bash
# =============================================================================
#  ExplaGraphs -- 90.43 (3B) / 92.42 (8B)
#
#  Best among the baselines that remain after excluding GRAFF (concurrent work,
#  Findings of EACL 2026, code unreleased). ReGraph is the only frozen,
#  VECTOR-ONLY method in that table -- the graph is never serialized.
#
#  READ THIS BEFORE QUOTING THE NUMBER. GRAFF's own table is already 3B, so at
#  matched backbone ReGraph is 90.43 against LoRA/GRAG's 88.9: +1.53 at only
#  1.22 SE, NOT significant. The significant margin (+3.52, 3.13 SE) exists only
#  at 8B, i.e. with a LARGER backbone than the baselines it beats. Set
#  BACKBONE=() below for that number, and print the backbone alongside it.
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

CONFIG=configs/expla_graphs.yaml
CKPT=runs/expla_graphs/llama3b          # 3B; the 8B run is runs/expla_graphs/fix-seed0

STORE=$(python -c "
from regraph.utils.config import load_config
from regraph.data.preprocess import store_dir_for
print(store_dir_for(load_config('$CONFIG')))")
if [ ! -f "$STORE/graphs.pt" ]; then
  echo "[data] needs G-Retriever's train_dev.tsv under data/raw/expla_graphs/"
  python -m regraph.data.preprocess --config "$CONFIG"
fi

# --- STEP 2 (default): evaluate the released checkpoint ---------------------
python -m regraph.eval --config "$CONFIG" --ckpt "$CKPT/best.pt" --split test "${BACKBONE[@]}"
echo "expected: 90.43 (3B).  With BACKBONE=() and runs/expla_graphs/fix-seed0: 92.42 (8B)"

# --- STEP 3: retrain from scratch -------------------------------------------
# python -m regraph.train --config "$CONFIG" run_name=my3b "${BACKBONE[@]}"
# python -m regraph.eval  --config "$CONFIG" --ckpt runs/expla_graphs/my3b/best.pt --split test "${BACKBONE[@]}"
