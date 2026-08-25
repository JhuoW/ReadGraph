#!/usr/bin/env bash
# =============================================================================
#  SceneGraphs -- 89.41 (8B, + token channel)
#
#  READ THIS BEFORE QUOTING THE NUMBER -- SceneGraphs is NOT a matched-backbone
#  SOTA result, despite appearing in the six-dataset list.
#
#  At 8B this configuration scores 89.41, ahead of every non-GRAFF baseline.
#  But GRAFF's table is Llama-3.2-3B throughout, and the matched 3B run scores
#  83.54: ahead of G-Retriever (82.3) but 1.76 BEHIND LoRA (85.3) at 6.7 SE,
#  i.e. 3rd of 7 rather than 1st of 8. The 8B advantage comes from doubling the
#  backbone, not from the method.
#
#  It also SERIALIZES THE GRAPH INTO THE PROMPT, violating ReGraph.md section
#  3.2 -- the augmented setup, not ReGraph as specified, whose SceneGraphs
#  score is 51.83 (8B) / 53.24 (3B).
#
#  The 8B->3B drop (-5.87) is the most useful thing this run produced: the same
#  dataset is backbone-INsensitive without the token channel (+1.41) and
#  strongly sensitive with it, so serializing the graph moves the work into the
#  language model. Both arms are evaluated below.
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
  echo "        8B: runs/scene_graphs/dual-full   3B: runs/scene_graphs/dual-3b"
  exit 1
fi

# --- STEP 2 (default): evaluate the released checkpoint ---------------------
python -m regraph.eval --config "$CONFIG" --ckpt "$CKPT/best.pt" --split test "${BACKBONE[@]}"
echo "expected: 89.41 (8B, dual-full) / 83.54 (3B, dual-3b), all 20,025 test examples"

# --- STEP 3: retrain from scratch (~45 h at 8B, ~19 h at 3B) ----------------
# python -m regraph.train --config "$CONFIG" run_name=mydual \
#     train.batch_size=2 train.grad_accum=2 "${BACKBONE[@]}"
# python -m regraph.eval  --config "$CONFIG" --ckpt runs/scene_graphs/mydual/best.pt \
#     --split test "${BACKBONE[@]}"
