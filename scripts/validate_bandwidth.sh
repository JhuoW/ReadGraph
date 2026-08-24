#!/bin/bash
# Validates the one open question from the 2026-08-20 tuning log: does N_B = 32 actually
# beat the protocol's N_B = 8, or was the apparent gain just more optimizer steps?
#
# Rows E (43.47) and G (45.13) were 10k-step runs measured against a 3k-step reference,
# because the matched 10k control (row I) died when the HF cache was deleted. This runs
# that control plus a wider arm, all at an identical 10k steps on identical data.
#
# Prereq:  source scripts/env.sh && hf auth login
# Usage:   bash scripts/validate_bandwidth.sh [GPU_INDEX]
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${1:-2}"

python3 - <<'PY'
from transformers import AutoConfig
AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print("backbone reachable, starting validation")
PY

SWEEP=/tmp/claude-1003/-home-zhuowei-My-Proj-ReadGraph/df8e5c9e-f370-4a76-92f8-b8b522db3e73/scratchpad/sweep.py
STEPS=10000

# I: the missing control -- protocol settings at the same horizon as E and G
python3 $SWEEP --config configs/scene_graphs.yaml --steps $STEPS --tag I-protocol-NB8-10k

# E64: is more channel width monotonically better?
python3 $SWEEP --config configs/scene_graphs.yaml --steps $STEPS --tag E64-NB64-zeros \
    model.num_query_tokens=64

echo
echo "Compare against the 2026-08-20 log in docs/experimental-protocol.md:"
echo "  E  N_B=32 zeros  43.47"
echo "  G  N_B=32 normal 45.13"
echo "If I-protocol lands near 43-45, channel width is NOT the driver and the tuned"
echo "configs should be reverted to N_B=8 rather than shipped."
