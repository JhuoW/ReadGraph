# Source before any ReGraph run:  source scripts/env.sh
#
# All large artifacts live on /mnt/ssd1 rather than the root disk, which sits at ~93% full
# and had its HuggingFace cache wiped on 2026-08-20 (taking the Llama backbone with it).
export HF_HOME=/mnt/ssd1/zhuowei/hf-cache          # models + hub cache
export TOKENIZERS_PARALLELISM=false                 # silences the dataloader-fork warning

# Preprocessed attribute-embedding caches are configured separately, in
# configs/default.yaml -> data.cache_dir (/mnt/ssd1/zhuowei/regraph-cache).

# One-time setup for the gated backbone (needs an account with Llama-3.1 access):
#   hf auth login
#   hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
# The --exclude skips the duplicate consolidated checkpoint and saves ~16 GB.
