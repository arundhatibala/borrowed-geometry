#!/bin/bash
#SBATCH --job-name=flores-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --output=cluster/logs/ablation_%A_%a.out
#SBATCH --error=cluster/logs/ablation_%A_%a.err
#
# Stage 6a+6b — causal ablation dose-response, one array task per family
# pair (index into config.FAMILIES, in insertion order: iberian, slavic,
# germanic_west, germanic_north). Requires Stage 5 (layer selection) output
# for the translation condition of this model.
# Usage: sbatch cluster/04_ablation.sbatch <model_key> [n_sentences]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
activate_env

MODEL="${1:?usage: 04_ablation.sbatch <model_key> [n_sentences]}"
N_SENTENCES="${2:-50}"

PAIR=$(python3 -c "
from flores_pipeline.config import FAMILIES
pairs = [p for spec in FAMILIES.values() for p in spec['pairs']]
lr, hr = pairs[$SLURM_ARRAY_TASK_ID]
print(f'{lr} {hr}')
")
read -r LR_LANG HR_LANG <<< "$PAIR"

python3 cluster/run_ablation_from_selection.py "$MODEL" "$LR_LANG" "$HR_LANG" "$N_SENTENCES"
