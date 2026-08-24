#!/bin/bash
#SBATCH --job-name=flores-analysis
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/analysis_%A_%a.out
#SBATCH --error=cluster/logs/analysis_%A_%a.err
#
# Stages 3-5 — containment metric, permutation null (N_PERMUTATIONS shuffles,
# CPU-bound, no GPU needed), and layer selection. One run per (model,
# condition). Requires Stage 1 + Stage 2a/2b outputs for this model.
# Usage: sbatch cluster/03_analysis.sbatch <model_key> <condition>

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
activate_env

MODEL="${1:?usage: 03_analysis.sbatch <model_key> <condition>}"
CONDITION="${2:?usage: 03_analysis.sbatch <model_key> <condition>}"

python3 -c "
from flores_pipeline.analysis.containment_metric import run_containment_stage
from flores_pipeline.analysis.permutation_null import run_permutation_stage
from flores_pipeline.analysis.layer_selection import run_layer_selection_stage
from flores_pipeline.config import FAMILIES

print('Stage 3: containment')
run_containment_stage('$MODEL', '$CONDITION')
print('Stage 4: permutation null')
run_permutation_stage('$MODEL', '$CONDITION', FAMILIES)
print('Stage 5: layer selection')
run_layer_selection_stage('$MODEL', '$CONDITION')
"
