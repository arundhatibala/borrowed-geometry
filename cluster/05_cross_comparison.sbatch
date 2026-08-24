#!/bin/bash
#SBATCH --job-name=flores-cross-comparison
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=cluster/logs/cross_comparison_%j.out
#SBATCH --error=cluster/logs/cross_comparison_%j.err
#
# Stage 7 — cross-model / cross-condition comparison. Run once, after all
# models/conditions/pairs have finished Stages 3-6.
# Usage: sbatch cluster/05_cross_comparison.sbatch <model_key> [<model_key> ...]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
activate_env

if [ "$#" -lt 1 ]; then
    echo "usage: 05_cross_comparison.sbatch <model_key> [<model_key> ...]" >&2
    exit 1
fi
MODELS=("$@")
MODELS_PY=$(python3 -c "import sys; print(sys.argv[1:])" "${MODELS[@]}")

python3 -c "
from flores_pipeline.analysis.cross_comparison import run_cross_comparison_stage
from flores_pipeline.config import FAMILIES

pairs = [p for spec in FAMILIES.values() for p in spec['pairs']]
run_cross_comparison_stage($MODELS_PY, pairs)
"
