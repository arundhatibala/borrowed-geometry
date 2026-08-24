#!/bin/bash
#SBATCH --job-name=flores-refvec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=cluster/logs/refvec_%A_%a.out
#SBATCH --error=cluster/logs/refvec_%A_%a.err
#
# Stage 1 — reference vectors, one model per submission.
# Usage: sbatch cluster/01_reference_vectors.sbatch <model_key> [n_dev_sentences]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
activate_env

MODEL="${1:?usage: 01_reference_vectors.sbatch <model_key> [n_dev_sentences]}"
N_DEV="${2:-200}"

python3 -c "
from flores_pipeline.extraction.reference_vectors import build_reference_vectors
build_reference_vectors('$MODEL', n_dev_sentences=$N_DEV)
"
