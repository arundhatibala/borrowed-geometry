#!/bin/bash
#SBATCH --job-name=flores-extract-translation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=cluster/logs/extract_translation_%A_%a.out
#SBATCH --error=cluster/logs/extract_translation_%A_%a.err
#
# Stage 2a — translation-condition corpus extraction.
# One array task per target language (index into config.TARGET_LANGS).
# Usage: sbatch cluster/02a_extract_translation.sbatch <model_key> [n_sentences]
#
# Requires Stage 1 (reference vectors) to have completed for this model —
# not a hard dependency of extraction itself, but containment_metric.py
# (Stage 3) will fail without it, so run 01 first or chain via submit_all.sh.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
activate_env

MODEL="${1:?usage: 02a_extract_translation.sbatch <model_key> [n_sentences]}"
N_SENTENCES="${2:-None}"

python3 -c "
from flores_pipeline.config import TARGET_LANGS
from flores_pipeline.extraction.corpus_translation import extract_translation_condition

lang = TARGET_LANGS[$SLURM_ARRAY_TASK_ID]
n_sentences = $N_SENTENCES
print(f'extracting translation condition: model=$MODEL lang={lang} n_sentences={n_sentences}')
extract_translation_condition('$MODEL', lang, n_sentences=n_sentences)
"
