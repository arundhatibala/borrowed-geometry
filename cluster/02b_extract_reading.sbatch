#!/bin/bash
#SBATCH --job-name=flores-extract-reading
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=cluster/logs/extract_reading_%A_%a.out
#SBATCH --error=cluster/logs/extract_reading_%A_%a.err
#
# Stage 2b — reading-condition corpus extraction (forward pass only, no
# generation, so this is much cheaper than 02a — shorter default walltime).
# One array task per target language (index into config.TARGET_LANGS).
# Usage: sbatch cluster/02b_extract_reading.sbatch <model_key> [n_sentences]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
activate_env

MODEL="${1:?usage: 02b_extract_reading.sbatch <model_key> [n_sentences]}"
N_SENTENCES="${2:-None}"

python3 -c "
from flores_pipeline.config import TARGET_LANGS
from flores_pipeline.extraction.corpus_reading import extract_reading_condition

lang = TARGET_LANGS[$SLURM_ARRAY_TASK_ID]
n_sentences = $N_SENTENCES
print(f'extracting reading condition: model=$MODEL lang={lang} n_sentences={n_sentences}')
extract_reading_condition('$MODEL', lang, n_sentences=n_sentences)
"
