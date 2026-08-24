#!/bin/bash
# Chains Stages 1-7 for every model via SLURM job dependencies, so the whole
# pipeline can be launched with one command and runs unattended.
#
# Usage:
#   bash cluster/submit_all.sh [model_key ...]
#   (defaults to every model in config.py's MODEL_PATHS if none given)
#
# Prerequisite: cluster/00_setup_env.sh has been run once already.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
    activate_env
    mapfile -t MODELS < <(python3 -c "from flores_pipeline.config import MODEL_PATHS; print('\n'.join(MODEL_PATHS))")
fi

N_DEV="${N_DEV:-200}"
N_DEVTEST="${N_DEVTEST:-50}"

# GPU_COMMON overrides partition/mem for the GPU-bound stages (1, 2a, 2b, 6);
# CPU_COMMON only overrides account/mem so the CPU-partition default baked
# into 03_analysis.sbatch / 05_cross_comparison.sbatch is left alone.
GPU_COMMON=(--partition="$SLURM_PARTITION" --mem="$SLURM_MEM")
CPU_COMMON=(--mem="$SLURM_MEM")
if [ -n "$SLURM_ACCOUNT" ]; then
    GPU_COMMON+=(--account="$SLURM_ACCOUNT")
    CPU_COMMON+=(--account="$SLURM_ACCOUNT")
fi

echo "Submitting pipeline for models: ${MODELS[*]}"

FINAL_ABLATION_JOBS=()

for MODEL in "${MODELS[@]}"; do
    echo "=== $MODEL ==="

    JID_REF=$(sbatch --parsable "${GPU_COMMON[@]}" cluster/01_reference_vectors.sbatch "$MODEL" "$N_DEV")
    echo "  Stage 1 (reference vectors): job $JID_REF"

    JID_EXTRACT_T=$(sbatch --parsable --dependency=afterok:"$JID_REF" "${GPU_COMMON[@]}" cluster/02a_extract_translation.sbatch "$MODEL" "$N_DEVTEST")
    echo "  Stage 2a (translation extraction, array): job $JID_EXTRACT_T"

    JID_EXTRACT_R=$(sbatch --parsable --dependency=afterok:"$JID_REF" "${GPU_COMMON[@]}" cluster/02b_extract_reading.sbatch "$MODEL" "$N_DEVTEST")
    echo "  Stage 2b (reading extraction, array): job $JID_EXTRACT_R"

    JID_ANALYSIS_T=$(sbatch --parsable --dependency=afterok:"$JID_EXTRACT_T" "${CPU_COMMON[@]}" cluster/03_analysis.sbatch "$MODEL" translation)
    echo "  Stage 3-5 (translation analysis): job $JID_ANALYSIS_T"

    JID_ANALYSIS_R=$(sbatch --parsable --dependency=afterok:"$JID_EXTRACT_R" "${CPU_COMMON[@]}" cluster/03_analysis.sbatch "$MODEL" reading)
    echo "  Stage 3-5 (reading analysis): job $JID_ANALYSIS_R"

    JID_ABLATION=$(sbatch --parsable --dependency=afterok:"$JID_ANALYSIS_T":"$JID_ANALYSIS_R" "${GPU_COMMON[@]}" cluster/04_ablation.sbatch "$MODEL" "$N_DEVTEST")
    echo "  Stage 6 (ablation, array): job $JID_ABLATION"

    FINAL_ABLATION_JOBS+=("$JID_ABLATION")
done

DEP_STRING=$(IFS=:; echo "afterok:${FINAL_ABLATION_JOBS[*]}")
JID_CROSS=$(sbatch --parsable --dependency="$DEP_STRING" "${CPU_COMMON[@]}" cluster/05_cross_comparison.sbatch "${MODELS[@]}")
echo "Stage 7 (cross-comparison, all models): job $JID_CROSS"

echo "All jobs submitted. Monitor with: squeue -u \$USER"
