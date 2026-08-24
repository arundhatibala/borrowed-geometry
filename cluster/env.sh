#!/bin/bash
# Shared cluster settings, sourced by every job script in this directory.
#
# TODO / CONFIRM before submitting anything: partition, account, module
# names, and GPU type/count below are placeholders — fill in the real
# values for your cluster. This mirrors the config.py convention in the
# rest of the repo: flagged assumptions, not silent guesses.

set -euo pipefail

# --- SLURM defaults (override per-job with #SBATCH lines or sbatch flags) ---
export SLURM_PARTITION="${SLURM_PARTITION:-gpu}"          # TODO: confirm partition name
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"                  # TODO: fill in account/allocation
export SLURM_GRES="${SLURM_GRES:-gpu:1}"                   # TODO: confirm GPU request syntax
export SLURM_TIME_SHORT="${SLURM_TIME_SHORT:-02:00:00}"    # reference vectors, analysis stages
export SLURM_TIME_LONG="${SLURM_TIME_LONG:-24:00:00}"      # extraction, ablation (generation-heavy)
export SLURM_MEM="${SLURM_MEM:-64G}"

# --- Paths ---
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
export LOG_DIR="${LOG_DIR:-$REPO_ROOT/cluster/logs}"
mkdir -p "$LOG_DIR"

# --- Modules — TODO: confirm module names/versions available on your cluster ---
# module load python/3.11
# module load cuda/12.1

activate_env() {
    source "$VENV_DIR/bin/activate"
    cd "$REPO_ROOT"
}
