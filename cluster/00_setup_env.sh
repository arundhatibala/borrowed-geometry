#!/bin/bash
# One-time environment setup. Run directly on a login node (not via sbatch):
#   bash cluster/00_setup_env.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# module load python/3.11   # TODO: uncomment/confirm once module names are known

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo "Environment ready at $VENV_DIR"
