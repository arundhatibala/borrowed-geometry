# Cluster scripts

SLURM job scripts for running the FLORES+ stepping-stone containment
pipeline (`../flores_pipeline`) on a cluster. If your cluster uses a
different scheduler (PBS, LSF, etc.), these need translating — the Python
calls inside each script are scheduler-agnostic and can be reused as-is.

**TODO before submitting anything**: `env.sh` has placeholder partition,
account, and module names — fill in real values for your cluster first,
same spirit as the `TODO`/`CONFIRM` markers in `../flores_pipeline/config.py`.

## One-time setup

```bash
bash cluster/00_setup_env.sh
```

## Run everything

```bash
bash cluster/submit_all.sh apertus gemma
```

Chains, per model: Stage 1 (reference vectors) -> Stage 2a/2b (extraction,
array jobs) -> Stage 3-5 (containment + permutation + layer selection) ->
Stage 6 (ablation, array job over the 4 family pairs) via SLURM
`--dependency=afterok`, then Stage 7 (cross-model comparison) once every
model's ablation jobs finish. Override `N_DEV` / `N_DEVTEST` env vars to
change sentence counts (defaults: 200 dev, 50 devtest).

## Run stages individually

```bash
sbatch cluster/01_reference_vectors.sbatch apertus 200
sbatch cluster/02a_extract_translation.sbatch apertus 50   # array 0-7, one per target lang
sbatch cluster/02b_extract_reading.sbatch apertus 50       # array 0-7
sbatch cluster/03_analysis.sbatch apertus translation
sbatch cluster/03_analysis.sbatch apertus reading
sbatch cluster/04_ablation.sbatch apertus 50                # array 0-3, one per family pair
sbatch cluster/05_cross_comparison.sbatch apertus gemma
```

Logs land in `cluster/logs/`.

## Notes

- Stage 6 (`04_ablation.sbatch`) reads the layer band Stage 5 selected for
  the **translation** condition and reuses it for both the translation and
  reading ablation runs on a given pair — see the assumption noted in
  `run_ablation_from_selection.py`. If a pair has no FDR-significant layer,
  that array task exits early (no ablation to run).
- `02a` (translation, generation-heavy) defaults to a 24h walltime; `02b`
  (reading, forward-pass only) defaults to 12h. Adjust in the `#SBATCH
  --time` line if your corpus size or hardware differs.
