"""Stage 4 — permutation null.

For each (model, condition, pair, layer):
  - Permutation 1 (shuffle): randomly re-pair sentence indices between the
    LR and HR hidden-state sets N_PERMUTATIONS times, recompute Asymmetry(l)
    on each shuffle -> null distribution.
  - Permutation 2 (cross-family control): compare observed Asymmetry(l)
    against the distribution of Asymmetry(l) from the genuine cross-family
    control pairs.

Reports percentile + z-score against the shuffle null, then applies FDR
correction (config.FDR_METHOD) across the full family of tests
(model x condition x pair x layer) before anything is declared significant.
"""

import json
import os

import numpy as np
from statsmodels.stats.multitest import multipletests

from flores_pipeline.analysis.containment_metric import (
    _containment_per_layer,
    _drift,
    load_hidden_states,
    load_reference_vectors,
)
from flores_pipeline.config import FDR_METHOD, N_PERMUTATIONS, OUTPUTS_DIR


def _shuffle_null_asymmetry(r_lr, r_hr, v_lr, v_hr, n_permutations, rng):
    """Shuffle null for Asymmetry(l), all layers at once.

    Returns array [n_permutations, n_layers].
    """
    n_lr = r_lr.shape[0]
    n_hr = r_hr.shape[0]
    n_layers = r_lr.shape[1]
    null = np.zeros((n_permutations, n_layers))

    pooled = np.concatenate([r_lr, r_hr], axis=0)  # [n_lr+n_hr, n_layers, dim]
    n_total = pooled.shape[0]

    for p in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        shuf_lr = pooled[perm_idx[:n_lr]]
        shuf_hr = pooled[perm_idx[n_lr:n_lr + n_hr]]
        c_lr_to_hr = _containment_per_layer(shuf_lr, v_hr)
        c_hr_to_lr = _containment_per_layer(shuf_hr, v_lr)
        null[p] = c_lr_to_hr - c_hr_to_lr

    return null


def permutation_test_for_pair(
    model_key: str, condition: str, lr_lang: str, hr_lang: str, refs=None, seed: int = 0
):
    refs = refs or load_reference_vectors(model_key)
    en_ref = refs["en_ref"]
    v_lr = refs[f"v_lang_{lr_lang}"]
    v_hr = refs[f"v_lang_{hr_lang}"]

    h_lr = load_hidden_states(model_key, condition, lr_lang)
    h_hr = load_hidden_states(model_key, condition, hr_lang)
    r_lr = _drift(h_lr, en_ref)
    r_hr = _drift(h_hr, en_ref)

    observed = _containment_per_layer(r_lr, v_hr) - _containment_per_layer(r_hr, v_lr)

    rng = np.random.default_rng(seed)
    null = _shuffle_null_asymmetry(r_lr, r_hr, v_lr, v_hr, N_PERMUTATIONS, rng)

    n_layers = observed.shape[0]
    percentile = np.zeros(n_layers)
    z_score = np.zeros(n_layers)
    p_value = np.zeros(n_layers)
    for l in range(n_layers):
        null_l = null[:, l]
        percentile[l] = float(np.mean(null_l < observed[l])) * 100
        mu, sigma = np.mean(null_l), np.std(null_l)
        z_score[l] = (observed[l] - mu) / sigma if sigma > 0 else np.nan
        # two-sided permutation p-value
        p_value[l] = float(np.mean(np.abs(null_l) >= np.abs(observed[l])))

    return {
        "observed": observed,
        "percentile": percentile,
        "z_score": z_score,
        "p_value": p_value,
    }


def run_permutation_stage(model_key: str, condition: str, families):
    """families: dict of family_name -> {"pairs": [(lr, hr), ...]}"""
    refs = load_reference_vectors(model_key)

    raw_results = {}
    all_p_values = []
    all_keys = []
    for family, spec in families.items():
        for lr_lang, hr_lang in spec["pairs"]:
            key = f"{model_key}:{condition}:{family}:{lr_lang}-{hr_lang}"
            res = permutation_test_for_pair(model_key, condition, lr_lang, hr_lang, refs)
            raw_results[key] = res
            n_layers = res["observed"].shape[0]
            for l in range(n_layers):
                all_keys.append((key, l))
                all_p_values.append(res["p_value"][l])

    method = "fdr_bh" if FDR_METHOD == "benjamini-hochberg" else "bonferroni"
    reject, p_corrected, _, _ = multipletests(all_p_values, alpha=0.05, method=method)

    significant = {}
    for (key, l), sig, p_adj in zip(all_keys, reject, p_corrected):
        significant.setdefault(key, {})[l] = {"significant": bool(sig), "p_adjusted": float(p_adj)}

    out_dir = os.path.join(OUTPUTS_DIR, "containment")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_key}_{condition}_permutation.json")
    serializable = {
        key: {
            "observed": res["observed"].tolist(),
            "percentile": res["percentile"].tolist(),
            "z_score": res["z_score"].tolist(),
            "p_value": res["p_value"].tolist(),
            "fdr": significant.get(key, {}),
        }
        for key, res in raw_results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return serializable


if __name__ == "__main__":
    import sys

    from flores_pipeline.config import FAMILIES

    model_key = sys.argv[1] if len(sys.argv) > 1 else "apertus"
    condition = sys.argv[2] if len(sys.argv) > 2 else "translation"
    run_permutation_stage(model_key, condition, FAMILIES)
