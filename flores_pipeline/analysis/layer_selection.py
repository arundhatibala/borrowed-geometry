"""Stage 5 — layer selection.

    l* = argmax_l |Asymmetry(l)|  among layers where the FDR-corrected
         permutation test clears significance

Replaces the thesis's manual sigma threshold / l>=16 rule. If the
null-clearing band spans more than one contiguous layer, the full band is
kept for the ablation stage (Stage 6), not just the single argmax layer.
"""

import json
import os
from dataclasses import dataclass
from typing import List

from flores_pipeline.config import OUTPUTS_DIR


@dataclass
class LayerSelection:
    l_star: int
    band: List[int]  # contiguous significant layers containing l_star
    all_significant_layers: List[int]


def select_layers(permutation_result: dict) -> LayerSelection:
    """permutation_result: one entry from permutation_null's serialized output,
    i.e. {"observed": [...], "fdr": {"0": {"significant": bool, ...}, ...}}
    """
    observed = permutation_result["observed"]
    fdr = permutation_result["fdr"]

    significant_layers = sorted(
        int(l) for l, info in fdr.items() if info["significant"]
    )
    if not significant_layers:
        return LayerSelection(l_star=None, band=[], all_significant_layers=[])

    l_star = max(significant_layers, key=lambda l: abs(observed[l]))

    # contiguous band of significant layers containing l_star
    sig_set = set(significant_layers)
    band = [l_star]
    l = l_star - 1
    while l in sig_set:
        band.insert(0, l)
        l -= 1
    l = l_star + 1
    while l in sig_set:
        band.append(l)
        l += 1

    return LayerSelection(l_star=l_star, band=band, all_significant_layers=significant_layers)


def run_layer_selection_stage(model_key: str, condition: str):
    perm_path = os.path.join(
        OUTPUTS_DIR, "containment", f"{model_key}_{condition}_permutation.json"
    )
    with open(perm_path) as f:
        perm_results = json.load(f)

    selections = {}
    for key, res in perm_results.items():
        sel = select_layers(res)
        selections[key] = {
            "l_star": sel.l_star,
            "band": sel.band,
            "all_significant_layers": sel.all_significant_layers,
        }

    out_path = os.path.join(
        OUTPUTS_DIR, "containment", f"{model_key}_{condition}_layer_selection.json"
    )
    with open(out_path, "w") as f:
        json.dump(selections, f, indent=2)

    return selections


if __name__ == "__main__":
    import sys

    model_key = sys.argv[1] if len(sys.argv) > 1 else "apertus"
    condition = sys.argv[2] if len(sys.argv) > 2 else "translation"
    run_layer_selection_stage(model_key, condition)
