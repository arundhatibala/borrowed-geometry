"""Stage 7 — cross-condition / cross-model comparison.

- Per pair, per model: does a monotonic dose-response exist in both
  DeltaCOMET (translation) and Delta-perplexity (reading)? Reports
  presence/absence only, not magnitude comparison (plain-delta tradeoff,
  see config.USE_NORMALIZED_DEGRADATION).
- Compares Asymmetry(l) significance (Stage 4) across models.
- Compares across the 4 language-family pairs vs. cross-family controls.
"""

import json
import os
from typing import List

from flores_pipeline.config import FAMILIES, OUTPUTS_DIR


def _is_monotonic(values: List[float], tol: float = 1e-9) -> bool:
    non_decreasing = all(b >= a - tol for a, b in zip(values, values[1:]))
    non_increasing = all(b <= a + tol for a, b in zip(values, values[1:]))
    return non_decreasing or non_increasing


def check_dose_response_agreement(model_key: str, lr_lang: str, hr_lang: str):
    ablation_dir = os.path.join(OUTPUTS_DIR, "ablation")

    with open(os.path.join(ablation_dir, f"{model_key}_translation_{lr_lang}-{hr_lang}.json")) as f:
        translation_curve = json.load(f)
    with open(os.path.join(ablation_dir, f"{model_key}_reading_{lr_lang}-{hr_lang}.json")) as f:
        reading_curve = json.load(f)

    def by_layer(curve, direction, field):
        grouped = {}
        for point in curve[direction]:
            grouped.setdefault(point["layer"], []).append(point[field])
        return grouped

    result = {}
    for direction in ("forward", "reversed"):
        comet_by_layer = by_layer(translation_curve, direction, "delta_comet")
        ppl_by_layer = by_layer(reading_curve, direction, "delta_perplexity")
        layers = sorted(set(comet_by_layer) & set(ppl_by_layer))
        per_layer = {}
        for l in layers:
            per_layer[l] = {
                "translation_monotonic": _is_monotonic(comet_by_layer[l]),
                "reading_monotonic": _is_monotonic(ppl_by_layer[l]),
            }
        result[direction] = per_layer

    return result


def compare_significance_across_models(model_keys: List[str], condition: str):
    """Compares which (family, pair, layer) cells clear FDR significance
    across models — does the pattern replicate across architectures?
    """
    per_model = {}
    for model_key in model_keys:
        path = os.path.join(OUTPUTS_DIR, "containment", f"{model_key}_{condition}_permutation.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            per_model[model_key] = json.load(f)

    if len(per_model) < 2:
        return {"note": "need >=2 models with permutation results to compare", "models_found": list(per_model)}

    all_keys = set()
    for res in per_model.values():
        all_keys.update(res.keys())

    comparison = {}
    for key in all_keys:
        comparison[key] = {}
        for model_key, res in per_model.items():
            if key not in res:
                continue
            fdr = res[key]["fdr"]
            comparison[key][model_key] = {
                l: info["significant"] for l, info in fdr.items()
            }
    return comparison


def compare_family_vs_cross_family(model_key: str, condition: str):
    containment_dir = os.path.join(OUTPUTS_DIR, "containment")
    with open(os.path.join(containment_dir, f"{model_key}_{condition}_cross_family_controls.json")) as f:
        control_results = json.load(f)

    family_results = {}
    for family in FAMILIES:
        path = os.path.join(containment_dir, f"{model_key}_{condition}_{family}.json")
        if os.path.exists(path):
            with open(path) as f:
                family_results[family] = json.load(f)

    return {"within_family": family_results, "cross_family_controls": control_results}


def run_cross_comparison_stage(model_keys: List[str], pairs, condition_for_significance="translation"):
    out = {
        "dose_response_agreement": {},
        "significance_across_models": compare_significance_across_models(
            model_keys, condition_for_significance
        ),
        "family_vs_cross_family": {},
    }
    for model_key in model_keys:
        for lr_lang, hr_lang in pairs:
            key = f"{model_key}:{lr_lang}-{hr_lang}"
            try:
                out["dose_response_agreement"][key] = check_dose_response_agreement(
                    model_key, lr_lang, hr_lang
                )
            except FileNotFoundError:
                continue
        out["family_vs_cross_family"][model_key] = compare_family_vs_cross_family(
            model_key, condition_for_significance
        )

    out_dir = os.path.join(OUTPUTS_DIR, "containment")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cross_comparison.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out
