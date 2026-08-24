"""Stage 6b — causal ablation, READING condition.

At the same l* per pair, suppress the HR-axis / LR-axis projection
component during the teacher-forced forward pass over the real gold
sentence (no generation). No LID here: there's no generated output to
check, this is a forward pass over known-correct text.
"""

import json
import math
import os

from flores_pipeline.ablation.ablate import suppress_projection
from flores_pipeline.analysis.containment_metric import load_reference_vectors
from flores_pipeline.config import ALPHAS, FLORES_DEVTEST_SPLIT, MODEL_PATHS, OUTPUTS_DIR
from flores_pipeline.data.load_flores import load_flores_split
from flores_pipeline.extraction.model_utils import load_model, teacher_forced_mean_nll


def _mean_nll_batch(loaded, examples, lang, layer_band, direction, alpha):
    nlls = []
    with suppress_projection(loaded.model, layer_band, direction, alpha):
        for ex in examples:
            sentence = ex.sentences[lang]
            nlls.append(teacher_forced_mean_nll(loaded, sentence))
    return sum(nlls) / len(nlls) if nlls else 0.0


def run_dose_response_reading(
    model_key: str, lr_lang: str, hr_lang: str, layer_band, n_sentences: int = 50
):
    loaded = load_model(model_key, MODEL_PATHS[model_key])
    refs = load_reference_vectors(model_key)
    v_lr = refs[f"v_lang_{lr_lang}"]
    v_hr = refs[f"v_lang_{hr_lang}"]

    examples_lr = load_flores_split(FLORES_DEVTEST_SPLIT, [lr_lang])[:n_sentences]
    examples_hr = load_flores_split(FLORES_DEVTEST_SPLIT, [hr_lang])[:n_sentences]

    baseline_nll_lr = _mean_nll_batch(loaded, examples_lr, lr_lang, [], v_hr[0], 0.0)
    baseline_nll_hr = _mean_nll_batch(loaded, examples_hr, hr_lang, [], v_lr[0], 0.0)
    baseline_ppl_lr = math.exp(baseline_nll_lr)
    baseline_ppl_hr = math.exp(baseline_nll_hr)

    curve = {"forward": [], "reversed": []}
    for l in layer_band:
        for alpha in ALPHAS:
            nll_alpha_fwd = _mean_nll_batch(loaded, examples_lr, lr_lang, [l], v_hr[l], alpha)
            nll_alpha_rev = _mean_nll_batch(loaded, examples_hr, hr_lang, [l], v_lr[l], alpha)

            ppl_alpha_fwd = math.exp(nll_alpha_fwd)
            ppl_alpha_rev = math.exp(nll_alpha_rev)

            curve["forward"].append(
                {
                    "layer": l,
                    "alpha": alpha,
                    "delta_perplexity": ppl_alpha_fwd - baseline_ppl_lr,
                }
            )
            curve["reversed"].append(
                {
                    "layer": l,
                    "alpha": alpha,
                    "delta_perplexity": ppl_alpha_rev - baseline_ppl_hr,
                }
            )

    out_dir = os.path.join(OUTPUTS_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_key}_reading_{lr_lang}-{hr_lang}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "baseline_perplexity_lr": baseline_ppl_lr,
                "baseline_perplexity_hr": baseline_ppl_hr,
                **curve,
            },
            f,
            indent=2,
        )
    return curve
