"""Stage 6a — causal ablation, TRANSLATION condition.

At l* (and the null-clearing band), suppress the HR-axis projection
component from LR hidden states during generation (and mirror: suppress
LR-axis from HR), at each alpha in ALPHAS, forward + reversed. Re-generate
under ablation, score with COMET and LID exactly as Stage 2a.

Outcome per alpha:
  - DeltaCOMET(alpha) = comet_baseline - comet_alpha  (plain delta — see
    config.USE_NORMALIZED_DEGRADATION; the ceiling/floor-normalized
    alternative is NOT implemented per the top-of-spec assumption).
  - LID-collapse-rate(alpha): fraction of sentences where lid_lang !=
    target_language, reported as its own curve, never folded into
    DeltaCOMET (spec section 9 — avoids contaminating the quality metric
    with LID's own known unreliability on degenerate high-alpha text).
"""

import json
import os

from flores_pipeline.ablation.ablate import suppress_projection
from flores_pipeline.analysis.containment_metric import load_reference_vectors
from flores_pipeline.config import (
    ALPHAS,
    FLORES_DEVTEST_SPLIT,
    LANGUAGES,
    MAX_NEW_TOKENS,
    MODEL_PATHS,
    OUTPUTS_DIR,
    TRANSLATION_PROMPT_TEMPLATE,
    USE_NORMALIZED_DEGRADATION,
)
from flores_pipeline.data.load_flores import load_flores_split
from flores_pipeline.extraction.model_utils import generate, load_model
from flores_pipeline.scoring.comet_scoring import compute_comet
from flores_pipeline.scoring.lid_scoring import run_lid


def _run_translation_batch(loaded, examples, source_lang, target_lang, layer_band, direction, alpha):
    _, target_language_name = LANGUAGES[target_lang]
    results = []
    with suppress_projection(loaded.model, layer_band, direction, alpha):
        for ex in examples:
            source = ex.sentences[source_lang]
            gold_reference = ex.sentences[target_lang]
            prompt = TRANSLATION_PROMPT_TEMPLATE.format(
                target_language=target_language_name, sentence=source
            )
            hypothesis = generate(loaded, prompt, MAX_NEW_TOKENS)
            comet = compute_comet(source, gold_reference, hypothesis) if hypothesis.strip() else 0.0
            lid_lang, lid_conf = run_lid(hypothesis)
            results.append({"comet": comet, "lid_lang": lid_lang, "lid_confidence": lid_conf})
    return results


def run_dose_response_translation(
    model_key: str, lr_lang: str, hr_lang: str, layer_band, n_sentences: int = 50
):
    loaded = load_model(model_key, MODEL_PATHS[model_key])
    refs = load_reference_vectors(model_key)
    v_lr = refs[f"v_lang_{lr_lang}"]
    v_hr = refs[f"v_lang_{hr_lang}"]

    examples_lr = load_flores_split(FLORES_DEVTEST_SPLIT, ["en", lr_lang])[:n_sentences]
    examples_hr = load_flores_split(FLORES_DEVTEST_SPLIT, ["en", hr_lang])[:n_sentences]

    lr_target_code, _ = LANGUAGES[lr_lang]
    hr_target_code, _ = LANGUAGES[hr_lang]

    def collapse_rate(results, target_flores_code):
        if not results:
            return 0.0
        return sum(1 for r in results if r["lid_lang"] != target_flores_code) / len(results)

    def mean_comet(results):
        return sum(r["comet"] for r in results) / len(results) if results else 0.0

    baseline_lr = _run_translation_batch(loaded, examples_lr, "en", lr_lang, [], v_hr[0], 0.0)
    baseline_hr = _run_translation_batch(loaded, examples_hr, "en", hr_lang, [], v_lr[0], 0.0)
    comet_baseline_lr = mean_comet(baseline_lr)
    comet_baseline_hr = mean_comet(baseline_hr)

    curve = {"forward": [], "reversed": []}  # forward: suppress HR-axis from LR; reversed: suppress LR-axis from HR
    for l in layer_band:
        for alpha in ALPHAS:
            fwd_results = _run_translation_batch(
                loaded, examples_lr, "en", lr_lang, [l], v_hr[l], alpha
            )
            rev_results = _run_translation_batch(
                loaded, examples_hr, "en", hr_lang, [l], v_lr[l], alpha
            )

            comet_alpha_fwd = mean_comet(fwd_results)
            comet_alpha_rev = mean_comet(rev_results)

            delta_comet_fwd = comet_baseline_lr - comet_alpha_fwd
            delta_comet_rev = comet_baseline_hr - comet_alpha_rev

            curve["forward"].append(
                {
                    "layer": l,
                    "alpha": alpha,
                    "delta_comet": delta_comet_fwd,
                    "lid_collapse_rate": collapse_rate(fwd_results, lr_target_code),
                }
            )
            curve["reversed"].append(
                {
                    "layer": l,
                    "alpha": alpha,
                    "delta_comet": delta_comet_rev,
                    "lid_collapse_rate": collapse_rate(rev_results, hr_target_code),
                }
            )

    assert USE_NORMALIZED_DEGRADATION is False, (
        "ceiling/floor-normalized degradation is not implemented; "
        "flip config.USE_NORMALIZED_DEGRADATION only after implementing it"
    )

    out_dir = os.path.join(OUTPUTS_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_key}_translation_{lr_lang}-{hr_lang}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "comet_baseline_lr": comet_baseline_lr,
                "comet_baseline_hr": comet_baseline_hr,
                **curve,
            },
            f,
            indent=2,
        )
    return curve
