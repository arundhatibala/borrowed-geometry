"""Helper invoked by cluster/04_ablation.sbatch.

Reads the layer band Stage 5 selected for a (model, pair) and runs both
Stage 6a (translation) and Stage 6b (reading) ablation dose-response over
that band.

ASSUMPTION (not fully specified in the spec): Stage 5 is run independently
per condition (Stage 3/4/5 loop over both translation and reading), so each
condition could in principle select a different l*/band for the same pair.
Spec section 10 says Stage 6b ablates "at the same l* per pair" as 6a —
read here as: reuse the TRANSLATION condition's selected band for both 6a
and 6b, so a single band is shared across both ablation runs for a given
pair. CONFIRM if the reading condition's own band should be used instead.
"""

import json
import os
import sys

from flores_pipeline.ablation.dose_response_reading import run_dose_response_reading
from flores_pipeline.ablation.dose_response_translation import run_dose_response_translation
from flores_pipeline.config import FAMILIES, OUTPUTS_DIR


def _pair_key(family, lr_lang, hr_lang):
    return f"{family}:{lr_lang}-{hr_lang}"


def main():
    if len(sys.argv) < 4:
        print("usage: run_ablation_from_selection.py <model_key> <lr_lang> <hr_lang> [n_sentences]")
        sys.exit(1)

    model_key, lr_lang, hr_lang = sys.argv[1], sys.argv[2], sys.argv[3]
    n_sentences = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    family = next(
        (f for f, spec in FAMILIES.items() if (lr_lang, hr_lang) in spec["pairs"]), None
    )
    if family is None:
        print(f"pair ({lr_lang}, {hr_lang}) not found in config.FAMILIES", file=sys.stderr)
        sys.exit(1)

    selection_path = os.path.join(
        OUTPUTS_DIR, "containment", f"{model_key}_translation_layer_selection.json"
    )
    with open(selection_path) as f:
        selections = json.load(f)

    key = _pair_key(family, lr_lang, hr_lang)
    if key not in selections or not selections[key]["band"]:
        print(f"no significant layer band found for {key}; skipping ablation", file=sys.stderr)
        sys.exit(1)

    band = selections[key]["band"]
    print(f"model={model_key} pair={lr_lang}-{hr_lang} band={band}")

    print("Stage 6a: translation ablation dose-response")
    run_dose_response_translation(model_key, lr_lang, hr_lang, band, n_sentences=n_sentences)

    print("Stage 6b: reading ablation dose-response")
    run_dose_response_reading(model_key, lr_lang, hr_lang, band, n_sentences=n_sentences)


if __name__ == "__main__":
    main()
