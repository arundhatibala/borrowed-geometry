"""Stage 2a — corpus extraction, TRANSLATION condition.

For each FLORES+ devtest sentence: generate a translation, teacher-force the
prompt+generation to extract hidden states over the generated span only,
score with reference-based COMET, and run LID on the hypothesis.

No good/bad group assignment happens here — that's a downstream step once
the full corpus's COMET distribution is known.
"""

import os

import numpy as np

from flores_pipeline.config import (
    FLORES_DEVTEST_SPLIT,
    LANGUAGES,
    MAX_NEW_TOKENS,
    MODEL_PATHS,
    OUTPUTS_DIR,
    TARGET_LANGS,
    TRANSLATION_PROMPT_TEMPLATE,
)
from flores_pipeline.data.load_flores import load_flores_split
from flores_pipeline.extraction.model_utils import (
    extract_hidden_states_for_span,
    generate,
    load_model,
)
from flores_pipeline.scoring.comet_scoring import compute_comet
from flores_pipeline.scoring.lid_scoring import run_lid


def extract_translation_condition(model_key: str, lang: str, n_sentences: int = None):
    loaded = load_model(model_key, MODEL_PATHS[model_key])
    _, target_language_name = LANGUAGES[lang]

    examples = load_flores_split(FLORES_DEVTEST_SPLIT, ["en", lang])
    if n_sentences is not None:
        examples = examples[:n_sentences]

    out_dir = os.path.join(
        OUTPUTS_DIR, f"hidden_states_{model_key}_translation", f"{lang}_{model_key}"
    )
    os.makedirs(out_dir, exist_ok=True)

    for ex in examples:
        source = ex.sentences["en"]
        gold_reference = ex.sentences[lang]
        prompt = TRANSLATION_PROMPT_TEMPLATE.format(
            target_language=target_language_name, sentence=source
        )

        hypothesis = generate(loaded, prompt, MAX_NEW_TOKENS)
        if not hypothesis.strip():
            continue

        full_text = prompt + hypothesis
        translation_hidden = extract_hidden_states_for_span(loaded, full_text, hypothesis)

        comet = compute_comet(source, gold_reference, hypothesis)
        lid_lang, lid_confidence = run_lid(hypothesis)

        np.savez(
            os.path.join(out_dir, f"ex_{ex.idx:04d}.npz"),
            translation_hidden=translation_hidden,
            comet=np.array(comet),
            lid_lang=np.array(lid_lang),
            lid_confidence=np.array(lid_confidence),
            hypothesis_text=np.array(hypothesis),
        )


if __name__ == "__main__":
    import sys

    model_key = sys.argv[1] if len(sys.argv) > 1 else "apertus"
    for lang in TARGET_LANGS:
        extract_translation_condition(model_key, lang)
