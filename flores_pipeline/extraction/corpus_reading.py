"""Stage 2b — corpus extraction, READING condition.

The model reads the real target-language gold sentence (never English,
never model output, no translation instruction anywhere) wrapped in the
same neutral continuation prompt used for Stage 1. Also computes
teacher-forced mean NLL, needed later for the ablation dose-response
(Stage 6b) so ablated-vs-baseline deltas can be computed per sentence
without re-running the forward pass.
"""

import os

import numpy as np

from flores_pipeline.config import (
    FLORES_DEVTEST_SPLIT,
    MODEL_PATHS,
    NEUTRAL_PROMPT_TEMPLATE,
    OUTPUTS_DIR,
    TARGET_LANGS,
)
from flores_pipeline.data.load_flores import load_flores_split
from flores_pipeline.extraction.model_utils import (
    extract_hidden_states_for_span,
    load_model,
    teacher_forced_mean_nll,
)


def extract_reading_condition(model_key: str, lang: str, n_sentences: int = None):
    loaded = load_model(model_key, MODEL_PATHS[model_key])

    examples = load_flores_split(FLORES_DEVTEST_SPLIT, [lang])
    if n_sentences is not None:
        examples = examples[:n_sentences]

    out_dir = os.path.join(
        OUTPUTS_DIR, f"hidden_states_{model_key}_reading", f"{lang}_{model_key}"
    )
    os.makedirs(out_dir, exist_ok=True)

    for ex in examples:
        sentence = ex.sentences[lang]
        prompt = NEUTRAL_PROMPT_TEMPLATE.format(sentence=sentence)

        reading_hidden = extract_hidden_states_for_span(loaded, prompt, sentence)
        mean_nll = teacher_forced_mean_nll(loaded, sentence)

        np.savez(
            os.path.join(out_dir, f"ex_{ex.idx:04d}.npz"),
            reading_hidden=reading_hidden,
            mean_nll=np.array(mean_nll),
        )


if __name__ == "__main__":
    import sys

    model_key = sys.argv[1] if len(sys.argv) > 1 else "apertus"
    for lang in TARGET_LANGS:
        extract_reading_condition(model_key, lang)
