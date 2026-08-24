"""Stage 1 — reference vectors, shared across both conditions.

Built once per model from FLORES+ `dev`, using the neutral continuation
prompt (not translation), mean-pooled over the sentence's own token span.
"""

import json
import os

import numpy as np

from flores_pipeline.config import (
    GAP_FLOOR,
    LANGUAGES,
    MODEL_PATHS,
    NEUTRAL_PROMPT_TEMPLATE,
    OUTPUTS_DIR,
    TARGET_LANGS,
)
from flores_pipeline.data.load_flores import load_flores_split
from flores_pipeline.extraction.model_utils import extract_hidden_states_for_span, load_model


def _mean_hidden_for_lang(loaded, sentences):
    """Mean hidden-state vectors per layer, averaged over the given sentences."""
    per_sentence = []
    for sentence in sentences:
        prompt = NEUTRAL_PROMPT_TEMPLATE.format(sentence=sentence)
        hs = extract_hidden_states_for_span(loaded, prompt, sentence)  # [n_layers+1, dim]
        per_sentence.append(hs)
    return np.mean(np.stack(per_sentence, axis=0), axis=0)  # [n_layers+1, dim]


def build_reference_vectors(model_key: str, n_dev_sentences: int = 200):
    loaded = load_model(model_key, MODEL_PATHS[model_key])

    langs = ["en"] + TARGET_LANGS
    dev_examples = load_flores_split("dev", langs)[:n_dev_sentences]

    en_ref = _mean_hidden_for_lang(loaded, [ex.sentences["en"] for ex in dev_examples])
    target_refs = {}
    for lang in TARGET_LANGS:
        target_refs[lang] = _mean_hidden_for_lang(
            loaded, [ex.sentences[lang] for ex in dev_examples]
        )

    n_layers = en_ref.shape[0]
    v_lang = {lang: target_refs[lang] - en_ref for lang in TARGET_LANGS}

    # Gap-floor check per language, per layer: flags layers with an
    # unreliable (near-zero) denominator before they can produce a blown-up
    # containment ratio downstream.
    gap_flags = {}
    for lang in TARGET_LANGS:
        gaps = []
        for l in range(n_layers):
            dot_target = float(np.dot(target_refs[lang][l], v_lang[lang][l]))
            dot_en = float(np.dot(en_ref[l], v_lang[lang][l]))
            gaps.append(abs(dot_target - dot_en))
        gap_flags[lang] = [g < GAP_FLOOR for g in gaps]

    out_dir = os.path.join(OUTPUTS_DIR, f"references_{model_key}")
    os.makedirs(out_dir, exist_ok=True)

    reference_pairs = {
        "model": model_key,
        "n_dev_sentences": len(dev_examples),
        "n_layers": n_layers,
        "languages": TARGET_LANGS,
        "gap_floor": GAP_FLOOR,
        "unreliable_layers_by_lang": {
            lang: [i for i, flag in enumerate(flags) if flag]
            for lang, flags in gap_flags.items()
        },
    }
    with open(os.path.join(out_dir, "reference_pairs.json"), "w") as f:
        json.dump(reference_pairs, f, indent=2)

    save_dict = {"en_ref": en_ref}
    for lang in TARGET_LANGS:
        save_dict[f"target_ref_{lang}"] = target_refs[lang]
        save_dict[f"v_lang_{lang}"] = v_lang[lang]
    np.savez(os.path.join(out_dir, "contrastive_vectors.npz"), **save_dict)

    return reference_pairs


if __name__ == "__main__":
    import sys

    model_key = sys.argv[1] if len(sys.argv) > 1 else "apertus"
    build_reference_vectors(model_key)
