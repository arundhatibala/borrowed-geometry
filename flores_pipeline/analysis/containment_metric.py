"""Stage 3 — containment metric (variance-explained).

For pair (LR, HR) at layer l, using a condition's hidden states and Stage 1's
reference vectors:

    r_i = h_LR(l, sentence_i) - en_ref(l)          # LR's drift away from English
    Containment_LR->HR(l) = Var_i( dot(r_i, v_HR(l)) ) / Var_i(r_i)
    Containment_HR->LR(l) = mirror, using h_HR and v_LR
    Asymmetry(l) = Containment_LR->HR(l) - Containment_HR->LR(l)

Var_i(r_i) is the trace of the drift vectors' covariance across sentences —
a variance over many sentences, not a single English-vs-target distance, so
it structurally removes the small-denominator blowup risk that the thesis's
raw two-point gap had.

Run identically on both conditions' corpora, independently, per model.
"""

import glob
import json
import os

import numpy as np

from flores_pipeline.config import CROSS_FAMILY_CONTROLS, FAMILIES, OUTPUTS_DIR


HIDDEN_KEY_BY_CONDITION = {
    "translation": "translation_hidden",
    "reading": "reading_hidden",
}


def load_reference_vectors(model_key: str):
    ref_dir = os.path.join(OUTPUTS_DIR, f"references_{model_key}")
    data = np.load(os.path.join(ref_dir, "contrastive_vectors.npz"))
    return {key: data[key] for key in data.files}


def load_hidden_states(model_key: str, condition: str, lang: str) -> np.ndarray:
    """Returns array [n_sentences, n_layers+1, dim]."""
    hidden_key = HIDDEN_KEY_BY_CONDITION[condition]
    lang_dir = os.path.join(
        OUTPUTS_DIR, f"hidden_states_{model_key}_{condition}", f"{lang}_{model_key}"
    )
    files = sorted(glob.glob(os.path.join(lang_dir, "ex_*.npz")))
    vectors = []
    for fp in files:
        data = np.load(fp)
        vectors.append(data[hidden_key])
    return np.stack(vectors, axis=0)


def _drift(h_lang: np.ndarray, en_ref: np.ndarray) -> np.ndarray:
    """h_lang: [n_sentences, n_layers, dim], en_ref: [n_layers, dim] -> [n_sentences, n_layers, dim]"""
    return h_lang - en_ref[None, :, :]


def _containment_per_layer(r: np.ndarray, v_target: np.ndarray) -> np.ndarray:
    """r: [n_sentences, n_layers, dim], v_target: [n_layers, dim] -> [n_layers] containment ratio."""
    n_layers = r.shape[1]
    out = np.zeros(n_layers)
    for l in range(n_layers):
        proj = r[:, l, :] @ v_target[l]  # [n_sentences]
        numerator = np.var(proj)
        denominator = np.trace(np.cov(r[:, l, :], rowvar=False))
        out[l] = numerator / denominator if denominator > 0 else np.nan
    return out


def compute_containment_for_pair(
    model_key: str, condition: str, lr_lang: str, hr_lang: str, refs=None
):
    refs = refs or load_reference_vectors(model_key)
    en_ref = refs["en_ref"]
    v_lr = refs[f"v_lang_{lr_lang}"]
    v_hr = refs[f"v_lang_{hr_lang}"]

    h_lr = load_hidden_states(model_key, condition, lr_lang)
    h_hr = load_hidden_states(model_key, condition, hr_lang)

    r_lr = _drift(h_lr, en_ref)
    r_hr = _drift(h_hr, en_ref)

    containment_lr_to_hr = _containment_per_layer(r_lr, v_hr)
    containment_hr_to_lr = _containment_per_layer(r_hr, v_lr)
    asymmetry = containment_lr_to_hr - containment_hr_to_lr

    return {
        "containment_lr_to_hr": containment_lr_to_hr,
        "containment_hr_to_lr": containment_hr_to_lr,
        "asymmetry": asymmetry,
    }


def run_containment_stage(model_key: str, condition: str):
    refs = load_reference_vectors(model_key)
    out_dir = os.path.join(OUTPUTS_DIR, "containment")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for family, spec in FAMILIES.items():
        for lr_lang, hr_lang in spec["pairs"]:
            res = compute_containment_for_pair(model_key, condition, lr_lang, hr_lang, refs)
            results[f"{family}:{lr_lang}-{hr_lang}"] = {
                k: v.tolist() for k, v in res.items()
            }
            out_path = os.path.join(
                out_dir, f"{model_key}_{condition}_{family}.json"
            )
            with open(out_path, "w") as f:
                json.dump(
                    {"pair": [lr_lang, hr_lang], **{k: v.tolist() for k, v in res.items()}},
                    f,
                    indent=2,
                )

    # Cross-family control pairs, treated as their own (non-family) group.
    control_results = {}
    for lr_lang, hr_lang in CROSS_FAMILY_CONTROLS:
        res = compute_containment_for_pair(model_key, condition, lr_lang, hr_lang, refs)
        control_results[f"{lr_lang}-{hr_lang}"] = {k: v.tolist() for k, v in res.items()}
    with open(
        os.path.join(out_dir, f"{model_key}_{condition}_cross_family_controls.json"), "w"
    ) as f:
        json.dump(control_results, f, indent=2)

    return results, control_results


if __name__ == "__main__":
    import sys

    model_key = sys.argv[1] if len(sys.argv) > 1 else "apertus"
    condition = sys.argv[2] if len(sys.argv) > 2 else "translation"
    run_containment_stage(model_key, condition)
