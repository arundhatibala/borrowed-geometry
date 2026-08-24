"""Language-ID wrapper around facebook/fasttext-language-identification.

Note (spec section 9): this model is validated at 93% accuracy under
truncation but drops to 53% under repeated-word degeneration — exactly the
failure mode high-alpha ablated generations can produce. LID output is kept
as its own curve (never folded into ΔCOMET) for this reason.
"""

from typing import Tuple

from flores_pipeline.config import LID_MODEL

_LID_MODEL_CACHE = None


def _get_lid_model():
    global _LID_MODEL_CACHE
    if _LID_MODEL_CACHE is None:
        import fasttext
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=LID_MODEL, filename="model.bin")
        _LID_MODEL_CACHE = fasttext.load_model(model_path)
    return _LID_MODEL_CACHE


def run_lid(text: str) -> Tuple[str, float]:
    """Returns (lang_code, confidence). lang_code is a FLORES-style code
    (e.g. "por_Latn"), stripped of the fasttext "__label__" prefix."""
    model = _get_lid_model()
    text = text.replace("\n", " ").strip()
    if not text:
        return "unk", 0.0
    labels, probs = model.predict(text, k=1)
    lang_code = labels[0].replace("__label__", "")
    return lang_code, float(probs[0])
