"""Reference-based COMET wrapper.

Uses the checkpoint named in config.COMET_MODEL — assumed
Unbabel/wmt22-comet-da (reference-based), NOT COMET-Kiwi
(reference-free). See config.py ASSUMPTION #4.
"""

from typing import List

from flores_pipeline.config import COMET_MODEL

_COMET_MODEL_CACHE = None


def _get_comet_model():
    global _COMET_MODEL_CACHE
    if _COMET_MODEL_CACHE is None:
        from comet import download_model, load_from_checkpoint

        ckpt_path = download_model(COMET_MODEL)
        _COMET_MODEL_CACHE = load_from_checkpoint(ckpt_path)
    return _COMET_MODEL_CACHE


def compute_comet(source: str, gold_reference: str, hypothesis: str) -> float:
    """Reference-based COMET score for a single (source, reference, hypothesis) triple."""
    return compute_comet_batch([source], [gold_reference], [hypothesis])[0]


def compute_comet_batch(
    sources: List[str], gold_references: List[str], hypotheses: List[str]
) -> List[float]:
    model = _get_comet_model()
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, r, h in zip(sources, gold_references, hypotheses)
    ]
    output = model.predict(data, batch_size=8, gpus=0)
    return list(output["scores"])
