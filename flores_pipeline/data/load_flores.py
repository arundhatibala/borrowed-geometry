"""FLORES+ text loading and line-alignment checks."""

from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset

from flores_pipeline.config import LANGUAGES


@dataclass
class FloresExample:
    idx: int
    sentences: Dict[str, str]  # lang_code (our short code, e.g. "en") -> sentence text


def _flores_config(lang_short: str) -> str:
    flores_code, _ = LANGUAGES[lang_short]
    return flores_code


def load_flores_split(split: str, langs: List[str]) -> List[FloresExample]:
    """Load a FLORES+ split for the given short language codes, line-aligned.

    `split` is one of "dev" / "devtest". Uses the openlanguagedata/flores_plus
    dataset on the Hugging Face Hub, one config per language, and zips rows
    by index (FLORES+ is sentence-aligned by construction: row i in every
    per-language config is a translation of the same underlying sentence).
    """
    per_lang_rows = {}
    n = None
    for lang in langs:
        cfg = _flores_config(lang)
        ds = load_dataset("openlanguagedata/flores_plus", cfg, split=split)
        per_lang_rows[lang] = [row["text"] for row in ds]
        if n is None:
            n = len(per_lang_rows[lang])
        elif len(per_lang_rows[lang]) != n:
            raise ValueError(
                f"FLORES+ line-alignment check failed: {lang} has "
                f"{len(per_lang_rows[lang])} rows, expected {n}"
            )

    examples = []
    for i in range(n):
        examples.append(
            FloresExample(idx=i, sentences={lang: per_lang_rows[lang][i] for lang in langs})
        )
    return examples
