"""
Central config for the FLORES+ stepping-stone containment pipeline.

Several values here are flagged TODO/CONFIRM per spec section 12
("Known unresolved assumptions") and must be confirmed before running
at scale — they are filled with reasonable defaults so the pipeline is
runnable end-to-end on synthetic/small data during development, but
the numbers it produces should not be trusted until these are resolved.
"""

import os

# ---------------------------------------------------------------------------
# Models — TODO: fill in real checkpoint paths / HF ids (assumption #? — new)
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "apertus": os.environ.get("APERTUS_CHECKPOINT", "TODO/path/to/local/apertus/checkpoint"),
    "gemma": os.environ.get("GEMMA_CHECKPOINT", "google/gemma-4-31b-it"),
}

# ---------------------------------------------------------------------------
# Languages — (FLORES+ code, display name)
# ---------------------------------------------------------------------------
LANGUAGES = {
    "en": ("eng_Latn", "English"),
    "gl": ("glg_Latn", "Galician"), "pt": ("por_Latn", "Portuguese"),
    "ru": ("rus_Cyrl", "Russian"), "be": ("bel_Cyrl", "Belarusian"),
    "nl": ("nld_Latn", "Dutch"), "af": ("afr_Latn", "Afrikaans"),
    "sv": ("swe_Latn", "Swedish"), "is": ("isl_Latn", "Icelandic"),
}
TARGET_LANGS = [k for k in LANGUAGES if k != "en"]

# ---------------------------------------------------------------------------
# Families / pairs.
#
# ASSUMPTION #5 (spec section 12): lr/hr order per pair is a name-recognition
# carryover from the thesis, NOT a verified resource-share check. CONFIRM
# against a real source (e.g. FineWeb per-language token counts, same method
# as the thesis's original characterization) before trusting Containment_*
# directionality or the ablation "forward vs reversed" framing.
# ---------------------------------------------------------------------------
FAMILIES = {
    "iberian": {"pairs": [("gl", "pt")]},         # gl=LR (assumed), pt=HR (assumed)
    "slavic": {"pairs": [("be", "ru")]},           # be=LR (assumed), ru=HR (assumed)
    "germanic_west": {"pairs": [("af", "nl")]},    # af=LR (assumed), nl=HR (assumed)
    "germanic_north": {"pairs": [("is", "sv")]},   # is=LR (assumed), sv=HR (assumed)
}

# Cross-family control pairs used as a null-adjacent comparison in Stage 3/4.
# TODO: adapt exact pairing to mirror the thesis's gl/de, af/es-style controls.
CROSS_FAMILY_CONTROLS = [
    ("gl", "ru"),
    ("af", "sv"),
    ("is", "nl"),
    ("be", "pt"),
]

# ---------------------------------------------------------------------------
# Prompts — ASSUMPTION #1 (spec section 12): wording is a guess, not the
# original thesis wrapper text. CONFIRM exact wording before running at scale
# — prompt wording changes hidden-state geometry and is not a cosmetic choice.
# ---------------------------------------------------------------------------
NEUTRAL_PROMPT_TEMPLATE = "Continue the following sentence naturally: {sentence}"
TRANSLATION_PROMPT_TEMPLATE = "Translate the following English sentence into {target_language}: {sentence}"

# ---------------------------------------------------------------------------
# Reference-vector construction — ASSUMPTION #2 (spec section 12): difference
# of means. CONFIRM this matches whatever method actually produced the
# thesis's precomputed reference vectors.
# ---------------------------------------------------------------------------
GAP_FLOOR = 0.05

# ASSUMPTION #4 (spec section 12): reference-based COMET checkpoint, not
# COMET-Kiwi (reference-free). CONFIRM this is the intended model.
COMET_MODEL = "Unbabel/wmt22-comet-da"
LID_MODEL = "facebook/fasttext-language-identification"

MAX_NEW_TOKENS = 256
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
N_PERMUTATIONS = 1000

# ASSUMPTION #6 (spec section 12): Benjamini-Hochberg assumed over
# Bonferroni for the primary layer-scan family of tests. CONFIRM.
FDR_METHOD = "benjamini-hochberg"

# ---------------------------------------------------------------------------
# Degradation metric — ASSUMPTION #3 (spec section 12, called out at top of
# doc as needing an explicit yes/no): plain ΔCOMET(reference-based), NOT the
# ceiling/floor-normalized version. Cheaper to implement/explain, at the cost
# of not being able to compare degradation *magnitude* between the
# translation and reading conditions — only whether a monotonic dose-response
# exists in each independently (Stage 7). Flip this flag if normalization is
# actually wanted; downstream code in ablation/dose_response_translation.py
# branches on it.
# ---------------------------------------------------------------------------
USE_NORMALIZED_DEGRADATION = False

# ---------------------------------------------------------------------------
# Splits / paths
# ---------------------------------------------------------------------------
FLORES_DEV_SPLIT = "dev"
FLORES_DEVTEST_SPLIT = "devtest"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "flores_pipeline", "outputs")

CONDITIONS = ("translation", "reading")
