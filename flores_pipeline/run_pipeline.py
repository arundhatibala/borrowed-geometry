"""End-to-end runner: Stage 1 -> Stage 7, per model, per condition.

Usage:
    python -m flores_pipeline.run_pipeline --models apertus gemma
"""

import argparse

from flores_pipeline.analysis.containment_metric import run_containment_stage
from flores_pipeline.analysis.cross_comparison import run_cross_comparison_stage
from flores_pipeline.analysis.layer_selection import run_layer_selection_stage
from flores_pipeline.analysis.permutation_null import run_permutation_stage
from flores_pipeline.config import CONDITIONS, FAMILIES, MODEL_PATHS, TARGET_LANGS
from flores_pipeline.extraction.corpus_reading import extract_reading_condition
from flores_pipeline.extraction.corpus_translation import extract_translation_condition
from flores_pipeline.extraction.reference_vectors import build_reference_vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_PATHS.keys()))
    parser.add_argument("--n-dev", type=int, default=200)
    parser.add_argument("--n-devtest", type=int, default=50)
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    pairs = [pair for spec in FAMILIES.values() for pair in spec["pairs"]]

    for model_key in args.models:
        print(f"=== {model_key}: Stage 1 (reference vectors) ===")
        build_reference_vectors(model_key, n_dev_sentences=args.n_dev)

        if not args.skip_extraction:
            for lang in TARGET_LANGS:
                print(f"=== {model_key}/{lang}: Stage 2a (translation extraction) ===")
                extract_translation_condition(model_key, lang, n_sentences=args.n_devtest)
                print(f"=== {model_key}/{lang}: Stage 2b (reading extraction) ===")
                extract_reading_condition(model_key, lang, n_sentences=args.n_devtest)

        for condition in CONDITIONS:
            print(f"=== {model_key}/{condition}: Stage 3 (containment) ===")
            run_containment_stage(model_key, condition)
            print(f"=== {model_key}/{condition}: Stage 4 (permutation null) ===")
            run_permutation_stage(model_key, condition, FAMILIES)
            print(f"=== {model_key}/{condition}: Stage 5 (layer selection) ===")
            run_layer_selection_stage(model_key, condition)

    print("=== Stage 7 (cross-model / cross-condition comparison) ===")
    run_cross_comparison_stage(args.models, pairs)

    print(
        "NOTE: Stage 6 (causal ablation) requires the layer band selected in "
        "Stage 5 per pair and is intentionally not auto-run here — invoke "
        "ablation.dose_response_translation / dose_response_reading directly "
        "with the band from outputs/containment/<model>_<condition>_layer_selection.json."
    )


if __name__ == "__main__":
    main()
