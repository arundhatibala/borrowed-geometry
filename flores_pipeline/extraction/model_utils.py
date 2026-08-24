"""Shared model-loading and hidden-state extraction utilities.

Extraction always distinguishes "wrapper template tokens" from "the
sentence's own token span" so that mean-pooling can be restricted to the
span that actually belongs to the sentence (or, in the translation
condition, to the generated hypothesis) — never the prompt scaffolding.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str


_MODEL_CACHE = {}


def load_model(model_key: str, model_path: str, device: Optional[str] = None) -> LoadedModel:
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, output_hidden_states=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    loaded = LoadedModel(name=model_key, model=model, tokenizer=tokenizer, device=device)
    _MODEL_CACHE[model_key] = loaded
    return loaded


def _find_span_token_range(
    tokenizer, full_text: str, span_text: str
) -> Tuple[int, int]:
    """Locate the token index range covered by `span_text` within `full_text`.

    Uses offset mapping from a fast tokenizer to map the character span of
    `span_text` (assumed to be a substring of `full_text`, e.g. the sentence
    within a wrapper prompt) to a [start_tok, end_tok) token range.
    """
    char_start = full_text.rindex(span_text)
    char_end = char_start + len(span_text)
    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    start_tok, end_tok = None, None
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if start_tok is None and e > char_start:
            start_tok = i
        if s < char_end:
            end_tok = i + 1
    if start_tok is None or end_tok is None or start_tok >= end_tok:
        raise ValueError(f"Could not locate span {span_text!r} within {full_text!r}")
    return start_tok, end_tok


def extract_hidden_states_for_span(
    loaded: LoadedModel, full_text: str, span_text: str
) -> np.ndarray:
    """Forward pass over `full_text`, mean-pool hidden states over `span_text`'s
    own token range, per layer.

    Returns array of shape [n_layers + 1, hidden_dim] (includes embedding
    layer at index 0, matching `output_hidden_states` convention).
    """
    tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device
    start_tok, end_tok = _find_span_token_range(tokenizer, full_text, span_text)

    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    layers = []
    for layer_hidden in out.hidden_states:  # tuple of [1, seq_len, dim]
        span_repr = layer_hidden[0, start_tok:end_tok, :].mean(dim=0)
        layers.append(span_repr.cpu().numpy())
    return np.stack(layers, axis=0)


def teacher_forced_mean_nll(loaded: LoadedModel, text: str) -> float:
    """Mean per-token cross-entropy (natural log) of `text` under teacher forcing."""
    tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return float(out.loss.item())


def generate(loaded: LoadedModel, prompt: str, max_new_tokens: int) -> str:
    tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)
