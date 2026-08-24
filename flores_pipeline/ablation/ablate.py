"""Projection suppression at a given layer, dose alpha.

Suppression removes `alpha` of the hidden state's projection onto a given
direction vector `v` at the specified decoder layer(s):

    h' = h - alpha * (h . v_hat) * v_hat

alpha=0 is a no-op (baseline); alpha=1 fully zeroes the component along v.
Hooked onto `model.model.layers[l]` (or `model.transformer.h[l]` fallback)
forward output, applied to every token position in the sequence.
"""

from contextlib import contextmanager
from typing import List

import numpy as np
import torch


def _get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate decoder layer list on model")


def _make_hook(v_hat: torch.Tensor, alpha: float):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        proj = torch.einsum("bsd,d->bs", hidden, v_hat)
        hidden = hidden - alpha * proj.unsqueeze(-1) * v_hat
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook


@contextmanager
def suppress_projection(model, layer_indices: List[int], direction: np.ndarray, alpha: float):
    """Context manager: while active, forward passes through `model` have the
    projection onto `direction` suppressed by `alpha` at each layer in
    `layer_indices` (matching the `hidden_states` layer indexing convention,
    where index 0 is the embedding layer and index i corresponds to the
    output of decoder block i-1).
    """
    if alpha == 0.0 or not layer_indices:
        yield
        return

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    v = torch.tensor(direction, device=device, dtype=dtype)
    v_hat = v / v.norm()

    decoder_layers = _get_decoder_layers(model)
    handles = []
    for l in layer_indices:
        block_idx = l - 1  # hidden_states[l] is the output of decoder_layers[l - 1]
        if block_idx < 0 or block_idx >= len(decoder_layers):
            continue
        handle = decoder_layers[block_idx].register_forward_hook(_make_hook(v_hat, alpha))
        handles.append(handle)

    try:
        yield
    finally:
        for h in handles:
            h.remove()
