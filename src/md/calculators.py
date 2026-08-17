"""The MACE calculator, loaded once and reused.

Two precisions are kept alive at the same time: float64 for the hydrogen
relaxation, where the forces matter, and float32 for the dynamics, which is far
faster and well inside the thermal noise being sampled. Loading the model is
slow enough that caching it across motifs is worth the few hundred MB.
"""

_CACHE = {}


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def calculator(model, dtype, dev=None):
    dev = dev or device()
    key = (model, dtype, dev)
    if key not in _CACHE:
        from mace.calculators import mace_off
        _CACHE[key] = mace_off(model=model, default_dtype=dtype, device=dev)
    return _CACHE[key]
