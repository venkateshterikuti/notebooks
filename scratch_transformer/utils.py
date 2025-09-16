import time
from typing import Dict

from .xb import np

np.set_printoptions(precision=4, suppress=True)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def lap(self, msg: str = "") -> float:
        t = time.time() - self.t0
        self.t0 = time.time()
        if msg:
            print(f"[timer] {msg}: {t:.3f}s")
        return t


def count_params(param_dict: Dict[str, np.ndarray]) -> int:
    return int(sum(int(p.size) for p in param_dict.values()))


def gelu(x):
    coeff = np.sqrt(2.0 / np.pi)
    cubic = x * x * x
    inner = coeff * (x + 0.044715 * cubic)
    return 0.5 * x * (1.0 + np.tanh(inner))


def dgelu(x):
    coeff = np.sqrt(2.0 / np.pi)
    cubic = x * x * x
    inner = coeff * (x + 0.044715 * cubic)
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    term1 = 0.5 * (1.0 + tanh_inner)
    term2 = 0.5 * x * sech2 * coeff * (1.0 + 3.0 * 0.044715 * x * x)
    return term1 + term2


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_backward(grad_out, softmax_out):
    s = softmax_out
    dot = np.sum(grad_out * s, axis=-1, keepdims=True)
    return (grad_out - dot) * s


def causal_mask(T: int):
    return np.triu(np.ones((T, T), dtype=bool), k=1)


def xavier_uniform(shape, gain=1.0, rng=None):
    if rng is None:
        default_rng = getattr(np.random, "default_rng", None)
        rng = default_rng() if callable(default_rng) else np.random
    fan_in, fan_out = shape[0], shape[1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    values = rng.uniform(-limit, limit, size=shape)
    return np.asarray(values, dtype=np.float32)
