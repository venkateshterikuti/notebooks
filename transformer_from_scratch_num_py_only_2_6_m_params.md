# Transformer-from-scratch (NumPy-only) — 2–6M params

An educational, fully-from-scratch decoder-only Transformer written with **just Python + NumPy**. No PyTorch/JAX/TF. All forward **and backward** passes are implemented layer-by-layer (Linear, LayerNorm, Multi-Head Self-Attention, MLP, GELU, Softmax-CE) along with an **AdamW** optimizer.

It’s intentionally compact yet faithful, sized to ~**2.5M** and **~5.1M** parameter presets so it trains on CPU (e.g., M‑series MacBooks) in a reasonable time on **TinyShakespeare** or similar byte-level corpora. Pairs nicely with a separate, focused **“Attention from Scratch”** blog—see `attention_only.py` below for a standalone, explainable attention demo you can link to.

---

## Repo Layout
```
transformer-from-scratch/
├─ README.md
├─ scratch_transformer/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ layers.py
│  ├─ model.py
│  ├─ optim.py
│  ├─ trainer.py
│  ├─ utils.py
│  └─ attention_only.py
├─ experiments/
│  ├─ tinyshakespeare.yaml
│  └─ run_tinyshakespeare.py
├─ scripts/
│  └─ download_tinyshakespeare.py
└─ tests/
   └─ test_shapes.py
```

---

## Quickstart
```bash
# 1) Create & activate a lightweight env (optional)
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip numpy

# 2) Get a tiny toy dataset
python scripts/download_tinyshakespeare.py

# 3) Run a training experiment (defaults to the ~2.5M-parameter preset)
python experiments/run_tinyshakespeare.py --config experiments/tinyshakespeare.yaml

# 4) Try the ~5.1M-parameter preset
python experiments/run_tinyshakespeare.py --config experiments/tinyshakespeare.yaml --size small5m
```

> Tip: Edit `experiments/tinyshakespeare.yaml` for batch size, sequence length, learning rate, steps, etc.

---

## Parameters at a glance
- **mini2p5m**: `d_model=256`, `n_heads=4`, `n_layers=3`, `d_ff=4*d` → ≈ **2.5M** params
- **small5m**: `d_model=320`, `n_heads=5`, `n_layers=4`, `d_ff=4*d` → ≈ **5.1M** params

These keep memory & compute accessible for CPU-only runs while demonstrating authentic Transformer internals.

---

## File-by-file

### `scratch_transformer/utils.py`
```python
import math, time
from .xb import np  # NumPy or CuPy backend selected via SCRATCH_DEVICE

class Timer:
    def __init__(self):
        self.t0 = time.time()
    def lap(self, msg=""):
        t = time.time() - self.t0
        self.t0 = time.time()
        if msg:
            print(f"[timer] {msg}: {t:.3f}s")
        return t

# pretty prints
try:
    np.set_printoptions(precision=4, suppress=True)
except Exception:
    pass

def count_params(param_dict):
    return sum(int(p.size) for p in param_dict.values())

def gelu(x):
    # exact GELU via erf
    return 0.5 * x * (1.0 + np.erf(x / np.sqrt(2.0)))

def dgelu(x):
    # derivative of exact GELU
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    z = x / np.sqrt(2.0)
    return 0.5 * (1.0 + np.erf(z)) + x * np.exp(-z*z) * inv_sqrt_2pi


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_backward(grad_out, softmax_out):
    s = softmax_out
    dot = np.sum(grad_out * s, axis=-1, keepdims=True)
    return (grad_out - dot) * s


def causal_mask(T):
    m = np.triu(np.ones((T, T), dtype=bool), k=1)
    return m


def xavier_uniform(shape, gain=1.0):
    fan_in, fan_out = shape[0], shape[1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return (np.random.uniform(-limit, limit, size=shape)).astype(np.float32)
```python
import math, time, numpy as np

class Timer:
    def __init__(self):
        self.t0 = time.time()
    def lap(self, msg=""):
        t = time.time() - self.t0
        self.t0 = time.time()
        if msg:
            print(f"[timer] {msg}: {t:.3f}s")
        return t

np.set_printoptions(precision=4, suppress=True)

def count_params(param_dict):
    return sum(p.size for p in param_dict.values())

def gelu(x):
    # exact GELU via erf
    return 0.5 * x * (1.0 + np.erf(x / np.sqrt(2.0)))

def dgelu(x):
    # derivative of exact GELU
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    z = x / np.sqrt(2.0)
    return 0.5 * (1.0 + np.erf(z)) + x * np.exp(-z*z) * inv_sqrt_2pi


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_backward(grad_out, softmax_out):
    # grad wrt logits given upstream grad wrt softmax output
    # dL/dz = (dL/dy - sum(dL/dy * y)) * y
    s = softmax_out
    dot = np.sum(grad_out * s, axis=-1, keepdims=True)
    return (grad_out - dot) * s


def causal_mask(T):
    # shape [T, T], True where masked (future)
    m = np.triu(np.ones((T, T), dtype=bool), k=1)
    return m


def xavier_uniform(shape, gain=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    fan_in, fan_out = shape[0], shape[1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)
```

---

### `scratch_transformer/layers.py`
```python
from .xb import np
from .utils import gelu, dgelu, softmax, softmax_backward, xavier_uniform

class Linear:
    def __init__(self, d_in, d_out):
        self.W = xavier_uniform((d_in, d_out))
        self.b = np.zeros((d_out,), dtype=np.float32)
        self.grads = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        self.cache = None
    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b
    def backward(self, grad_y):
        x = self.cache
        self.grads["W"][:] = x.reshape(-1, x.shape[-1]).T @ grad_y.reshape(-1, grad_y.shape[-1])
        self.grads["b"][:] = grad_y.sum(axis=tuple(range(grad_y.ndim-1)))
        grad_x = grad_y @ self.W.T
        return grad_x

class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.gamma = np.ones((d,), dtype=np.float32)
        self.beta  = np.zeros((d,), dtype=np.float32)
        self.eps = eps
        self.grads = {"gamma": np.zeros_like(self.gamma), "beta": np.zeros_like(self.beta)}
        self.cache = None
    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mu) * inv
        out = xhat * self.gamma + self.beta
        self.cache = (xhat, inv, x)
        return out
    def backward(self, grad_y):
        xhat, inv, x = self.cache
        N = x.shape[-1]
        self.grads["gamma"][:] = (grad_y * xhat).sum(axis=tuple(range(grad_y.ndim-1)))
        self.grads["beta" ][:] = grad_y.sum(axis=tuple(range(grad_y.ndim-1)))
        dxhat = grad_y * self.gamma
        sum_dxhat = dxhat.sum(axis=-1, keepdims=True)
        sum_dxhat_xhat = (dxhat * xhat).sum(axis=-1, keepdims=True)
        grad_x = (dxhat - xhat * sum_dxhat_xhat - sum_dxhat / N) * inv
        return grad_x

class MLP:
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.cache = None
    def forward(self, x):
        z1 = self.fc1.forward(x)
        a1 = gelu(z1)
        z2 = self.fc2.forward(a1)
        self.cache = (z1, a1)
        return z2
    def backward(self, grad_y):
        z1, a1 = self.cache
        da1 = self.fc2.backward(grad_y)
        dz1 = dgelu(z1) * da1
        grad_x = self.fc1.backward(dz1)
        return grad_x

class MultiHeadSelfAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = xavier_uniform((d_model, d_model))
        self.Wk = xavier_uniform((d_model, d_model))
        self.Wv = xavier_uniform((d_model, d_model))
        self.Wo = xavier_uniform((d_model, d_model))
        self.grads = {
            "Wq": np.zeros((d_model, d_model), dtype=np.float32),
            "Wk": np.zeros((d_model, d_model), dtype=np.float32),
            "Wv": np.zeros((d_model, d_model), dtype=np.float32),
            "Wo": np.zeros((d_model, d_model), dtype=np.float32),
        }
        self.cache = None
    def _split_heads(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        return x.reshape(B, T, H, Dh).transpose(0,2,1,3).reshape(B*H, T, Dh)
    def _merge_heads(self, xh, B):
        H, Dh = self.n_heads, self.d_head
        T = xh.shape[1]
        return xh.reshape(B, H, T, Dh).transpose(0,2,1,3).reshape(B, T, H*Dh)
    def forward(self, x, attn_mask):
        B, T, D = x.shape
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)
        scale = 1.0 / np.sqrt(self.d_head)
        scores = Qh @ Kh.transpose(0, 2, 1) * scale
        if attn_mask is not None:
            scores = scores.copy()
            scores[:, attn_mask] = -1e9
        P = softmax(scores, axis=-1)
        Ah = P @ Vh
        A = self._merge_heads(Ah, B)
        out = A @ self.Wo
        self.cache = (x, Qh, Kh, Vh, P, attn_mask, A, Ah)
        return out
    def backward(self, grad_y):
        x, Qh, Kh, Vh, P, attn_mask, A, Ah = self.cache
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        self.grads["Wo"][:] = A.reshape(-1, D).T @ grad_y.reshape(-1, D)
        dA = grad_y @ self.Wo.T
        dAh = dA.reshape(B, T, H, Dh).transpose(0,2,1,3).reshape(B*H, T, Dh)
        dP = dAh @ Vh.transpose(0,2,1)
        dVh = P.transpose(0,2,1) @ dAh
        dScores = softmax_backward(dP, P)
        if attn_mask is not None:
            dScores[:, attn_mask] = 0.0
        dQh = dScores @ Kh
        dKh = dScores.transpose(0,2,1) @ Qh
        inv = (1.0 / np.sqrt(Dh))
        dQh *= inv
        dKh *= inv
        dQ = dQh.reshape(B, H, T, Dh).transpose(0,2,1,3).reshape(B, T, D)
        dK = dKh.reshape(B, H, T, Dh).transpose(0,2,1,3).reshape(B, T, D)
        dV = dVh.reshape(B, H, T, Dh).transpose(0,2,1,3).reshape(B, T, D)
        self.grads["Wq"][:] = x.reshape(-1, D).T @ dQ.reshape(-1, D)
        self.grads["Wk"][:] = x.reshape(-1, D).T @ dK.reshape(-1, D)
        self.grads["Wv"][:] = x.reshape(-1, D).T @ dV.reshape(-1, D)
        dx = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        return dx
```python
import numpy as np
from .utils import gelu, dgelu, softmax, softmax_backward, xavier_uniform

class Linear:
    def __init__(self, d_in, d_out, rng=None):
        self.W = xavier_uniform((d_in, d_out), rng=rng)
        self.b = np.zeros((d_out,), dtype=np.float32)
        self.grads = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        self.cache = None
    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b
    def backward(self, grad_y):
        x = self.cache
        self.grads["W"][:] = x.reshape(-1, x.shape[-1]).T @ grad_y.reshape(-1, grad_y.shape[-1])
        self.grads["b"][:] = grad_y.sum(axis=(0,1)) if grad_y.ndim==3 else grad_y.sum(axis=0)
        grad_x = grad_y @ self.W.T
        return grad_x

class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.gamma = np.ones((d,), dtype=np.float32)
        self.beta  = np.zeros((d,), dtype=np.float32)
        self.eps = eps
        self.grads = {"gamma": np.zeros_like(self.gamma), "beta": np.zeros_like(self.beta)}
        self.cache = None
    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mu) * inv
        out = xhat * self.gamma + self.beta
        self.cache = (xhat, inv, x)
        return out
    def backward(self, grad_y):
        xhat, inv, x = self.cache
        N = x.shape[-1]
        self.grads["gamma"][:] = (grad_y * xhat).sum(axis=(0,1)) if grad_y.ndim==3 else (grad_y * xhat).sum(axis=0)
        self.grads["beta" ][:] = grad_y.sum(axis=(0,1)) if grad_y.ndim==3 else grad_y.sum(axis=0)
        dxhat = grad_y * self.gamma
        # LN backward (vectorized per row)
        sum_dxhat = dxhat.sum(axis=-1, keepdims=True)
        sum_dxhat_xhat = (dxhat * xhat).sum(axis=-1, keepdims=True)
        grad_x = (dxhat - xhat * sum_dxhat_xhat - sum_dxhat / N) * inv
        return grad_x

class MLP:
    def __init__(self, d_model, d_ff, rng=None):
        self.fc1 = Linear(d_model, d_ff, rng)
        self.fc2 = Linear(d_ff, d_model, rng)
        self.cache = None
    def forward(self, x):
        z1 = self.fc1.forward(x)
        a1 = gelu(z1)
        z2 = self.fc2.forward(a1)
        self.cache = (z1, a1)
        return z2
    def backward(self, grad_y):
        z1, a1 = self.cache
        da1 = self.fc2.backward(grad_y)
        dz1 = dgelu(z1) * da1
        grad_x = self.fc1.backward(dz1)
        return grad_x

class MultiHeadSelfAttention:
    def __init__(self, d_model, n_heads, rng=None):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = xavier_uniform((d_model, d_model), rng=rng)
        self.Wk = xavier_uniform((d_model, d_model), rng=rng)
        self.Wv = xavier_uniform((d_model, d_model), rng=rng)
        self.Wo = xavier_uniform((d_model, d_model), rng=rng)
        self.grads = {
            "Wq": np.zeros_like(self.Wq),
            "Wk": np.zeros_like(self.Wk),
            "Wv": np.zeros_like(self.Wv),
            "Wo": np.zeros_like(self.Wo),
        }
        self.cache = None
    def _split_heads(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        return x.reshape(B, T, H, Dh).transpose(0,2,1,3).reshape(B*H, T, Dh)
    def _merge_heads(self, xh, B):
        # xh: [B*H, T, Dh] → [B, T, D]
        H, Dh = self.n_heads, self.d_head
        T = xh.shape[1]
        return xh.reshape(B, H, T, Dh).transpose(0,2,1,3).reshape(B, T, H*Dh)
    def forward(self, x, attn_mask):
        # x: [B, T, D]
        B, T, D = x.shape
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)
        scale = 1.0 / np.sqrt(self.d_head)
        # scores: [B*H, T, T]
        scores = Qh @ Kh.transpose(0, 2, 1) * scale
        # apply causal mask (broadcast across heads)
        # attn_mask: [T, T] bool, True means mask
        if attn_mask is not None:
            scores = scores.copy()
            scores[:, attn_mask] = -1e9
        P = softmax(scores, axis=-1)
        Ah = P @ Vh
        A = self._merge_heads(Ah, B)
        out = A @ self.Wo
        self.cache = (x, Qh, Kh, Vh, P, attn_mask, A, Ah)
        return out
    def backward(self, grad_y):
        x, Qh, Kh, Vh, P, attn_mask, A, Ah = self.cache
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        # out = A @ Wo
        self.grads["Wo"][:] = A.reshape(-1, D).T @ grad_y.reshape(-1, D)
        dA = grad_y @ self.Wo.T
        # split heads
        dAh = dA.reshape(B, T, H, Dh).transpose(0,2,1,3).reshape(B*H, T, Dh)
        # Ah = P @ Vh
        dP = dAh @ Vh.transpose(0,2,1)
        dVh = P.transpose(0,2,1) @ dAh
        # P = softmax(scores)
        dScores = softmax_backward(dP, P)
        if attn_mask is not None:
            dScores[:, attn_mask] = 0.0
        # scores = (Qh @ Kh^T) * scale
        dQh = dScores @ Kh
        dKh = dScores.transpose(0,2,1) @ Qh
        dQh *= (1.0 / np.sqrt(Dh))
        dKh *= (1.0 / np.sqrt(Dh))
        # merge head grads
        dQ = dQh.reshape(B, H, T, Dh).transpose(0,2,1,3).reshape(B, T, D)
        dK = dKh.reshape(B, H, T, Dh).transpose(0,2,1,3).reshape(B, T, D)
        dV = dVh.reshape(B, H, T, Dh).transpose(0,2,1,3).reshape(B, T, D)
        # Q = x @ Wq; K = x @ Wk; V = x @ Wv
        self.grads["Wq"][:] = x.reshape(-1, D).T @ dQ.reshape(-1, D)
        self.grads["Wk"][:] = x.reshape(-1, D).T @ dK.reshape(-1, D)
        self.grads["Wv"][:] = x.reshape(-1, D).T @ dV.reshape(-1, D)
        dx = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        return dx
```

---

### `scratch_transformer/model.py`
```python
from .layers import Linear, LayerNorm, MLP, MultiHeadSelfAttention
from .utils import causal_mask
from .xb import np

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)
    def forward(self, x, mask):
        h = self.ln1.forward(x)
        a = self.attn.forward(h, mask)
        x = x + a
        h2 = self.ln2.forward(x)
        m = self.mlp.forward(h2)
        x = x + m
        return x
    def backward(self, grad_x):
        dm = grad_x
        dh2 = self.mlp.backward(dm)
        grad_x = grad_x + dh2
        grad_x = self.ln2.backward(grad_x)
        da = grad_x
        dh = self.attn.backward(da)
        grad_x = grad_x + dh
        grad_x = self.ln1.backward(grad_x)
        return grad_x

class Transformer:
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tok_emb = (np.random.rand(vocab_size, d_model).astype(np.float32) - 0.5) / d_model
        self.pos_emb = (np.random.rand(max_seq_len, d_model).astype(np.float32) - 0.5) / d_model
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.mask = causal_mask(max_seq_len)

    def parameters(self):
        params = {"tok_emb": self.tok_emb, "pos_emb": self.pos_emb,
                  "lm_head.W": self.lm_head.W, "lm_head.b": self.lm_head.b}
        for i, blk in enumerate(self.blocks):
            params.update({
                f"blocks.{i}.ln1.gamma": blk.ln1.gamma,
                f"blocks.{i}.ln1.beta":  blk.ln1.beta,
                f"blocks.{i}.attn.Wq":   blk.attn.Wq,
                f"blocks.{i}.attn.Wk":   blk.attn.Wk,
                f"blocks.{i}.attn.Wv":   blk.attn.Wv,
                f"blocks.{i}.attn.Wo":   blk.attn.Wo,
                f"blocks.{i}.ln2.gamma": blk.ln2.gamma,
                f"blocks.{i}.ln2.beta":  blk.ln2.beta,
                f"blocks.{i}.fc1.W":     blk.mlp.fc1.W,
                f"blocks.{i}.fc1.b":     blk.mlp.fc1.b,
                f"blocks.{i}.fc2.W":     blk.mlp.fc2.W,
                f"blocks.{i}.fc2.b":     blk.mlp.fc2.b,
            })
        return params

    def grads(self):
        grads = {"tok_emb": np.zeros_like(self.tok_emb), "pos_emb": np.zeros_like(self.pos_emb),
                 "lm_head.W": self.lm_head.grads["W"], "lm_head.b": self.lm_head.grads["b"]}
        for i, blk in enumerate(self.blocks):
            g = {
                f"blocks.{i}.ln1.gamma": blk.ln1.grads["gamma"],
                f"blocks.{i}.ln1.beta":  blk.ln1.grads["beta"],
                f"blocks.{i}.attn.Wq":   blk.attn.grads["Wq"],
                f"blocks.{i}.attn.Wk":   blk.attn.grads["Wk"],
                f"blocks.{i}.attn.Wv":   blk.attn.grads["Wv"],
                f"blocks.{i}.attn.Wo":   blk.attn.grads["Wo"],
                f"blocks.{i}.ln2.gamma": blk.ln2.grads["gamma"],
                f"blocks.{i}.ln2.beta":  blk.ln2.grads["beta"],
                f"blocks.{i}.fc1.W":     blk.mlp.fc1.grads["W"],
                f"blocks.{i}.fc1.b":     blk.mlp.fc1.grads["b"],
                f"blocks.{i}.fc2.W":     blk.mlp.fc2.grads["W"],
                f"blocks.{i}.fc2.b":     blk.mlp.fc2.grads["b"],
            }
            grads.update(g)
        return grads

    def zero_grads(self):
        self.lm_head.grads["W"].fill(0.0)
        self.lm_head.grads["b"].fill(0.0)
        for blk in self.blocks:
            blk.ln1.grads["gamma"].fill(0.0); blk.ln1.grads["beta"].fill(0.0)
            blk.attn.grads["Wq"].fill(0.0); blk.attn.grads["Wk"].fill(0.0); blk.attn.grads["Wv"].fill(0.0); blk.attn.grads["Wo"].fill(0.0)
            blk.ln2.grads["gamma"].fill(0.0); blk.ln2.grads["beta"].fill(0.0)
            blk.mlp.fc1.grads["W"].fill(0.0); blk.mlp.fc1.grads["b"].fill(0.0)
            blk.mlp.fc2.grads["W"].fill(0.0); blk.mlp.fc2.grads["b"].fill(0.0)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len
        tok = self.tok_emb[idx]
        pos = self.pos_emb[:T][None, :, :]
        x = tok + pos
        mask = self.mask[:T, :T]
        for blk in self.blocks:
            x = blk.forward(x, mask)
        x = self.ln_f.forward(x)
        logits = self.lm_head.forward(x)
        cache = (idx, tok, pos)
        return logits, cache

    def backward(self, dlogits, cache):
        idx, tok, pos = cache
        dx = self.lm_head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        grad_tok_emb = np.zeros_like(self.tok_emb)
        B, T = idx.shape
        for b in range(B):
            for t in range(T):
                grad_tok_emb[idx[b, t]] += dx[b, t]
        return grad_tok_emb, dx.sum(axis=0)
```python
import numpy as np
from .layers import Linear, LayerNorm, MLP, MultiHeadSelfAttention
from .utils import causal_mask

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, rng=None):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, rng)
    def forward(self, x, mask):
        h = self.ln1.forward(x)
        a = self.attn.forward(h, mask)
        x = x + a
        h2 = self.ln2.forward(x)
        m = self.mlp.forward(h2)
        x = x + m
        return x
    def backward(self, grad_x):
        # reverse of forward residual path
        # x = x + m
        dm = grad_x
        dh2 = self.mlp.backward(dm)
        grad_x = grad_x + dh2  # residual path to ln2 input
        # ln2 backward
        grad_x = self.ln2.backward(grad_x)
        # x = x + a
        da = grad_x
        dh = self.attn.backward(da)
        grad_x = grad_x + dh  # residual path to ln1 input
        # ln1 backward
        grad_x = self.ln1.backward(grad_x)
        return grad_x

class Transformer:
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, rng=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tok_emb = (np.random.rand(vocab_size, d_model).astype(np.float32) - 0.5) / d_model
        self.pos_emb = (np.random.rand(max_seq_len, d_model).astype(np.float32) - 0.5) / d_model
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, rng) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, rng)
        self.mask = causal_mask(max_seq_len)  # [T,T] bool

    def parameters(self):
        params = {"tok_emb": self.tok_emb, "pos_emb": self.pos_emb,
                  "lm_head.W": self.lm_head.W, "lm_head.b": self.lm_head.b}
        for i, blk in enumerate(self.blocks):
            params.update({
                f"blocks.{i}.ln1.gamma": blk.ln1.gamma,
                f"blocks.{i}.ln1.beta":  blk.ln1.beta,
                f"blocks.{i}.attn.Wq":   blk.attn.Wq,
                f"blocks.{i}.attn.Wk":   blk.attn.Wk,
                f"blocks.{i}.attn.Wv":   blk.attn.Wv,
                f"blocks.{i}.attn.Wo":   blk.attn.Wo,
                f"blocks.{i}.ln2.gamma": blk.ln2.gamma,
                f"blocks.{i}.ln2.beta":  blk.ln2.beta,
                f"blocks.{i}.fc1.W":     blk.mlp.fc1.W,
                f"blocks.{i}.fc1.b":     blk.mlp.fc1.b,
                f"blocks.{i}.fc2.W":     blk.mlp.fc2.W,
                f"blocks.{i}.fc2.b":     blk.mlp.fc2.b,
            })
        return params

    def grads(self):
        grads = {"tok_emb": np.zeros_like(self.tok_emb), "pos_emb": np.zeros_like(self.pos_emb),
                 "lm_head.W": self.lm_head.grads["W"], "lm_head.b": self.lm_head.grads["b"]}
        for i, blk in enumerate(self.blocks):
            g = {
                f"blocks.{i}.ln1.gamma": blk.ln1.grads["gamma"],
                f"blocks.{i}.ln1.beta":  blk.ln1.grads["beta"],
                f"blocks.{i}.attn.Wq":   blk.attn.grads["Wq"],
                f"blocks.{i}.attn.Wk":   blk.attn.grads["Wk"],
                f"blocks.{i}.attn.Wv":   blk.attn.grads["Wv"],
                f"blocks.{i}.attn.Wo":   blk.attn.grads["Wo"],
                f"blocks.{i}.ln2.gamma": blk.ln2.grads["gamma"],
                f"blocks.{i}.ln2.beta":  blk.ln2.grads["beta"],
                f"blocks.{i}.fc1.W":     blk.mlp.fc1.grads["W"],
                f"blocks.{i}.fc1.b":     blk.mlp.fc1.grads["b"],
                f"blocks.{i}.fc2.W":     blk.mlp.fc2.grads["W"],
                f"blocks.{i}.fc2.b":     blk.mlp.fc2.grads["b"],
            }
            grads.update(g)
        return grads

    def zero_grads(self):
        self.lm_head.grads["W"].fill(0.0)
        self.lm_head.grads["b"].fill(0.0)
        for blk in self.blocks:
            blk.ln1.grads["gamma"].fill(0.0); blk.ln1.grads["beta"].fill(0.0)
            blk.attn.grads["Wq"].fill(0.0); blk.attn.grads["Wk"].fill(0.0);
            blk.attn.grads["Wv"].fill(0.0); blk.attn.grads["Wo"].fill(0.0)
            blk.ln2.grads["gamma"].fill(0.0); blk.ln2.grads["beta"].fill(0.0)
            blk.mlp.fc1.grads["W"].fill(0.0); blk.mlp.fc1.grads["b"].fill(0.0)
            blk.mlp.fc2.grads["W"].fill(0.0); blk.mlp.fc2.grads["b"].fill(0.0)

    def forward(self, idx):
        # idx: [B,T] int16/32
        B, T = idx.shape
        assert T <= self.max_seq_len
        tok = self.tok_emb[idx]                     # [B,T,D]
        pos = self.pos_emb[:T][None, :, :]          # [1,T,D]
        x = tok + pos                               # [B,T,D]
        mask = self.mask[:T, :T]                    # [T,T]
        for blk in self.blocks:
            x = blk.forward(x, mask)
        x = self.ln_f.forward(x)
        logits = self.lm_head.forward(x)            # [B,T,V]
        cache = (idx, tok, pos)
        return logits, cache

    def backward(self, dlogits, cache):
        # dlogits: [B,T,V]
        idx, tok, pos = cache
        # back through lm_head and ln_f and blocks
        dx = self.lm_head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        # x = tok + pos, so gradient flows to both embeddings
        # accumulate into embedding matrices
        # tok = tok_emb[idx]
        B, T = idx.shape
        for b in range(B):
            for t in range(T):
                self.tok_emb[idx[b, t]] += 0.0  # ensure array is writable
        # The gradient wrt embedding lookup is just dx at those positions
        # We'll scatter-add into grads for tok_emb via an auxiliary buffer
        grad_tok_emb = np.zeros_like(self.tok_emb)
        for b in range(B):
            for t in range(T):
                grad_tok_emb[idx[b, t]] += dx[b, t]
        # Store into a special slot in grads dict via return (handled in trainer)
        return grad_tok_emb, dx.sum(axis=0)  # pos_emb grad is sum over batch
```

---

### `scratch_transformer/optim.py`
```python
from .xb import np

class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr; self.b1, self.b2 = betas; self.eps = eps; self.wd = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.b1, self.b2
        for k in params.keys():
            g = grads[k]
            if g is None: continue
            if self.wd > 0 and params[k].ndim >= 2:
                params[k] -= self.lr * self.wd * params[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g*g)
            mhat = self.m[k] / (1 - b1**self.t)
            vhat = self.v[k] / (1 - b2**self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
```python
import numpy as np

class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr; self.b1, self.b2 = betas; self.eps = eps; self.wd = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.b1, self.b2
        for k in params.keys():
            g = grads[k]
            if g is None: continue
            # weight decay (decoupled)
            if self.wd > 0 and params[k].ndim >= 2:
                params[k] -= self.lr * self.wd * params[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g*g)
            mhat = self.m[k] / (1 - b1**self.t)
            vhat = self.v[k] / (1 - b2**self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
```

---

### `scratch_transformer/data.py`
```python
import os
import numpy as _np  # file IO on CPU

class ByteDataset:
    def __init__(self, path, seq_len=256, split=0.9, seed=1337):
        with open(path, 'rb') as f:
            data = _np.frombuffer(f.read(), dtype=_np.uint8)
        N = len(data)
        n_train = int(N * split)
        self.train = data[:n_train]
        self.val   = data[n_train:]
        self.seq_len = seq_len
        self.rng = _np.random.default_rng(seed)
    def _sample(self, arr, batch_size):
        L = len(arr) - self.seq_len - 1
        ix = self.rng.integers(0, L, size=(batch_size,))
        x = _np.stack([arr[i:i+self.seq_len] for i in ix], axis=0).astype(_np.int32)
        y = _np.stack([arr[i+1:i+self.seq_len+1] for i in ix], axis=0).astype(_np.int32)
        return x, y
    def get_batch(self, split, batch_size):
        arr = self.train if split == 'train' else self.val
        return self._sample(arr, batch_size)
```python
import os, numpy as np

class ByteDataset:
    def __init__(self, path, seq_len=256, split=0.9, seed=1337):
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        N = len(data)
        n_train = int(N * split)
        self.train = data[:n_train]
        self.val   = data[n_train:]
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
    def _sample(self, arr, batch_size):
        L = len(arr) - self.seq_len - 1
        ix = self.rng.integers(0, L, size=(batch_size,))
        x = np.stack([arr[i:i+self.seq_len] for i in ix], axis=0).astype(np.int32)
        y = np.stack([arr[i+1:i+self.seq_len+1] for i in ix], axis=0).astype(np.int32)
        return x, y
    def get_batch(self, split, batch_size):
        arr = self.train if split == 'train' else self.val
        return self._sample(arr, batch_size)
```

---

### `scratch_transformer/trainer.py`
```python
from .utils import softmax, count_params
from .xb import np

class Trainer:
    def __init__(self, model, optimizer, vocab_size, grad_clip=None):
        self.model = model
        self.opt = optimizer
        self.vocab_size = vocab_size
        self.grad_clip = grad_clip
    def step(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        logits, cache = self.model.forward(x)
        B, T, V = logits.shape
        logits_2d = logits.reshape(B*T, V)
        y_flat = y.reshape(B*T)
        probs = softmax(logits_2d, axis=-1)
        loss = -np.log(probs[np.arange(B*T), y_flat] + 1e-9).mean()
        dlogits = probs
        dlogits[np.arange(B*T), y_flat] -= 1.0
        dlogits = dlogits.reshape(B, T, V) / (B*T)
        self.model.zero_grads()
        grad_tok_emb, grad_pos_emb = self.model.backward(dlogits, cache)
        grads = self.model.grads()
        grads["tok_emb"] = grad_tok_emb
        grads["pos_emb"] = grad_pos_emb
        if self.grad_clip is not None:
            total = 1e-8
            for g in grads.values():
                total += float((g*g).sum())
            norm = np.sqrt(total)
            if norm > self.grad_clip:
                scale = self.grad_clip / (norm + 1e-8)
                for k in grads.keys():
                    grads[k] *= scale
        self.opt.step(self.model.parameters(), grads)
        return float(loss)
```python
import numpy as np
from .utils import softmax, count_params

class Trainer:
    def __init__(self, model, optimizer, vocab_size, grad_clip=None):
        self.model = model
        self.opt = optimizer
        self.vocab_size = vocab_size
        self.grad_clip = grad_clip
    def step(self, x, y):
        # forward
        logits, cache = self.model.forward(x)
        B, T, V = logits.shape
        # cross-entropy loss
        logits_2d = logits.reshape(B*T, V)
        y_flat = y.reshape(B*T)
        probs = softmax(logits_2d, axis=-1)
        loss = -np.log(probs[np.arange(B*T), y_flat] + 1e-9).mean()
        # backward into logits
        dlogits = probs
        dlogits[np.arange(B*T), y_flat] -= 1.0
        dlogits = dlogits.reshape(B, T, V) / (B*T)
        # zero grads and backprop
        self.model.zero_grads()
        grad_tok_emb, grad_pos_emb = self.model.backward(dlogits, cache)
        # collect grads
        grads = self.model.grads()
        grads["tok_emb"] = grad_tok_emb
        grads["pos_emb"] = grad_pos_emb
        # gradient clipping (by global norm)
        if self.grad_clip is not None:
            total = 1e-8
            for g in grads.values():
                total += float((g*g).sum())
            norm = np.sqrt(total)
            if norm > self.grad_clip:
                scale = self.grad_clip / (norm + 1e-8)
                for k in grads.keys():
                    grads[k] *= scale
        # update
        self.opt.step(self.model.parameters(), grads)
        return float(loss)
```

---

### `scratch_transformer/config.py`
```python
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 256
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 1024  # typically 4*d_model
    lr: float = 3e-4
    weight_decay: float = 0.01
    steps: int = 5000
    batch_size: int = 32
    grad_clip: float = 1.0

PRESETS = {
    "mini2p5m": dict(d_model=256, n_heads=4, n_layers=3, d_ff=1024),
    "small5m":  dict(d_model=320, n_heads=5, n_layers=4, d_ff=1280),
}
```

---

### `scratch_transformer/attention_only.py`
```python
from .xb import np
from .utils import softmax

def attention_demo(seq_len=6, d_model=8, n_heads=2, seed=0):
    rng = np.random.default_rng(seed)
    d_head = d_model // n_heads
    x = rng.normal(size=(1, seq_len, d_model)).astype(np.float32)
    Wq = rng.normal(scale=0.2, size=(d_model, d_model)).astype(np.float32)
    Wk = rng.normal(scale=0.2, size=(d_model, d_model)).astype(np.float32)
    Wv = rng.normal(scale=0.2, size=(d_model, d_model)).astype(np.float32)
    Q = x @ Wq; K = x @ Wk; V = x @ Wv
    def split(t):
        return t.reshape(1, seq_len, n_heads, d_head).transpose(0,2,1,3).reshape(n_heads, seq_len, d_head)
    Qh, Kh, Vh = split(Q), split(K), split(V)
    scores = Qh @ Kh.transpose(0,2,1) / np.sqrt(d_head)
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), 1)
    scores[:, mask] = -1e9
    P = softmax(scores, axis=-1)
    Ah = P @ Vh
    print("Attention weights per head (rows: query positions, cols: key positions):
", (P if hasattr(P, "get") else P))
    return P, Ah

if __name__ == "__main__":
    attention_demo()
```python
import numpy as np
from .utils import softmax

def attention_demo(seq_len=6, d_model=8, n_heads=2, seed=0):
    rng = np.random.default_rng(seed)
    d_head = d_model // n_heads
    x = rng.normal(size=(1, seq_len, d_model)).astype(np.float32)
    Wq = rng.normal(scale=0.2, size=(d_model, d_model)).astype(np.float32)
    Wk = rng.normal(scale=0.2, size=(d_model, d_model)).astype(np.float32)
    Wv = rng.normal(scale=0.2, size=(d_model, d_model)).astype(np.float32)
    Q = x @ Wq; K = x @ Wk; V = x @ Wv
    def split(t):
        return t.reshape(1, seq_len, n_heads, d_head).transpose(0,2,1,3).reshape(n_heads, seq_len, d_head)
    Qh, Kh, Vh = split(Q), split(K), split(V)
    scores = Qh @ Kh.transpose(0,2,1) / np.sqrt(d_head)
    # simple causal mask
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), 1)
    scores[:, mask] = -1e9
    P = softmax(scores, axis=-1)
    Ah = P @ Vh
    print("Attention weights per head (rows: query positions, cols: key positions):\n", P)
    return P, Ah

if __name__ == "__main__":
    attention_demo()
```

---

### `experiments/tinyshakespeare.yaml`
```yaml
# Default experiment config for CPU training on TinyShakespeare
size: mini2p5m   # or: small5m
steps: 6000
batch_size: 32
seq_len: 256
lr: 3.0e-4
weight_decay: 0.01
grad_clip: 1.0
print_every: 100
```

---

### `experiments/run_tinyshakespeare.py`
```python
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='experiments/tinyshakespeare.yaml')
parser.add_argument('--size', type=str, default=None, help='override model size preset')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda'], help='cpu or cuda (requires CuPy & NVIDIA CUDA)')
parser.add_argument('--save_every', type=int, default=0, help='save checkpoint every N steps (0 = only at end)')
parser.add_argument('--out', type=str, default='checkpoints/last.pkl')
args, _ = parser.parse_known_args()

# Set backend before importing the rest
os.environ['SCRATCH_DEVICE'] = args.device

import numpy as _np
from scratch_transformer.config import Config, PRESETS
from scratch_transformer.data import ByteDataset
from scratch_transformer.model import Transformer
from scratch_transformer.optim import AdamW
from scratch_transformer.trainer import Trainer
from scratch_transformer.utils import Timer, count_params
from scratch_transformer.checkpoint import save_params

# crude YAML parse (no dependency)
cfg_txt = open(args.config).read().strip().splitlines()
raw = {}
for line in cfg_txt:
    line = line.strip()
    if not line or line.startswith('#'): continue
    k, v = [x.strip() for x in line.split(':', 1)]
    raw[k] = v

preset_name = args.size or raw.get('size', 'mini2p5m')
preset = PRESETS[preset_name]
steps = int(raw.get('steps', 5000))
bs = int(raw.get('batch_size', 32))
seq = int(raw.get('seq_len', 256))
lr = float(raw.get('lr', 3e-4))
wd = float(raw.get('weight_decay', 0.01))
clip = float(raw.get('grad_clip', 1.0))
print_every = int(raw.get('print_every', 100))

cfg = Config(**{**preset, 'max_seq_len': seq})

# data
os.makedirs('data', exist_ok=True)
train_path = 'data/tinyshakespeare.txt'
assert os.path.exists(train_path), "Run scripts/download_tinyshakespeare.py first"
D = ByteDataset(train_path, seq_len=seq)

# model
from scratch_transformer.xb import np  # after SCRATCH_DEVICE set
model = Transformer(
    vocab_size=cfg.vocab_size,
    d_model=cfg.d_model,
    n_heads=cfg.n_heads,
    n_layers=cfg.n_layers,
    d_ff=cfg.d_ff,
    max_seq_len=cfg.max_seq_len,
)

opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
trainer = Trainer(model, opt, cfg.vocab_size, grad_clip=clip)

print(f"Model params: {count_params(model.parameters()):,}")

# train
os.makedirs('checkpoints', exist_ok=True)
timer = Timer()
for step in range(1, steps+1):
    x, y = D.get_batch('train', batch_size=bs)
    loss = trainer.step(x, y)
    if step % print_every == 0:
        print(f"step {step:6d} | loss {loss:.4f}")
    if args.save_every and (step % args.save_every == 0):
        save_params(model.parameters(), args.out)
        print(f"[ckpt] saved at step {step} → {args.out}")

# final save
save_params(model.parameters(), args.out)
print("Done. Saved:", args.out)
```python
import argparse, json, os
import numpy as np
from scratch_transformer.config import Config, PRESETS
from scratch_transformer.data import ByteDataset
from scratch_transformer.model import Transformer
from scratch_transformer.optim import AdamW
from scratch_transformer.trainer import Trainer
from scratch_transformer.utils import Timer, count_params

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='experiments/tinyshakespeare.yaml')
parser.add_argument('--size', type=str, default=None, help='override model size preset')
args = parser.parse_args()

# crude YAML parse (no dependency): key: value per line
cfg_txt = open(args.config).read().strip().splitlines()
raw = {}
for line in cfg_txt:
    line = line.strip()
    if not line or line.startswith('#'): continue
    k, v = [x.strip() for x in line.split(':', 1)]
    raw[k] = v

preset_name = args.size or raw.get('size', 'mini2p5m')
preset = PRESETS[preset_name]
steps = int(raw.get('steps', 5000))
bs = int(raw.get('batch_size', 32))
seq = int(raw.get('seq_len', 256))
lr = float(raw.get('lr', 3e-4))
wd = float(raw.get('weight_decay', 0.01))
clip = float(raw.get('grad_clip', 1.0))

cfg = Config(**{**preset, 'max_seq_len': seq})

# data
os.makedirs('data', exist_ok=True)
train_path = 'data/tinyshakespeare.txt'
assert os.path.exists(train_path), "Run scripts/download_tinyshakespeare.py first"

D = ByteDataset(train_path, seq_len=seq)

# model
rng = np.random.default_rng(42)
model = Transformer(
    vocab_size=cfg.vocab_size,
    d_model=cfg.d_model,
    n_heads=cfg.n_heads,
    n_layers=cfg.n_layers,
    d_ff=cfg.d_ff,
    max_seq_len=cfg.max_seq_len,
    rng=rng,
)

opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
trainer = Trainer(model, opt, cfg.vocab_size, grad_clip=clip)

print(f"Model params: {count_params(model.parameters()):,}")

# train
timer = Timer()
for step in range(1, steps+1):
    x, y = D.get_batch('train', batch_size=bs)
    loss = trainer.step(x, y)
    if step % int(raw.get('print_every', 100)) == 0:
        print(f"step {step:6d} | loss {loss:.4f}")

print("Done.")
```

---

### `scripts/download_tinyshakespeare.py`
```python
import os, urllib.request
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
os.makedirs('data', exist_ok=True)
path = 'data/tinyshakespeare.txt'
if not os.path.exists(path):
    print('Downloading TinyShakespeare...')
    urllib.request.urlretrieve(url, path)
    print('Saved to', path)
else:
    print('Already exists:', path)
```

---

### `tests/test_shapes.py`
```python
from scratch_transformer.model import Transformer
from scratch_transformer.xb import np

B, T = 2, 16
vocab = 256
m = Transformer(vocab, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=T)
idx = np.random.randint(0, vocab, size=(B, T), dtype=np.int32)
logits, cache = m.forward(idx)
assert logits.shape == (B, T, vocab)
print('OK: forward shapes')
```python
import numpy as np
from scratch_transformer.model import Transformer

B, T = 2, 16
vocab = 256
m = Transformer(vocab, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=T)
idx = np.random.randint(0, vocab, size=(B, T), dtype=np.int32)
logits, cache = m.forward(idx)
assert logits.shape == (B, T, vocab)
print('OK: forward shapes')
```

---

### `scratch_transformer/xb.py`
```python
"""Backend selector: NumPy (CPU) or CuPy (CUDA) via env var SCRATCH_DEVICE.
Usage:
  export SCRATCH_DEVICE=cuda  # if CuPy + CUDA available
  export SCRATCH_DEVICE=cpu   # default
"""
import os
_device = os.getenv("SCRATCH_DEVICE", "cpu").lower()
try:
    if _device == "cuda":
        import cupy as np
    else:
        import numpy as np
except Exception:
    import numpy as np  # graceful fallback

# utilities to move to/from CPU when needed
_def_name = getattr(np, "__name__", "numpy")

def to_cpu(x):
    if _def_name.startswith("cupy"):
        import cupy
        return cupy.asnumpy(x)
    return x
```

---

### `scratch_transformer/checkpoint.py`
```python
import os, pickle
from .xb import np, to_cpu

def save_params(params: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cpu_params = {k: to_cpu(v) for k, v in params.items()}
    with open(path, 'wb') as f:
        pickle.dump(cpu_params, f)

def load_params(path: str, device: str = 'cpu') -> dict:
    with open(path, 'rb') as f:
        cpu_params = pickle.load(f)
    if device == 'cuda':
        import cupy
        return {k: cupy.asarray(v) for k, v in cpu_params.items()}
    else:
        return cpu_params

def assign_params_to_model(model, params: dict):
    mp = model.parameters()
    for k, v in params.items():
        mp[k][...] = v
```

---

### `scripts/sample.py`
```python
import argparse, os
os.environ.setdefault('SCRATCH_DEVICE', 'cpu')  # sampling is light; CPU default

from scratch_transformer.config import PRESETS, Config
from scratch_transformer.model import Transformer
from scratch_transformer.checkpoint import load_params, assign_params_to_model
from scratch_transformer.xb import np
from scratch_transformer.utils import softmax

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoints/last.pkl')
parser.add_argument('--size', type=str, default='mini2p5m')
parser.add_argument('--seq_len', type=int, default=256)
parser.add_argument('--max_new', type=int, default=400)
parser.add_argument('--start', type=str, default='To be, or not to be')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda'])
args = parser.parse_args()

os.environ['SCRATCH_DEVICE'] = args.device

preset = PRESETS[args.size]
cfg = Config(**{**preset, 'max_seq_len': args.seq_len})

model = Transformer(cfg.vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.max_seq_len)
params = load_params(args.checkpoint, device=args.device)
assign_params_to_model(model, params)

# byte-level encoding
context = bytearray(args.start.encode('utf-8', errors='ignore'))

for _ in range(args.max_new):
    idx = np.asarray([context[-cfg.max_seq_len:] + bytearray(max(0, cfg.max_seq_len - len(context)))], dtype=np.int32)
    # right-pad with zeros if needed; we only look at the tail positions actually filled
    logits, _ = model.forward(idx)
    last = logits[0, len(context)-1 if len(context)>0 else 0]
    if args.temperature != 1.0:
        last = last / args.temperature
    if args.top_k and args.top_k > 0:
        topk_idx = np.argpartition(last, -args.top_k)[-args.top_k:]
        mask = np.ones_like(last, dtype=bool)
        mask[topk_idx] = False
        last = last.copy(); last[mask] = -1e9
    probs = softmax(last[None, :], axis=-1)[0]
    # greedy for simplicity: argmax
    token = int(np.argmax(probs))
    context.append(token)

print(bytes(context).decode('utf-8', errors='ignore'))
```

---

### `tests/gradcheck.py`
```python
"""Finite-difference gradient checks for a couple of layers.
Usage:
  python -m tests.gradcheck
"""
import os
os.environ['SCRATCH_DEVICE'] = 'cpu'  # numeric diff on CPU for stability

from scratch_transformer.xb import np
from scratch_transformer.layers import Linear, LayerNorm

EPS = 1e-3
RTOL, ATOL = 1e-2, 1e-3


def check_linear():
    B, T, Din, Dout = 2, 3, 4, 5
    x = np.random.randn(B, T, Din).astype(np.float32)
    layer = Linear(Din, Dout)
    # forward
    y = layer.forward(x)
    loss = (y*y).mean()
    # backward analytic
    grad_y = (2.0 / y.size) * y
    dx = layer.backward(grad_y)
    # numeric W grad
    W_num = np.zeros_like(layer.W)
    for i in range(Din):
        for j in range(Dout):
            layer.W[i, j] += EPS
            lp = (layer.forward(x)**2).mean()
            layer.W[i, j] -= 2*EPS
            lm = (layer.forward(x)**2).mean()
            layer.W[i, j] += EPS
            W_num[i, j] = (lp - lm) / (2*EPS)
    # compare
    rel = np.allclose(W_num, layer.grads['W'], rtol=RTOL, atol=ATOL)
    print('Linear dW gradcheck:', 'OK' if rel else 'FAIL', np.max(np.abs(W_num - layer.grads['W'])))


def check_layernorm():
    B, T, D = 2, 3, 6
    x = np.random.randn(B, T, D).astype(np.float32)
    ln = LayerNorm(D)
    y = ln.forward(x)
    loss = (y*y).mean()
    grad_y = (2.0 / y.size) * y
    dx = ln.backward(grad_y)
    # numeric gamma grad
    gnum = np.zeros_like(ln.gamma)
    for i in range(D):
        ln.gamma[i] += EPS
        lp = (ln.forward(x)**2).mean()
        ln.gamma[i] -= 2*EPS
        lm = (ln.forward(x)**2).mean()
        ln.gamma[i] += EPS
        gnum[i] = (lp - lm) / (2*EPS)
    ok = np.allclose(gnum, ln.grads['gamma'], rtol=RTOL, atol=ATOL)
    print('LayerNorm dgamma gradcheck:', 'OK' if ok else 'FAIL', np.max(np.abs(gnum - ln.grads['gamma'])))

if __name__ == '__main__':
    check_linear()
    check_layernorm()
```

---

## Notes & gotchas
- **CuPy toggle (optional GPU)**: set `--device cuda` (and install `cupy-cuda12x` matching your CUDA). This swaps NumPy for CuPy via `SCRATCH_DEVICE`. On **Apple M‑series**, CuPy CUDA is not applicable—run CPU.
- **Checkpoints**: training now saves to `checkpoints/last.pkl` (CPU tensors inside), so you can sample later with `python scripts/sample.py --checkpoint checkpoints/last.pkl --start "Once more"`.
- **Grad checks**: `python -m tests.gradcheck` sanity‑checks our manual backward for `Linear` and `LayerNorm`.
- **Sampling**: greedy decode for simplicity; you can add top‑p sampling if you want spicier text.
- **TinyShakespeare** is for pedagogy; don’t expect coherent long‑form.

