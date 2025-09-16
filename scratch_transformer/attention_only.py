from .xb import np
from .utils import softmax


def attention_demo(seq_len=6, d_model=8, n_heads=2, seed=0):
    rng = np.random.default_rng(seed)
    d_head = d_model // n_heads
    x = rng.normal(size=(1, seq_len, d_model)).astype(np.float32)
    Wq = rng.normal(scale=0.2, size=(d_model, d_model)).astype(np.float32)
    Wk = rng.normal(scale=0.2, size=(d_model, d_model)).astype(np.float32)
    Wv = rng.normal(scale=0.2, size=(d_model, d_model)).astype(np.float32)
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    def split(t):
        return t.reshape(1, seq_len, n_heads, d_head).transpose(0, 2, 1, 3).reshape(n_heads, seq_len, d_head)

    Qh, Kh, Vh = split(Q), split(K), split(V)
    scores = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(d_head)
    mask = np.asarray(np.triu(np.ones((seq_len, seq_len), dtype=bool), 1))
    neg_inf = np.array(-1e9, dtype=scores.dtype)
    scores = np.where(mask[None, :, :], neg_inf, scores)
    P = softmax(scores, axis=-1)
    Ah = P @ Vh
    try:
        import cupy as _cp
        P_print = _cp.asnumpy(P) if isinstance(P, _cp.ndarray) else P
    except Exception:
        P_print = P
    print("Attention weights per head (rows: query positions, cols: key positions):\n", P_print)
    return P, Ah


if __name__ == "__main__":
    attention_demo()
