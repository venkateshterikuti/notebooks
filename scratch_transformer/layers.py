from .xb import np
from .utils import dgelu, gelu, softmax, softmax_backward, xavier_uniform


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
        grad_y_2d = grad_y.reshape(-1, grad_y.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        self.grads["W"][:] = x_2d.T @ grad_y_2d
        self.grads["b"][:] = grad_y.sum(axis=tuple(range(grad_y.ndim - 1)))
        grad_x = grad_y @ self.W.T
        return grad_x


class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.gamma = np.ones((d,), dtype=np.float32)
        self.beta = np.zeros((d,), dtype=np.float32)
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
        axes = tuple(range(grad_y.ndim - 1))
        self.grads["gamma"][:] = (grad_y * xhat).sum(axis=axes)
        self.grads["beta"][:] = grad_y.sum(axis=axes)
        dxhat = grad_y * self.gamma
        N = x.shape[-1]
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
        return x.reshape(B, T, H, Dh).transpose(0, 2, 1, 3).reshape(B * H, T, Dh)

    def _merge_heads(self, xh, B):
        H, Dh = self.n_heads, self.d_head
        T = xh.shape[1]
        return xh.reshape(B, H, T, Dh).transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

    def forward(self, x, attn_mask):
        B, T, _ = x.shape
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)
        scale = 1.0 / np.sqrt(self.d_head)
        scores = (Qh @ Kh.transpose(0, 2, 1)) * scale
        mask = None
        if attn_mask is not None:
            mask = np.asarray(attn_mask, dtype=bool)[None, :, :]
            neg_inf = np.array(-1e9, dtype=scores.dtype)
            scores = np.where(mask, neg_inf, scores)
        P = softmax(scores, axis=-1)
        Ah = P @ Vh
        A = self._merge_heads(Ah, B)
        out = A @ self.Wo
        self.cache = (x, Qh, Kh, Vh, P, mask, A)
        return out

    def backward(self, grad_y):
        x, Qh, Kh, Vh, P, mask, A = self.cache
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        self.grads["Wo"][:] = A.reshape(-1, D).T @ grad_y.reshape(-1, D)
        dA = grad_y @ self.Wo.T
        dAh = dA.reshape(B, T, H, Dh).transpose(0, 2, 1, 3).reshape(B * H, T, Dh)
        dP = dAh @ Vh.transpose(0, 2, 1)
        dVh = P.transpose(0, 2, 1) @ dAh
        dScores = softmax_backward(dP, P)
        if mask is not None:
            dScores = np.where(mask, np.array(0.0, dtype=dScores.dtype), dScores)
        scale = 1.0 / np.sqrt(Dh)
        dQh = (dScores @ Kh) * scale
        dKh = (dScores.transpose(0, 2, 1) @ Qh) * scale
        dQ = dQh.reshape(B, H, T, Dh).transpose(0, 2, 1, 3).reshape(B, T, D)
        dK = dKh.reshape(B, H, T, Dh).transpose(0, 2, 1, 3).reshape(B, T, D)
        dV = dVh.reshape(B, H, T, Dh).transpose(0, 2, 1, 3).reshape(B, T, D)
        x_2d = x.reshape(-1, D)
        self.grads["Wq"][:] = x_2d.T @ dQ.reshape(-1, D)
        self.grads["Wk"][:] = x_2d.T @ dK.reshape(-1, D)
        self.grads["Wv"][:] = x_2d.T @ dV.reshape(-1, D)
        dx = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        return dx
