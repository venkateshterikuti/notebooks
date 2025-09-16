from .layers import LayerNorm, Linear, MLP, MultiHeadSelfAttention
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
        x1 = x + a
        h2 = self.ln2.forward(x1)
        m = self.mlp.forward(h2)
        out = x1 + m
        return out

    def backward(self, grad_out):
        grad_x1 = grad_out
        grad_h2 = self.mlp.backward(grad_out)
        grad_x1 = grad_x1 + self.ln2.backward(grad_h2)
        grad_attn_out = grad_x1
        grad_ln1_out = self.attn.backward(grad_attn_out)
        grad_x = grad_x1 + self.ln1.backward(grad_ln1_out)
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
        params = {
            "tok_emb": self.tok_emb,
            "pos_emb": self.pos_emb,
            "lm_head.W": self.lm_head.W,
            "lm_head.b": self.lm_head.b,
        }
        for i, blk in enumerate(self.blocks):
            params.update(
                {
                    f"blocks.{i}.ln1.gamma": blk.ln1.gamma,
                    f"blocks.{i}.ln1.beta": blk.ln1.beta,
                    f"blocks.{i}.attn.Wq": blk.attn.Wq,
                    f"blocks.{i}.attn.Wk": blk.attn.Wk,
                    f"blocks.{i}.attn.Wv": blk.attn.Wv,
                    f"blocks.{i}.attn.Wo": blk.attn.Wo,
                    f"blocks.{i}.ln2.gamma": blk.ln2.gamma,
                    f"blocks.{i}.ln2.beta": blk.ln2.beta,
                    f"blocks.{i}.fc1.W": blk.mlp.fc1.W,
                    f"blocks.{i}.fc1.b": blk.mlp.fc1.b,
                    f"blocks.{i}.fc2.W": blk.mlp.fc2.W,
                    f"blocks.{i}.fc2.b": blk.mlp.fc2.b,
                }
            )
        return params

    def grads(self):
        grads = {
            "tok_emb": np.zeros_like(self.tok_emb),
            "pos_emb": np.zeros_like(self.pos_emb),
            "lm_head.W": self.lm_head.grads["W"],
            "lm_head.b": self.lm_head.grads["b"],
        }
        for i, blk in enumerate(self.blocks):
            grads.update(
                {
                    f"blocks.{i}.ln1.gamma": blk.ln1.grads["gamma"],
                    f"blocks.{i}.ln1.beta": blk.ln1.grads["beta"],
                    f"blocks.{i}.attn.Wq": blk.attn.grads["Wq"],
                    f"blocks.{i}.attn.Wk": blk.attn.grads["Wk"],
                    f"blocks.{i}.attn.Wv": blk.attn.grads["Wv"],
                    f"blocks.{i}.attn.Wo": blk.attn.grads["Wo"],
                    f"blocks.{i}.ln2.gamma": blk.ln2.grads["gamma"],
                    f"blocks.{i}.ln2.beta": blk.ln2.grads["beta"],
                    f"blocks.{i}.fc1.W": blk.mlp.fc1.grads["W"],
                    f"blocks.{i}.fc1.b": blk.mlp.fc1.grads["b"],
                    f"blocks.{i}.fc2.W": blk.mlp.fc2.grads["W"],
                    f"blocks.{i}.fc2.b": blk.mlp.fc2.grads["b"],
                }
            )
        return grads

    def zero_grads(self):
        self.lm_head.grads["W"].fill(0.0)
        self.lm_head.grads["b"].fill(0.0)
        for blk in self.blocks:
            blk.ln1.grads["gamma"].fill(0.0)
            blk.ln1.grads["beta"].fill(0.0)
            blk.attn.grads["Wq"].fill(0.0)
            blk.attn.grads["Wk"].fill(0.0)
            blk.attn.grads["Wv"].fill(0.0)
            blk.attn.grads["Wo"].fill(0.0)
            blk.ln2.grads["gamma"].fill(0.0)
            blk.ln2.grads["beta"].fill(0.0)
            blk.mlp.fc1.grads["W"].fill(0.0)
            blk.mlp.fc1.grads["b"].fill(0.0)
            blk.mlp.fc2.grads["W"].fill(0.0)
            blk.mlp.fc2.grads["b"].fill(0.0)

    def forward(self, idx):
        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        tok = self.tok_emb[idx]
        pos = self.pos_emb[:T][None, :, :]
        x = tok + pos
        mask = self.mask[:T, :T]
        for blk in self.blocks:
            x = blk.forward(x, mask)
        x = self.ln_f.forward(x)
        logits = self.lm_head.forward(x)
        cache = (idx, T)
        return logits, cache

    def backward(self, dlogits, cache):
        idx, T = cache
        dx = self.lm_head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        grad_tok_emb = np.zeros_like(self.tok_emb)
        B = idx.shape[0]
        for b in range(B):
            for t in range(T):
                grad_tok_emb[idx[b, t]] += dx[b, t]
        grad_pos_emb = np.zeros_like(self.pos_emb)
        grad_pos_emb[:T] = dx.sum(axis=0)
        return grad_tok_emb, grad_pos_emb
