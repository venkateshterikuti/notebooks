from .utils import softmax
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
        logits_2d = logits.reshape(B * T, V)
        y_flat = y.reshape(B * T)
        probs = softmax(logits_2d, axis=-1)
        loss = -np.log(probs[np.arange(B * T), y_flat] + 1e-9).mean()
        dlogits = probs
        dlogits[np.arange(B * T), y_flat] -= 1.0
        dlogits = dlogits.reshape(B, T, V) / (B * T)
        self.model.zero_grads()
        grad_tok_emb, grad_pos_emb = self.model.backward(dlogits, cache)
        grads = self.model.grads()
        grads["tok_emb"] = grad_tok_emb
        grads["pos_emb"] = grad_pos_emb
        if self.grad_clip is not None:
            total = 1e-8
            for g in grads.values():
                total += float((g * g).sum())
            norm = np.sqrt(total)
            if norm > self.grad_clip:
                scale = self.grad_clip / (norm + 1e-8)
                for k in grads.keys():
                    grads[k] *= scale
        self.opt.step(self.model.parameters(), grads)
        return float(loss)
