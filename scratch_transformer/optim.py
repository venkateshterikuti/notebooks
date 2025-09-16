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





 