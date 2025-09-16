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