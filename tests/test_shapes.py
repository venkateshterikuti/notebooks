### `tests/test_shapes.py`
from scratch_transformer.model import Transformer
from scratch_transformer.xb import np

B, T = 2, 16
vocab = 256
m = Transformer(vocab, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=T)
idx = np.random.randint(0, vocab, size=(B, T), dtype=np.int32)
logits, cache = m.forward(idx)
assert logits.shape == (B, T, vocab)
print('OK: forward shapes')



 

