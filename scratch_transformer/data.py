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