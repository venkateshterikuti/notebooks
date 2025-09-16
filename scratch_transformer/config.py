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
    "small5m": dict(d_model=320, n_heads=5, n_layers=4, d_ff=1280),
}
