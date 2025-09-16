import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="experiments/tinyshakespeare.yaml")
parser.add_argument("--size", type=str, default=None, help="override model size preset")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="cpu or cuda (requires CuPy & NVIDIA CUDA)")
parser.add_argument("--save_every", type=int, default=0, help="save checkpoint every N steps (0 = only at end)")
parser.add_argument("--out", type=str, default="checkpoints/last.pkl")
args, _ = parser.parse_known_args()

os.environ["SCRATCH_DEVICE"] = args.device

from scratch_transformer.checkpoint import save_params  # noqa: E402
from scratch_transformer.config import Config, PRESETS  # noqa: E402
from scratch_transformer.data import ByteDataset  # noqa: E402
from scratch_transformer.model import Transformer  # noqa: E402
from scratch_transformer.optim import AdamW  # noqa: E402
from scratch_transformer.trainer import Trainer  # noqa: E402
from scratch_transformer.utils import Timer, count_params  # noqa: E402

cfg_txt = open(args.config).read().strip().splitlines()
raw = {}
for line in cfg_txt:
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    k, v = [x.strip() for x in line.split(":", 1)]
    raw[k] = v

preset_name = args.size or raw.get("size", "mini2p5m")
preset = PRESETS[preset_name]
steps = int(raw.get("steps", 5000))
bs = int(raw.get("batch_size", 32))
seq = int(raw.get("seq_len", 256))
lr = float(raw.get("lr", 3e-4))
wd = float(raw.get("weight_decay", 0.01))
clip = float(raw.get("grad_clip", 1.0))
print_every = int(raw.get("print_every", 100))

cfg = Config(**{**preset, "max_seq_len": seq})

os.makedirs("data", exist_ok=True)
train_path = "data/tinyshakespeare.txt"
if not os.path.exists(train_path):
    raise FileNotFoundError("Run scripts/download_tinyshakespeare.py first")
D = ByteDataset(train_path, seq_len=seq)

from scratch_transformer.xb import np  # noqa: E402

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

os.makedirs("checkpoints", exist_ok=True)
timer = Timer()
for step in range(1, steps + 1):
    x, y = D.get_batch("train", batch_size=bs)
    loss = trainer.step(x, y)
    if step % print_every == 0:
        print(f"step {step:6d} | loss {loss:.4f}")
    if args.save_every and step % args.save_every == 0:
        save_params(model.parameters(), args.out)
        print(f"[ckpt] saved at step {step} -> {args.out}")

save_params(model.parameters(), args.out)
print("Done. Saved:", args.out)

