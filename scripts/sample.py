import argparse
import os

os.environ.setdefault("SCRATCH_DEVICE", "cpu")

from scratch_transformer.checkpoint import assign_params_to_model, load_params
from scratch_transformer.config import Config, PRESETS
from scratch_transformer.model import Transformer
from scratch_transformer.utils import softmax
from scratch_transformer.xb import np

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="checkpoints/last.pkl")
parser.add_argument("--size", type=str, default="mini2p5m")
parser.add_argument("--seq_len", type=int, default=256)
parser.add_argument("--max_new", type=int, default=400)
parser.add_argument("--start", type=str, default="To be, or not to be")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
args = parser.parse_args()

os.environ["SCRATCH_DEVICE"] = args.device

preset = PRESETS[args.size]
cfg = Config(**{**preset, "max_seq_len": args.seq_len})

model = Transformer(cfg.vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.max_seq_len)
params = load_params(args.checkpoint, device=args.device)
assign_params_to_model(model, params)

context = bytearray(args.start.encode("utf-8", errors="ignore"))

for _ in range(args.max_new):
    window = np.zeros((1, cfg.max_seq_len), dtype=np.int32)
    tail = context[-cfg.max_seq_len:]
    if tail:
        arr = np.asarray(list(tail), dtype=np.int32)
        window[0, -len(tail) :] = arr
    logits, _ = model.forward(window)
    last_idx = cfg.max_seq_len - 1
    last = logits[0, last_idx]
    if args.temperature != 1.0:
        last = last / args.temperature
    if args.top_k and args.top_k > 0:
        topk_idx = np.argpartition(last, -args.top_k)[-args.top_k:]
        mask = np.ones_like(last, dtype=bool)
        mask[topk_idx] = False
        last = last.copy()
        last[mask] = -1e9
    probs = softmax(last[None, :], axis=-1)[0]
    token = int(np.argmax(probs))
    context.append(token)

print(bytes(context).decode("utf-8", errors="ignore"))
