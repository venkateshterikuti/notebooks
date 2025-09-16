"""
Compute cross-entropy and perplexity on the validation split.

Example:
  python scripts/eval_ppl.py --checkpoint checkpoints/last.pkl --size mini2p5m \
      --seq_len 256 --batch_size 32 --max_tokens 100000 --device cpu
"""
import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoints/last.pkl')
parser.add_argument('--size', type=str, default='mini2p5m')
parser.add_argument('--seq_len', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_tokens', type=int, default=0, help='0 = evaluate all val tokens (may be slow)')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda'])
args = parser.parse_args()

os.environ['SCRATCH_DEVICE'] = args.device

from scratch_transformer.config import PRESETS, Config
from scratch_transformer.model import Transformer
from scratch_transformer.checkpoint import load_params, assign_params_to_model
from scratch_transformer.data import ByteDataset
from scratch_transformer.utils import softmax
from scratch_transformer.xb import np

# config & data
preset = PRESETS[args.size]
cfg = Config(**{**preset, 'max_seq_len': args.seq_len})

D = ByteDataset('data/tinyshakespeare.txt', seq_len=cfg.max_seq_len)
val = D.val  # 1D uint8 array
V = cfg.vocab_size

# model
model = Transformer(V, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.max_seq_len)
params = load_params(args.checkpoint, device=args.device)
assign_params_to_model(model, params)

# sequential, non-overlapping windows across val
stride = cfg.max_seq_len
npos = max(0, len(val) - cfg.max_seq_len - 1)
positions = list(range(0, npos, stride))

def make_batch(pos_list):
    xs, ys = [], []
    for p in pos_list:
        xs.append(val[p:p+cfg.max_seq_len])
        ys.append(val[p+1:p+cfg.max_seq_len+1])
    x = np.asarray(xs, dtype=np.int32)
    y = np.asarray(ys, dtype=np.int32)
    return x, y

total_nll = 0.0
total_tokens = 0
i = 0
while i < len(positions):
    batch_pos = positions[i:i+args.batch_size]
    if not batch_pos:
        break
    x, y = make_batch(batch_pos)
    logits, _ = model.forward(x)             # [B,T,V]
    B, T, Vocab = logits.shape
    probs = softmax(logits.reshape(B*T, Vocab), axis=-1)
    y_flat = y.reshape(B*T)
    # mean NLL over the batch
    nll = -np.log(probs[np.arange(B*T), y_flat] + 1e-9)
    total_nll += float(nll.sum())
    total_tokens += int(B*T)
    i += args.batch_size
    if args.max_tokens and total_tokens >= args.max_tokens:
        break

avg_nll = total_nll / max(1, total_tokens)
ppl = float(np.exp(avg_nll))
print(f"val CE: {avg_nll:.4f} | PPL: {ppl:.2f} | tokens: {total_tokens}")
