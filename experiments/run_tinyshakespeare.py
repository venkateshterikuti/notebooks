import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="experiments/tinyshakespeare.yaml")
parser.add_argument("--size", type=str, default=None, help="override model size preset")
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "cuda"],
    help="cpu or cuda (requires CuPy & NVIDIA CUDA)",
)
parser.add_argument("--save_every", type=int, default=0, help="save checkpoint every N steps (0 = only at end)")
parser.add_argument("--out", type=str, default="checkpoints/last.pkl")
parser.add_argument("--metrics_every", type=int, default=1, help="record metrics every N steps")
parser.add_argument("--metrics_csv", type=str, default=None, help="CSV file to write per-step metrics")
parser.add_argument("--metrics_json", type=str, default=None, help="JSON file to write per-step metrics")
parser.add_argument("--loss_plot", type=str, default=None, help="PNG file for the loss curve plot")
args, _ = parser.parse_known_args()

if args.metrics_every < 1:
    raise ValueError("--metrics_every must be >= 1")

plt = None
_matplotlib_error: Exception | None = None
if args.loss_plot:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure
        _matplotlib_error = exc

os.environ["SCRATCH_DEVICE"] = args.device

from scratch_transformer.checkpoint import save_params  # noqa: E402
from scratch_transformer.config import Config, PRESETS  # noqa: E402
from scratch_transformer.data import ByteDataset  # noqa: E402
from scratch_transformer.model import Transformer  # noqa: E402
from scratch_transformer.optim import AdamW  # noqa: E402
from scratch_transformer.trainer import Trainer  # noqa: E402
from scratch_transformer.utils import Timer, count_params  # noqa: E402


def _ensure_parent(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _dump_metrics_csv(path: Path, history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "elapsed_sec"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _dump_metrics_json(path: Path, history: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    if not history and not metadata:
        return
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "history": history,
        "metadata": metadata,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _save_loss_plot(path: Path, history: List[Dict[str, Any]]) -> None:
    if plt is None:
        if _matplotlib_error is not None:
            print(f"[warn] Matplotlib unavailable ({_matplotlib_error}). Skipping plot: {path}")
        return
    if not history:
        print(f"[plot] No history captured; skipping plot {path}")
        return
    steps = [row["step"] for row in history]
    losses = [row["loss"] for row in history]
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, marker="o", linewidth=1.5, markersize=2)
    plt.title("Training loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


cfg_txt = Path(args.config).read_text().strip().splitlines()
raw: Dict[str, str] = {}
for line in cfg_txt:
    line = line.strip()
    if "#" in line:
        line = line.split("#", 1)[0].strip()
    if not line:
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

param_count = count_params(model.parameters())
opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
trainer = Trainer(model, opt, cfg.vocab_size, grad_clip=clip)

print(f"Model params: {param_count:,}")

os.makedirs("checkpoints", exist_ok=True)
metrics_history: List[Dict[str, Any]] = []
timer = Timer()
last_loss = float("nan")
for step in range(1, steps + 1):
    x, y = D.get_batch("train", batch_size=bs)
    loss = trainer.step(x, y)
    last_loss = float(loss)
    if step % args.metrics_every == 0:
        metrics_history.append(
            {
                "step": step,
                "loss": last_loss,
                "elapsed_sec": timer.elapsed(),
            }
        )
    if step % print_every == 0:
        print(f"step {step:6d} | loss {loss:.4f}")
    if args.save_every and step % args.save_every == 0:
        save_params(model.parameters(), args.out)
        print(f"[ckpt] saved at step {step} -> {args.out}")

save_params(model.parameters(), args.out)
print("Done. Saved:", args.out)

if steps > 0:
    final_elapsed = timer.elapsed()
    if not metrics_history or metrics_history[-1]["step"] != steps:
        metrics_history.append(
            {
                "step": steps,
                "loss": last_loss,
                "elapsed_sec": final_elapsed,
            }
        )
else:
    final_elapsed = timer.elapsed()

metrics_csv_path = _ensure_parent(args.metrics_csv)
if metrics_csv_path is not None:
    _dump_metrics_csv(metrics_csv_path, metrics_history)
    if metrics_csv_path.exists():
        print(f"[metrics] wrote CSV -> {metrics_csv_path}")

metrics_json_path = _ensure_parent(args.metrics_json)
metadata: Dict[str, Any] = {
    "config_file": os.path.abspath(args.config),
    "preset": preset_name,
    "steps": steps,
    "batch_size": bs,
    "seq_len": seq,
    "lr": lr,
    "weight_decay": wd,
    "grad_clip": clip,
    "print_every": print_every,
    "metrics_every": args.metrics_every,
    "device": args.device,
    "save_every": args.save_every,
    "checkpoint": os.path.abspath(args.out),
    "param_count": param_count,
    "dataset": os.path.abspath(train_path),
    "total_train_time_sec": final_elapsed,
}
if metrics_history:
    metadata["final_step"] = metrics_history[-1]["step"]
    metadata["final_loss"] = metrics_history[-1]["loss"]
else:
    metadata["final_step"] = steps
    metadata["final_loss"] = last_loss
metadata["completed_at"] = datetime.utcnow().isoformat() + "Z"

if metrics_json_path is not None:
    _dump_metrics_json(metrics_json_path, metrics_history, metadata)
    if metrics_json_path.exists():
        print(f"[metrics] wrote JSON -> {metrics_json_path}")

loss_plot_path = _ensure_parent(args.loss_plot)
if loss_plot_path is not None:
    _save_loss_plot(loss_plot_path, metrics_history)
    if loss_plot_path.exists():
        print(f"[plot] wrote loss curve -> {loss_plot_path}")
