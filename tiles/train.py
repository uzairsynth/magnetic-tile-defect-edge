"""Train a tile classifier and keep the checkpoint with the best validation macro-F1.

    python -m tiles.train --arch mobilenet_v3_small
    python -m tiles.train --arch mobilenet_v3_small --split random   # leakage check only

The test split is scored once with the selected checkpoint; it never influences
training, early stopping or model choice.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tiles import data  # noqa: E402
from tiles.metrics import classification_summary  # noqa: E402
from tiles.model import build_model, count_parameters  # noqa: E402

DEFAULT_LR = {"mobilenet_v3_small": 1e-3}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x), 1).numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_one_epoch(model, loader, loss_fn, optimizer) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss) * len(y)
        count += len(y)
    return total / count


def plot_history(history: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.plot(history["epoch"], history["train_loss"], label="Train loss", color="tab:blue")
    ax1.set(xlabel="Epoch", ylabel="Loss", title=title)
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_macro_f1"], label="Val macro-F1", color="tab:orange")
    ax2.set_ylabel("Macro-F1")
    ax2.set_ylim(0, 1)
    fig.legend(loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", default="mobilenet_v3_small", choices=["mobilenet_v3_small"])
    p.add_argument("--split", default="grouped", choices=["grouped", "random"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--img-size", type=int, default=data.IMG_SIZE)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-minutes", type=float, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--data-dir", type=Path, default=data.DATA_DIR)
    p.add_argument("--splits", type=Path, default=data.SPLITS_CSV)
    p.add_argument("--out-dir", type=Path, default=data.ROOT)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    name = args.arch + ("_random_split" if args.split == "random" else "")
    models_dir, reports_dir = args.out_dir / "models", args.out_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    frame = data.load_splits(args.splits)
    if args.split == "random":
        frame["split"] = data.random_splits(frame, args.seed)
    parts = {s: frame[frame["split"] == s] for s in ("train", "val", "test")}

    def loader(split, train):
        transform = data.train_transform(args.img_size) if train else data.eval_transform(args.img_size)
        ds = data.TileDataset(parts[split], args.data_dir, transform)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=0, generator=torch.Generator().manual_seed(args.seed))

    train_loader, val_loader, test_loader = loader("train", True), loader("val", False), loader("test", False)
    model = build_model(args.arch, len(data.CLASSES), pretrained=not args.no_pretrained)
    loss_fn = nn.CrossEntropyLoss(weight=data.class_weights(parts["train"]["y"]), label_smoothing=0.05)
    lr = args.lr or DEFAULT_LR[args.arch]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"{name}: {count_parameters(model):,} parameters | train {len(parts['train'])} / val {len(parts['val'])} / test {len(parts['test'])}")
    started, best_f1, best_epoch, stale, history = time.perf_counter(), -1.0, 0, 0, []
    ckpt_path = models_dir / f"{name}.pt"
    stop_reason = "max epochs"
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        scheduler.step()
        val_probs, val_y = predict(model, val_loader)
        val = classification_summary(val_y, val_probs, data.CLASSES)
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "train_loss": loss, "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"], "seconds": elapsed})
        print(f"epoch {epoch:2d} loss {loss:.4f} val macro-F1 {val['macro_f1']:.3f} acc {val['accuracy']:.3f} ({elapsed:.0f}s)")

        if val["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, stale = val["macro_f1"], epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": args.arch,
                    "classes": data.CLASSES,
                    "img_size": args.img_size,
                    "epoch": epoch,
                    "split": args.split,
                    "val_macro_f1": best_f1,
                },
                ckpt_path,
            )
        else:
            stale += 1
        if stale >= args.patience:
            stop_reason = f"no val improvement for {args.patience} epochs"
            break
        if elapsed > args.max_minutes * 60:
            stop_reason = f"time budget ({args.max_minutes} min)"
            break

    best = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"])
    val_probs, val_y = predict(model, val_loader)
    test_probs, test_y = predict(model, test_loader)

    hist = pd.DataFrame(history)
    hist.to_csv(reports_dir / f"history_{name}.csv", index=False)
    plot_history(hist, reports_dir / "figures" / f"training_{name}.png", f"{name} (best epoch {best_epoch})")

    result = {
        "name": name,
        "arch": args.arch,
        "split": args.split,
        "parameters": count_parameters(model),
        "pretrained": not args.no_pretrained,
        "lr": lr,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "train_seconds": round(time.perf_counter() - started, 1),
        "split_sizes": {k: int(len(v)) for k, v in parts.items()},
        "val": classification_summary(val_y, val_probs, data.CLASSES),
        "test": classification_summary(test_y, test_probs, data.CLASSES),
    }
    (reports_dir / f"train_{name}.json").write_text(json.dumps(result, indent=2))
    t = result["test"]
    print(f"{name}: test macro-F1 {t['macro_f1']:.3f}, accuracy {t['accuracy']:.3f}, defect recall {t['defect_detection']['recall']:.3f}")
    return result


if __name__ == "__main__":
    main()
