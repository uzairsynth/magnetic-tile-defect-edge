"""Score trained checkpoints on the test split and check Grad-CAM against the defect masks.

    python -m tiles.evaluate

Grad-CAM pictures are easy to over-read, so they are measured: for every test
defect image, does the heatmap's hottest pixel land on the labelled defect
(pointing game), and what share of the heat lies on it? Two baselines keep
this honest: a random point, and always pointing at the image centre.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tiles import baseline, data  # noqa: E402
from tiles.gradcam import dilate, energy_inside, grad_cam, pointing_hit  # noqa: E402
from tiles.metrics import (  # noqa: E402
    choose_defect_threshold,
    classification_summary,
    defect_alarm,
    defect_recall,
    group_bootstrap,
    macro_f1,
)
from tiles.model import count_parameters, load_checkpoint  # noqa: E402
from tiles.train import predict  # noqa: E402

TOLERANCE = 8  # pixels at 224 x 224, about 3.5% of the image side
TARGET_DEFECT_RECALL = 0.98  # on validation: a missed defect ships a bad magnet


def plot_confusion(cm: list[list[int]], path: Path, title: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(data.CLASSES)), yticks=range(len(data.CLASSES)), xticklabels=data.CLASSES, yticklabels=data.CLASSES, xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def gradcam_check(model, frame, root: Path, figure: Path | None) -> dict:
    defects = frame[frame["y"] != 0].reset_index(drop=True)
    transform = data.eval_transform()
    centre = np.zeros((data.IMG_SIZE, data.IMG_SIZE), dtype=bool)
    centre[data.IMG_SIZE // 2, data.IMG_SIZE // 2] = True

    rows, examples = [], {}
    for _, row in defects.iterrows():
        with Image.open(root / row["path"]) as im:
            image = transform(im.convert("L"))
        mask = data.load_mask(row["path"], root)
        if not mask.any():
            continue
        cam, _ = grad_cam(model, image, target=int(row["y"]))
        dilated = dilate(mask, TOLERANCE)
        rows.append(
            {
                "label": row["label"],
                "hit": pointing_hit(cam, mask, TOLERANCE),
                "centre_hit": bool(dilated[centre].any()),
                "random_hit": float(dilated.mean()),
                "energy": energy_inside(cam, mask),
                "mask_area": float(mask.mean()),
            }
        )
        if row["label"] not in examples:
            examples[row["label"]] = (row["path"], cam, mask)

    def summarise(items):
        return {
            "images": len(items),
            "pointing_game": float(np.mean([r["hit"] for r in items])),
            "centre_baseline": float(np.mean([r["centre_hit"] for r in items])),
            "random_baseline": float(np.mean([r["random_hit"] for r in items])),
            "energy_on_defect": float(np.mean([r["energy"] for r in items])),
            "mask_area": float(np.mean([r["mask_area"] for r in items])),
        }

    result = summarise(rows)
    result["per_class"] = {c: summarise([r for r in rows if r["label"] == c]) for c in data.DEFECTS if any(r["label"] == c for r in rows)}

    if figure:
        fig, axes = plt.subplots(2, len(examples), figsize=(2.6 * len(examples), 5.4))
        for col, label in enumerate(c for c in data.DEFECTS if c in examples):
            path, cam, mask = examples[label]
            with Image.open(root / path) as im:
                gray = np.asarray(im.convert("L").resize((data.IMG_SIZE, data.IMG_SIZE)))
            axes[0, col].imshow(gray, cmap="gray")
            axes[0, col].contour(mask, levels=[0.5], colors="lime", linewidths=1)
            axes[0, col].set_title(label)
            axes[1, col].imshow(gray, cmap="gray")
            axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
            axes[1, col].contour(mask, levels=[0.5], colors="white", linewidths=0.8)
            for ax in axes[:, col]:
                ax.axis("off")
        axes[0, 0].text(-0.08, 0.5, "Image + labelled defect", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
        axes[1, 0].text(-0.08, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
        fig.tight_layout()
        fig.savefig(figure, dpi=120)
        plt.close(fig)
    return result


def leakage_check(frame, root: Path, out_dir: Path) -> dict | None:
    """Score the random-split model separately on test images whose tile it trained on and on those it did not."""
    ckpt = out_dir / "models" / "mobilenet_v3_small_random_split.pt"
    if not ckpt.exists():
        return None
    split = data.random_splits(frame)  # same seed as tiles.train --split random
    train_groups = set(frame.loc[split == "train", "group"])
    test = frame[split == "test"]
    twin = test["group"].isin(train_groups).to_numpy()
    model, _ = load_checkpoint(ckpt)
    probs, y = predict(model, DataLoader(data.TileDataset(test, root), batch_size=64))

    def score(mask):
        return {
            "images": int(mask.sum()),
            "defects": int((y[mask] != 0).sum()),
            "accuracy": float((probs[mask].argmax(1) == y[mask]).mean()),
            "defect_recall": defect_recall(y[mask], probs[mask]),
        }

    overall = classification_summary(y, probs, data.CLASSES)
    return {
        "test_images_with_a_training_twin": float(twin.mean()),
        "random_split_test": {"macro_f1": overall["macro_f1"], "accuracy": overall["accuracy"], "defect_recall": overall["defect_detection"]["recall"]},
        "with_twin": score(twin),
        "without_twin": score(~twin),
    }


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=data.DATA_DIR)
    p.add_argument("--splits", type=Path, default=data.SPLITS_CSV)
    p.add_argument("--out-dir", type=Path, default=data.ROOT)
    p.add_argument("--main", default="mobilenet_v3_small")
    args = p.parse_args(argv)
    torch.set_num_threads(4)

    frame = data.load_splits(args.splits)
    test, val = frame[frame["split"] == "test"], frame[frame["split"] == "val"]
    loader = DataLoader(data.TileDataset(test, args.data_dir), batch_size=64)
    val_loader = DataLoader(data.TileDataset(val, args.data_dir), batch_size=64)
    figures = args.out_dir / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    results = {
        "test_images": int(len(test)),
        "test_tiles": int(test["group"].nunique()),
        "test_tiles_per_class": test.groupby("label")["group"].nunique().to_dict(),
        "models": {},
    }
    groups = test["group"].to_numpy()

    def score(name, val_probs, val_y, probs, y, extra):
        summary = classification_summary(y, probs, data.CLASSES)
        threshold = choose_defect_threshold(val_y, val_probs, TARGET_DEFECT_RECALL)
        val_summary = classification_summary(val_y, val_probs, data.CLASSES)
        del val_summary["confusion_matrix"]
        summary.update(
            {
                **extra,
                "val": val_summary,
                "ci95": {
                    "macro_f1": group_bootstrap(y, probs, groups, macro_f1),
                    "defect_recall": group_bootstrap(y, probs, groups, defect_recall),
                    "alarm_recall": group_bootstrap(y, probs, groups, lambda t, p: defect_alarm(t, p, threshold)["recall"]),
                },
                "alarm": {"target_val_recall": TARGET_DEFECT_RECALL, "val": defect_alarm(val_y, val_probs, threshold), "test": defect_alarm(y, probs, threshold)},
            }
        )
        results["models"][name] = summary
        plot_confusion(summary["confusion_matrix"], figures / f"confusion_{name}.png", f"{name}: test set")

    base = baseline.fit(frame, args.data_dir)
    score("logreg_baseline", *base["val"], *base["test"], {"C": base["C"], "features": base["features"]})

    ckpt_path = args.out_dir / "models" / f"{args.main}.pt"
    if ckpt_path.exists():
        model, ckpt = load_checkpoint(ckpt_path)
        val_probs, val_y = predict(model, val_loader)
        probs, y = predict(model, loader)
        score(args.main, val_probs, val_y, probs, y, {"parameters": count_parameters(model), "best_epoch": ckpt["epoch"]})
        results["gradcam"] = gradcam_check(model, test, args.data_dir, figures / "gradcam_examples.png")

    results["leakage_check"] = leakage_check(frame, args.data_dir, args.out_dir)
    (args.out_dir / "reports" / "metrics.json").write_text(json.dumps(results, indent=2))
    for arch, s in results["models"].items():
        d, a, ci = s["defect_detection"], s["alarm"]["test"], s["ci95"]
        print(
            f"{arch:20s} macro-F1 {s['macro_f1']:.3f} [{ci['macro_f1'][0]:.2f}, {ci['macro_f1'][1]:.2f}]  acc {s['accuracy']:.3f}  "
            f"argmax: caught {d['caught']}/{d['defects']}, false alarms {d['false_alarms']}/{d['good_tiles']}  "
            f"alarm@{a['threshold']:.3f}: caught {a['caught']}/{a['defects']}, false alarms {a['false_alarms']}/{a['good_tiles']}"
        )
    if "gradcam" in results:
        g = results["gradcam"]
        print(f"Grad-CAM pointing game {g['pointing_game']:.3f} (centre {g['centre_baseline']:.3f}, random {g['random_baseline']:.3f}); energy on defect {g['energy_on_defect']:.3f} vs mask area {g['mask_area']:.3f}")
    if results["leakage_check"]:
        print("Leakage check:", json.dumps(results["leakage_check"]))
    return results


if __name__ == "__main__":
    main()
