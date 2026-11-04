"""Index the Magnetic Tile Defect images, group near-duplicates and split without leakage.

The dataset photographs each tile under several lighting setups (file prefixes
exp1..exp6), so one physical tile appears up to six times. A random split would
put views of the same tile in both train and test. Images are therefore grouped
by visual similarity, and whole groups are assigned to a split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "MT"
SPLITS_CSV = ROOT / "data" / "splits.csv"

# "Free" (no defect) first, so index 0 means a good tile.
CLASSES = ["Free", "Blowhole", "Break", "Crack", "Fray", "Uneven"]
DEFECTS = CLASSES[1:]
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def index_images(root: Path = DATA_DIR) -> pd.DataFrame:
    rows = []
    for label in CLASSES:
        for path in sorted((Path(root) / label).glob("*.jpg")):
            with Image.open(path) as im:
                width, height = im.size
            rows.append(
                {
                    "path": f"{label}/{path.name}",
                    "label": label,
                    "y": CLASSES.index(label),
                    "exp": path.stem.split("_")[0],
                    "width": width,
                    "height": height,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No images under {root}; run scripts/download_data.py")
    return pd.DataFrame(rows)


def _thumbnail(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size)), dtype=np.float32).ravel()
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a / np.sqrt(a.size)


def similarity_groups(frame: pd.DataFrame, root: Path = DATA_DIR, threshold: float = 0.95, max_size_diff: int = 10, thumb: int = 64) -> np.ndarray:
    """Union images whose 64x64 thumbnails correlate above `threshold` and whose sizes match.

    Over-merging only makes the split stricter, so the threshold errs on the side of grouping.
    """
    thumbs = np.stack([_thumbnail(Path(root) / p, thumb) for p in frame["path"]])
    corr = thumbs @ thumbs.T
    sizes = frame[["width", "height"]].to_numpy()
    size_diff = np.abs(sizes[:, None, :] - sizes[None, :, :]).max(-1)
    linked = np.argwhere(np.triu((corr > threshold) & (size_diff <= max_size_diff), 1))

    parent = np.arange(len(frame))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in linked:
        parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(frame))])
    return np.unique(roots, return_inverse=True)[1]


def assign_splits(frame: pd.DataFrame, groups: np.ndarray, seed: int = 42) -> pd.Series:
    """About 72 / 14 / 14 train / val / test, stratified by class, with no group in two splits."""
    split = np.full(len(frame), "train", dtype=object)
    y = frame["y"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest, test = next(outer.split(frame, y, groups))
    split[test] = "test"
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    _, val = next(inner.split(frame.iloc[rest], y[rest], groups[rest]))
    split[rest[val]] = "val"
    return pd.Series(split, index=frame.index, name="split")


def random_splits(frame: pd.DataFrame, seed: int = 42) -> pd.Series:
    """The leaky alternative, kept only to measure how much it inflates scores."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(frame))
    rest, test = train_test_split(idx, test_size=1 / 7, stratify=frame["y"], random_state=seed)
    train, val = train_test_split(rest, test_size=1 / 6, stratify=frame["y"].iloc[rest], random_state=seed)
    split = np.full(len(frame), "train", dtype=object)
    split[val], split[test] = "val", "test"
    return pd.Series(split, index=frame.index, name="split")


def build_splits(root: Path = DATA_DIR, out: Path = SPLITS_CSV, seed: int = 42) -> pd.DataFrame:
    frame = index_images(root)
    frame["group"] = similarity_groups(frame, root)
    frame["split"] = assign_splits(frame, frame["group"].to_numpy(), seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def load_splits(path: Path = SPLITS_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        return build_splits(out=Path(path))
    return pd.read_csv(path)


def eval_transform(img_size: int = IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def train_transform(img_size: int = IMG_SIZE):
    # Tiles have no fixed orientation, and lighting differs between setups.
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


class TileDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path = DATA_DIR, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with Image.open(self.root / row["path"]) as im:
            image = self.transform(im.convert("L"))
        return image, int(row["y"])


def load_mask(path: str, root: Path = DATA_DIR, img_size: int = IMG_SIZE) -> np.ndarray:
    """Binary defect mask resized like the image (the masks are stored next to each .jpg as .png)."""
    with Image.open(Path(root) / Path(path).with_suffix(".png")) as m:
        return np.asarray(m.convert("L").resize((img_size, img_size), Image.NEAREST)) > 127


def class_weights(labels, n_classes: int = len(CLASSES)) -> torch.Tensor:
    """Square-root inverse frequency: lifts rare defects without letting 32 Fray images dominate."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    weights = np.sqrt(counts.sum() / (n_classes * np.maximum(counts, 1)))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)
