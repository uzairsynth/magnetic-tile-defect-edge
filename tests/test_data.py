import numpy as np
import pandas as pd

from tiles import data


def test_index_finds_every_class(tile_root):
    frame = data.index_images(tile_root)
    assert set(frame["label"]) == set(data.CLASSES)
    assert (frame.loc[frame["label"] == "Free", "y"] == 0).all()


def test_near_duplicates_share_a_group(tile_root):
    frame = data.index_images(tile_root)
    groups = data.similarity_groups(frame, tile_root)
    by_stem = dict(zip(frame["path"], groups))
    assert by_stem["Crack/exp1_num_300.jpg"] == by_stem["Crack/exp2_num_350.jpg"]
    assert by_stem["Crack/exp1_num_300.jpg"] != by_stem["Crack/exp1_num_301.jpg"]


def test_splits_never_separate_a_group():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"y": np.repeat(np.arange(6), 70)})
    groups = rng.integers(0, 120, len(frame))
    split = data.assign_splits(frame, groups, seed=1)
    assert set(split) == {"train", "val", "test"}
    assert (pd.Series(split.values).groupby(groups).nunique() == 1).all()


def test_build_splits_writes_csv(tile_root, tmp_path):
    frame = data.build_splits(tile_root, tmp_path / "splits.csv")
    assert (tmp_path / "splits.csv").exists()
    assert frame.groupby("group")["split"].nunique().max() == 1


def test_class_weights_lift_rare_classes():
    w = data.class_weights([0] * 90 + [1] * 10, n_classes=2)
    assert w[1] > w[0]
    assert np.isclose(float(w.mean()), 1.0)


def test_mask_is_binary_and_resized(tile_root):
    mask = data.load_mask("Crack/exp1_num_300.jpg", tile_root)
    assert mask.shape == (data.IMG_SIZE, data.IMG_SIZE)
    assert mask.dtype == bool and 0 < mask.mean() < 0.5


def test_dataset_returns_normalised_three_channel_tensor(tile_root):
    frame = data.index_images(tile_root)
    image, label = data.TileDataset(frame, tile_root)[0]
    assert tuple(image.shape) == (3, data.IMG_SIZE, data.IMG_SIZE)
    assert label == 0
