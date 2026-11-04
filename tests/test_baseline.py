import numpy as np

from tiles import baseline, data


def test_features_have_fixed_length(tile_root):
    frame = data.index_images(tile_root)
    a = baseline.image_features(frame["path"].iloc[0], tile_root)
    b = baseline.image_features(frame["path"].iloc[-1], tile_root)
    assert a.shape == b.shape and np.isfinite(a).all()


def test_fit_returns_probabilities_for_val_and_test(tile_root, tmp_path):
    frame = data.build_splits(tile_root, tmp_path / "splits.csv")
    result = baseline.fit(frame, tile_root)
    for split in ("val", "test"):
        probs, y = result[split]
        assert probs.shape == (len(y), len(data.CLASSES))
        assert np.allclose(probs.sum(1), 1.0)
    assert result["C"] in baseline.C_GRID
