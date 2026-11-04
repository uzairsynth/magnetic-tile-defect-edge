"""A tiny synthetic copy of the dataset layout: no download or pretrained weights needed."""

import numpy as np
import pytest
from PIL import Image

from tiles import data


def save_pair(folder, stem, pixels, mask):
    Image.fromarray(pixels.astype(np.uint8), "L").save(folder / f"{stem}.jpg", quality=95)
    Image.fromarray((mask * 255).astype(np.uint8), "L").save(folder / f"{stem}.png")


@pytest.fixture(scope="session")
def tile_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("MT")
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:120, 0:160]
    for c, label in enumerate(data.CLASSES):
        folder = root / label
        folder.mkdir()
        for k in range(8):
            # Distinct random texture per tile, with a bright blob as the "defect".
            pixels = rng.normal(120, 25, (120, 160)).clip(0, 255)
            mask = np.zeros((120, 160), dtype=bool)
            if label != "Free":
                cy, cx = rng.integers(30, 90), rng.integers(30, 130)
                mask = (yy - cy) ** 2 + (xx - cx) ** 2 < 12**2
                pixels[mask] = 230
            save_pair(folder, f"exp1_num_{c * 100 + k}", pixels, mask)
            if k == 0:
                # The same tile under a second "lighting setup": a near-duplicate.
                save_pair(folder, f"exp2_num_{c * 100 + k + 50}", (pixels * 0.9 + 10).clip(0, 255), mask)
    return root
