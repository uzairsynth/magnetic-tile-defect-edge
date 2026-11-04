"""Download the Magnetic Tile Defect dataset into data/raw/MT (about 50 MB).

Source: https://github.com/abin24/Magnetic-tile-defect-datasets.
Only MT_<class>/Imgs/*.jpg images and *.png defect masks are extracted, by
base name, so nothing in the archive can write outside the target folder.
"""

import argparse
import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path

URL = "https://codeload.github.com/abin24/Magnetic-tile-defect-datasets./zip/refs/heads/master"
ROOT = Path(__file__).resolve().parents[1]
MEMBER = re.compile(r"[^/]+/MT_(Blowhole|Break|Crack|Fray|Free|Uneven)/Imgs/([\w.-]+\.(?:jpg|png))")
EXPECTED_IMAGES = 1344


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "MT")
    args = parser.parse_args()

    if len(list(args.out.glob("*/*.jpg"))) == EXPECTED_IMAGES:
        print(f"Already present: {args.out}")
        return

    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=300) as response:
        payload = response.read()
    print(f"{len(payload) / 1e6:.1f} MB, sha256 {hashlib.sha256(payload).hexdigest()}")

    images = masks = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            match = MEMBER.fullmatch(name)
            if not match:
                continue
            label, base = match.groups()
            target = args.out / label
            target.mkdir(parents=True, exist_ok=True)
            (target / base).write_bytes(archive.read(name))
            if base.endswith(".jpg"):
                images += 1
            else:
                masks += 1

    if images != EXPECTED_IMAGES:
        raise SystemExit(f"Expected {EXPECTED_IMAGES} images, extracted {images}")
    print(f"Extracted {images} images and {masks} masks to {args.out}")


if __name__ == "__main__":
    main()
