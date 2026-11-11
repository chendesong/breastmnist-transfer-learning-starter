"""Download and check the official BreastMNIST 28x28 archive."""

import argparse
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np

# Source: https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py
DATA_URL = "https://zenodo.org/records/10519652/files/breastmnist.npz?download=1"
DATA_MD5 = "750601b1f35ba3300ea97c75c52ff8f6"
SPLIT_SIZES = {"train": 546, "val": 78, "test": 156}
LABEL_NAMES = {0: "malignant", 1: "normal/benign"}


def fetch_archive(data_dir):
    path = data_dir / "breastmnist.npz"
    if path.exists():
        content = path.read_bytes()
    else:
        print("Downloading the official BreastMNIST 28x28 archive...", flush=True)
        with urlopen(DATA_URL, timeout=30) as response:
            content = response.read()
    if hashlib.md5(content).hexdigest() != DATA_MD5:
        raise ValueError(f"Checksum mismatch: {path}. No existing file was replaced.")
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.part")
        temporary.write_bytes(content)
        temporary.replace(path)
    print(f"PASS: official MD5 checksum ({DATA_MD5})")
    return path


def check_arrays(path):
    expected = {f"{split}_{kind}" for split in SPLIT_SIZES
                for kind in ("images", "labels")}
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != expected:
            raise ValueError("Expected exactly six official image/label arrays.")
        print("Labels: 0 = malignant; 1 = normal/benign")
        print("Split   Images          Labels    Malignant  Normal/benign  Pixel range")
        for split, size in SPLIT_SIZES.items():
            images, labels = data[f"{split}_images"], data[f"{split}_labels"]
            if images.shape != (size, 28, 28) or images.dtype != np.uint8:
                raise ValueError(f"{split}: expected uint8 images of shape ({size}, 28, 28).")
            if labels.shape != (size, 1) or labels.dtype != np.uint8:
                raise ValueError(f"{split}: expected uint8 labels of shape ({size}, 1).")
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError(f"{split}: expected both official labels 0 and 1.")
            counts = np.bincount(labels.ravel(), minlength=2)
            print(f"{split:<7} {str(images.shape):<15} {str(labels.shape):<9} "
                  f"{counts[0]:<10} {counts[1]:<14} {images.min()}-{images.max()}")
        return data["train_images"][:8], data["train_labels"][:8]


def save_preview(images, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(10, 6), layout="constrained")
    for index, (axis, image, label) in enumerate(zip(axes.flat, images, labels.ravel())):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(f"train[{index}]\n{label}: {LABEL_NAMES[int(label)]}", fontsize=9)
        axis.axis("off")
    figure.suptitle("Original 28x28 training images (display enlarged)")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Training preview: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data",
                        help="Dataset cache and preview directory (default: data beside this script).")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    archive = fetch_archive(data_dir)
    images, labels = check_arrays(archive)
    save_preview(images, labels, data_dir / "train_preview.png")
    print(f"PASS: all dataset checks completed. Archive: {archive}")
    print("Official splits and labels are unchanged. No model was trained or evaluated.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
