"""BreastMNIST helpers. Original labels: 0 malignant, 1 normal/benign.

Default inputs: whole-image resize, three grayscale channels, ImageNet
normalisation. Supply custom transforms when your chosen model needs them.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
from pathlib import Path
import platform
import random
import sys
import urllib.request

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


DATA_URL = "https://zenodo.org/records/10519652/files/breastmnist.npz?download=1"
DATA_MD5 = "750601b1f35ba3300ea97c75c52ff8f6"
LABEL_NAMES = {0: "malignant", 1: "normal, benign"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_SIZES = {"train": 546, "val": 78, "test": 156}


def set_seed(seed: int, threads: int = 4) -> None:
    """Seed Python/NumPy/PyTorch; exact results can still differ by platform."""
    if threads < 1:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_official_archive(data_dir, download: bool = True) -> dict[str, np.ndarray]:
    """Load/check the original archive without re-splitting or changing labels.

    A first download is cached locally only after its official MD5 matches.
    An existing archive with a different checksum raises an error, not an
    automatic replacement. MD5 is used here to match the dataset publisher.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    path = data_dir / "breastmnist.npz"
    if path.exists():
        content = path.read_bytes()
    elif download:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != DATA_MD5:
            raise ValueError("Downloaded BreastMNIST checksum differs from official metadata")
        data_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.download")
        temporary.write_bytes(content)
        temporary.replace(path)
    else:
        raise FileNotFoundError(f"Dataset not cached at {path}; enable download first")
    if hashlib.md5(content).hexdigest() != DATA_MD5:
        raise ValueError(f"Checksum mismatch for {path}; review this file before replacing it")

    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        arrays = {f"{s}_{k}": archive[f"{s}_{k}"].copy()
                  for s in SPLIT_SIZES for k in ("images", "labels")}
    for split, size in SPLIT_SIZES.items():
        images, labels = arrays[f"{split}_images"], arrays[f"{split}_labels"]
        if images.shape != (size, 28, 28) or labels.shape != (size, 1):
            raise ValueError(f"Unexpected official {split} array shape")
        if images.dtype != np.uint8 or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Unexpected {split} image dtype or label encoding")
    return arrays


def _limited_indices(labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    """A deterministic stratified smoke-test subset within one official split."""
    labels = labels.reshape(-1)
    if limit is None:
        return np.arange(len(labels))
    if not isinstance(limit, int) or not 1 <= limit <= len(labels):
        raise ValueError(f"limit must be between 1 and {len(labels)}")
    if limit == len(labels):
        return np.arange(len(labels))
    counts = np.bincount(labels.astype(int), minlength=2)
    quota = np.floor(limit * counts / len(labels)).astype(int)
    for label in np.argsort(-(limit * counts / len(labels) - quota)):
        if quota.sum() == limit:
            break
        quota[label] += 1
    if limit >= 2 and counts.min() > 0 and quota.min() == 0:
        missing = int(np.argmin(quota))
        quota[missing], quota[1 - missing] = 1, limit - 1
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == label), size=int(quota[label]), replace=False)
        for label in (0, 1)
    ]))


class BreastMNISTDataset(Dataset):
    """Yield (transformed image, original label); indices refer to NPZ rows."""

    def __init__(self, images, labels, transform, indices):
        self.images = images
        self.labels = labels.reshape(-1)
        self.transform = transform
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(self.indices[index])
        image = Image.fromarray(self.images[original_index])
        return self.transform(image), int(self.labels[original_index])


def make_loaders(data_dir, batch_size: int = 16, image_size: int = 224,
                 augment: bool = False, seed: int = 42, train_limit=None,
                 val_limit=None, include_test: bool = False,
                 train_transform=None, eval_transform=None) -> dict[str, DataLoader]:
    """Keep official splits; expose test only by opt-in. Limits are for smoke checks.

    Custom PIL-to-tensor transforms replace the defaults for the relevant split.
    Keep evaluation deterministic and use identical preprocessing across the
    two initialisations. Augmentation applies only to default training inputs.
    """
    if batch_size < 1 or image_size < 1:
        raise ValueError("batch_size and image_size must be positive")
    arrays = load_official_archive(data_dir)
    resize = transforms.Resize((image_size, image_size),
                               interpolation=InterpolationMode.BILINEAR, antialias=True)
    normalise = [transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    augmentation = [transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
                    transforms.RandomHorizontalFlip()] if augment else []
    if train_transform is None:
        train_transform = transforms.Compose([resize, *augmentation, *normalise])
    if eval_transform is None:
        eval_transform = transforms.Compose([resize, *normalise])
    loaders = {}
    for split in ("train", "val", "test") if include_test else ("train", "val"):
        limit = train_limit if split == "train" else val_limit if split == "val" else None
        indices = _limited_indices(arrays[f"{split}_labels"], limit,
                                   seed + (0 if split == "train" else 1))
        dataset = BreastMNISTDataset(arrays[f"{split}_images"], arrays[f"{split}_labels"],
                                     train_transform if split == "train" else eval_transform, indices)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                                    num_workers=0, drop_last=False,
                                    generator=torch.Generator().manual_seed(seed))
    return loaders


def _roc_auc(positive: np.ndarray, scores: np.ndarray) -> float | None:
    """Mann-Whitney ROC-AUC with average ranks for tied probabilities."""
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if not n_positive or not n_negative:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)]
    ranks = np.empty(len(scores), dtype=float)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


def metrics_from_scores(labels, malignant_scores, threshold: float = 0.5) -> dict:
    """Use malignant label 0 as positive; score is softmax(logits)[:, 0].

    Confusion rows=true and columns=predicted in original label order [0, 1],
    hence [[TP, FN], [FP, TN]]. Undefined class recalls/AUC/balanced accuracy
    return None; precision/F1 use zero_division=0. No one-class AUC warnings.
    """
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(malignant_scores, dtype=float).reshape(-1)
    if not len(labels) or len(labels) != len(scores):
        raise ValueError("labels and scores must have the same nonzero length")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Use original BreastMNIST labels: 0 malignant, 1 nonmalignant")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("malignant_scores must contain finite probabilities in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    positive, predicted_positive = labels == 0, scores >= threshold
    tp = int((positive & predicted_positive).sum())
    fn = int((positive & ~predicted_positive).sum())
    fp = int((~positive & predicted_positive).sum())
    tn = int((~positive & ~predicted_positive).sum())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "roc_auc": _roc_auc(positive, scores),
        "balanced_accuracy": ((sensitivity + specificity) / 2
                              if sensitivity is not None and specificity is not None else None),
        "malignant_sensitivity": sensitivity,
        "malignant_specificity": specificity,
        "malignant_precision": tp / (tp + fp) if tp + fp else 0.0,
        "malignant_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / len(labels),
        "threshold": float(threshold),
        "support": len(labels),
        "malignant_support": tp + fn,
        "nonmalignant_support": tn + fp,
        "confusion_matrix": [[tp, fn], [fp, tn]],
    }


def environment_info() -> dict:
    versions = {}
    for distribution in ("torch", "torchvision", "numpy", "Pillow", "matplotlib"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "platform": platform.platform(), "versions": versions,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda}
