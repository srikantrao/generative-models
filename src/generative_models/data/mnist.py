from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor

MNIST_IMAGE_SHAPE = (1, 28, 28)
MNIST_NUM_CLASSES = 10
MNIST_TRAIN_SIZE = 60_000
MNIST_TEST_SIZE = 10_000

MNIST_RAW_FILENAMES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)

MNIST_NORMALIZATION = "uint8 [0,255] -> float32 [0,1] -> 2*x-1 -> [-1,1]"

@dataclass(frozen=True)
class MNISTEvaluatorSplit:
    train_indices: Tensor
    validation_indices: Tensor

def normalize_mnist_image(image: Tensor) -> Tensor:
    """
    performs the following normalization to all the values:  [0,1] -> 2*x-1 -> [-1,1]
    """
    if not image.dtype.is_floating_point:
        raise ValueError("image must have floating-point dtype")
    return image.mul(2.0).sub(1.0)

def denormalize_mnist_image(image: Tensor) -> Tensor:
    """
    performs the following computation to the values: [-1.1] -> (x+1)/2 --> [0,1]
    """
    if not image.dtype.is_floating_point:
        raise ValueError("image must have floating-point dtype")
    return image.add(1.0).mul(0.5)

def mnist_transform() -> Compose:
    """
    transformations to be applied to mnist images. Current set -
    1. Covert to tensor
    2. Normalize from [0, 1] --> [-1, 1]
    """
    return Compose(
        [
            ToTensor(),
            Lambda(normalize_mnist_image),
        ]
    )

def load_mnist_split(
    root: str | Path,
    *,
    train: bool,
    download: bool,
) -> MNIST:
    return MNIST(
        root=root,
        train=train,
        transform=mnist_transform(),
        download=download
    )

def make_mnist_evaluator_split(
    *,
    num_examples: int = MNIST_TRAIN_SIZE,
    validation_size: int = 5000,
    seed: int = 2026,
) -> MNISTEvaluatorSplit:

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_examples, generator=generator)
    validation_indices = torch.sort(permutation[:validation_size]).values
    train_indices = torch.sort(permutation[validation_size:]).values

    return MNISTEvaluatorSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
    )

def index_set_sha256(indices: Tensor) -> str:
    canonical = torch.sort(indices.detach().to(device="cpu", dtype=torch.int64)).values
    payload = ",".join(str(index) for index in canonical.tolist()).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def mnist_raw_file_sha256(root: str | Path) -> dict[str, str]:
    """
    generate a SHA256 hash of the raw contents of the MNIST file.
    """
    raw_folder = Path(root) / MNIST.__name__ / "raw"
    digests: dict[str, str] = {}
    for filename in MNIST_RAW_FILENAMES:
        path = raw_folder / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing MNIST raw file: {path}")
        with path.open("rb") as raw_file:
            digests[filename] = hashlib.file_digest(
                raw_file,
                "sha256",
            ).hexdigest()

    return digests
