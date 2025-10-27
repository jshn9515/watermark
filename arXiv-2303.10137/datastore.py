from typing import Literal

import torch
import torchvision.transforms.v2 as v2
from torchvision.datasets import VisionDataset
from torchvision.transforms.v2 import InterpolationMode as Mode

__all__ = ['load_dataset']


def load_dataset(
    name: str,
    root: str,
    resolution: int | tuple[int, int] = 32,
    mode: Literal['resize', 'crop'] = 'resize',
) -> VisionDataset:
    match name.upper():
        case 'CIFAR10':
            return load_cifar10(root, resolution, mode)
        case 'MNIST':
            return load_mnist(root, resolution, mode)
        case 'LSUN':
            return load_lsun(root, resolution, mode)
        case _:
            raise ValueError(f'Unknown dataset: {name}.')


def load_cifar10(
    root: str,
    resolution: int | tuple[int, int] = 32,
    mode: Literal['resize', 'crop'] = 'resize',
) -> VisionDataset:
    from torchvision.datasets import CIFAR10

    transform = v2.Compose(
        [
            v2.Resize(resolution, interpolation=Mode.LANCZOS)
            if mode == 'resize'
            else v2.CenterCrop(resolution),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    ds = CIFAR10(root, train=True, transform=transform, download=True)
    return ds


def load_lsun(
    root: str,
    resolution: int | tuple[int, int] = 64,
    mode: Literal['resize', 'crop'] = 'resize',
) -> VisionDataset:
    from torchvision.datasets import LSUN

    transform = v2.Compose(
        [
            v2.Resize(resolution, interpolation=Mode.LANCZOS)
            if mode == 'resize'
            else v2.CenterCrop(resolution),
            v2.CenterCrop(resolution),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    ds = LSUN(root, classes='train', transform=transform)
    return ds


def load_mnist(
    root: str,
    resolution: int | tuple[int, int] = 32,
    mode: Literal['resize', 'crop'] = 'resize',
) -> VisionDataset:
    from torchvision.datasets import MNIST

    transform = v2.Compose(
        [
            v2.Resize(resolution, interpolation=Mode.LANCZOS)
            if mode == 'resize'
            else v2.CenterCrop(resolution),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    ds = MNIST(root, train=True, transform=transform, download=True)
    return ds
