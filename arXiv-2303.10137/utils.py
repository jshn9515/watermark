import argparse

import torch

__all__ = ['generate_random_fingerprints', 'parse_resolution']


def generate_random_fingerprints(bit_length: int, batch_size: int = 4) -> torch.Tensor:
    dist = torch.distributions.Bernoulli(probs=0.5)
    z = dist.sample((batch_size, bit_length))
    return z


def parse_resolution(value: str) -> int | tuple[int, ...]:
    try:
        return int(value)
    except ValueError:
        parts = value.split(',')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                'Resolution must be int or a tuple like (height, width).'
            )
        try:
            return tuple(int(x) for x in parts)
        except ValueError:
            raise argparse.ArgumentTypeError('Each dimension must be an int.')
