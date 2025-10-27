from typing import Any

import click
import torch

__all__ = ['generate_random_fingerprints', 'IntOrTuple', 'AttributeDict']


def generate_random_fingerprints(bit_length: int, batch_size: int = 4) -> torch.Tensor:
    dist = torch.distributions.Bernoulli(probs=0.5)
    z = dist.sample((batch_size, bit_length))
    return z


class IntOrTuple(click.ParamType):
    """Custom Click parameter type to parse an integer or a tuple of two integers."""

    name = 'int|tuple[int,int]'

    def convert(self, value, param, ctx):
        parts = str(value).replace(',', ' ').split()
        try:
            nums = list(map(int, parts))
        except ValueError:
            self.fail('must be int or two ints', param, ctx)
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return tuple(nums)
        else:
            self.fail('must be int or two ints', param, ctx)


class AttributeDict(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttributeDict' has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
    
    def __repr__(self) -> str:
        return f'AttributeDict({super().__repr__()})'
