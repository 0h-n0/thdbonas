import math
from enum import Flag, auto

import torch
import torch.distributions as tdist

from .utils import is_float


def normal_pdf(x, mean, std):
    return 1 / math.sqrt(2 * math.pi * std**2) * \
        torch.exp(-(x - mean) ** 2 / (2 * std**2))

def _expected_improvement(mean: torch.Tensor,
                          sigma: torch.Tensor,
                          min_val: torch.Tensor):
    assert isinstance(mean, torch.Tensor), f'instance type error, {type(mean)}'
    assert isinstance(sigma, torch.Tensor), f'instance type error, {type(mean)}'
    assert len(mean.shape) == 1, f'Invalid shape error, {mean.shape}'
    assert len(sigma.shape) == 1, f'Invalid shape error, {sigma.shape}'
    assert mean.size() == sigma.size(), f'Invalid shape error, {sigma.size()} != {sigma.size()}'
    assert not min_val.size(), f'Invalid shape error, {min_val.size()}'

    gamma = (min_val - mean) / sigma
    pdf = normal_pdf(gamma, mean=0., std=1.)
    cdf = tdist.Normal(loc=0., scale=1.).cdf(gamma)
    ei = (min_val - mean) * cdf + (sigma * pdf)
    return ei


class AcquisitonFunctionType(Flag):
    EI = auto()


class AcquisitonFunction:
    def __init__(self, aftype: AcquisitonFunctionType = AcquisitonFunctionType.EI):
        if AcquisitonFunctionType.EI == aftype:
            self.af_func = _expected_improvement
        else:
            raise NotImplementedError("EI is only supported")

    def __call__(self, *args, **kwargs):
        return self.af_func(*args, **kwargs)
