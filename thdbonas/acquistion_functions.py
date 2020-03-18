from enum import Flag, auto

import scipy.stats
import torch

from .utils import is_float

def _expected_improvement(mean: torch.tensor, sigma: torch.tensor, min_val: torch.tensor):
    assert isinstance(mean, np.ndarray), f'instance type error, {type(mean)}'
    assert isinstance(sigma, np.ndarray), f'instance type error, {type(mean)}'
    assert is_float(min_val), f'instance type error, {type(mean)}'
    assert len(mean.shape) == 1, f'Invalid shape error, {mean.shape}'
    assert len(sigma.shape) == 1, f'Invalid shape error, {sigma.shape}'
    assert mean.size == sigma.size, f'Invalid shape error, {sigma.size} != {sigma.size}'
    assert min_val.size == 1, f'Invalid shape error, {min_val.size}'

    dist = scipy.stats.norm(loc=0.0, scale=1.0)
    gamma = (min_val - mean) / sigma
    pdf = dist.pdf(x=gamma)
    cdf = scipy.stats.norm.cdf(x=gamma, loc=0., scale=1.)
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
