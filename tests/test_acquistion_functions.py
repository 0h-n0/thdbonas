from thdbonas.acquistion_functions import (_expected_improvement,
                                           AcquisitonFunction,
                                           AcquisitonFunctionType)
import torch
import numpy as np
import pytest


def test__expected_improvement():
    mean = torch.arange(0.1, 1, 0.1)
    sigma = torch.arange(0.1, 1, 0.1)
    min_val = torch.tensor(0.1).float()
    eis = _expected_improvement(mean, sigma, min_val)
    assert eis.dim() == 1
    assert int(eis.size()[0]) == 9

@pytest.mark.xfail
def test__expected_improvement_with_invalid_shape():
    mean = np.arange(0, 1, 0.1).reshape(2, 5)
    sigma = np.arange(0, 1, 0.1)
    min_val = np.float64(0.1)
    eis = _expected_improvement(mean, sigma, min_val)

@pytest.mark.xfail
def test__expected_improvement_invalid_type():
    mean = [0.1*i for i in range(10)]
    sigma = np.arange(0, 1, 0.1)
    min_val = np.float64(0.1)
    eis = _expected_improvement(mean, sigma, min_val)

@pytest.mark.xfail
def test__AcquisitonFunctionType():
    assert AcquisitonFunctionType.EI == 1

def test__AcquisitonFunction():
    f = AcquisitonFunction(AcquisitonFunctionType.EI)
    mean = torch.arange(0.1, 1, 0.1)
    sigma = torch.arange(0.1, 1, 0.1)
    min_val = torch.tensor(0.1).float()
    eis = f(mean, sigma, min_val)
    assert eis.dim() == 1
    assert int(eis.size()[0]) == 9
