from tfdbonas.acquistion_functions import (_expected_improvement,
                                           AcquisitonFunction,
                                           AcquisitonFunctionType)

import numpy as np
import pytest


def test__expected_improvement():
    mean = np.arange(0.1, 1, 0.1)
    sigma = np.arange(0.1, 1, 0.1)
    min_val = np.float64(0.1)
    eis = _expected_improvement(mean, sigma, min_val)
    assert eis.size == 9

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
    mean = np.arange(0.1, 1, 0.1)
    sigma = np.arange(0.1, 1, 0.1)
    min_val = np.float64(0.1)
    eis = f(mean, sigma, min_val)
    assert eis.size == 9
