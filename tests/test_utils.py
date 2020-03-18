from pathlib import Path

import numpy as np
import pytest

from tfdbonas.utils import load_class, is_float


def test_load_class():
    path = 'tfdbonas.deep_surrogate_models:SimpleNetwork'
    c = load_class(path)
    c()

@pytest.fixture(
    params=[[0.1, True],
            [np.float16(0.1), True],
            [np.float32(0.1), True],
            [np.float64(0.1), True],
            [1, False],
            [np.int8(1), False],
            [np.int16(1), False],
            [np.int32(1), False],
            [np.int64(1), False],
    ]
)
def float_types(request):
    input = request.param[0]
    expected = request.param[1]
    return input, expected

def test_is_float(float_types):
    input, expected = float_types
    assert is_float(input) == expected
