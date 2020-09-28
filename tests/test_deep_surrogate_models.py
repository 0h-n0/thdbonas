import pytest

from thdbonas.deep_surrogate_models import SimpleNetwork
from thdbonas.trial import Trial


def test_simple_network_learn():
    s = SimpleNetwork(4, 32)
    t = Trial()
    setattr(t, 'hidden1', 16)
    setattr(t, 'hidden2', 32)
    setattr(t, 'lr', 0.01)
    setattr(t, 'batchsize', 64)
    bases = s.learn([t, t], [0.2, 0.1], 3)
    assert int(bases.shape[0]) == 2
    assert int(bases.shape[1]) == 64


def test_simple_network_predict():
    s = SimpleNetwork(4, 32)
    t = Trial()
    setattr(t, 'hidden1', 16)
    setattr(t, 'hidden2', 32)
    setattr(t, 'lr', 0.01)
    setattr(t, 'batchsize', 64)
    bases = s.predict([t, t])
    assert int(bases.shape[0]) == 2
    assert int(bases.shape[1]) == 64
