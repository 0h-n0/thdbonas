import pytest

import tensorflow as tf

from tfdbonas.deep_surrogate_models import SimpleNetwork
from tfdbonas.trial import Trial

@pytest.mark.skipif(True, reason="remove kgcn")
def test_get_kgcn_gcn_class():
    import tensorflow as tf
    gcn_class = get_kgcn_gcn_class()
    g = gcn_class(32)
    features = tf.placeholder(tf.float32, shape=(1, 25, 10))
    adj = [tf.sparse_placeholder(tf.float32, shape=(25, 25)), ]
    o = g(features, [adj,])
    print(g.bases.shape)
    assert (1, 32) == o.shape
    assert (1, 64) == g.bases.shape


@pytest.mark.skipif(not tf.test.is_gpu_available(), reason="No GPU")
def test_simple_network():
    s = SimpleNetwork(1, 32)
    t = Trial()
    setattr(t, 'hidden1', 16)
    setattr(t, 'hidden2', 32)
    setattr(t, 'lr', 0.01)
    setattr(t, 'batchsize', 64)
    s.train([t,], [0.2,])
