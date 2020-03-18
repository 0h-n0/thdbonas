#!/usr/bin/env python
import warnings

import pytest
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.examples.tutorials.mnist import input_data
from tfdbonas import Searcher, Trial


def create_simple_network_model(hidden_size=128, activation='relu'):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(hidden_size, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model


def objectve(trial: Trial):
    model = create_simple_network_model(trial.hidden_size)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    opt = tf.keras.optimizers.SGD(learning_rate=trial.lr)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    model.fit(x_train, y_train, epochs=1, batch_size=trial.batchsize)
    out = model.evaluate(x_test,  y_test, verbose=2)
    accuracy = out[1]
    return accuracy

@pytest.mark.heavy
def test_simple_network():
    searcher = Searcher()
    searcher.register_trial('hidden_size', [64, 128, 256, 512, 1024])
    searcher.register_trial('batchsize', [32, 64, 128, 256, 512, 1024])
    searcher.register_trial('lr', [0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    n_trials = 30

    model_kwargs = dict(
        input_dim=3,
        n_train_epochs=200,
    )
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='tfdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=10,

                             model_kwargs=model_kwargs)
    assert len(searcher.result) == n_trials
    warnings.warn('results = {}'.format(searcher.result))
    warnings.warn('best_trial {}'.format(searcher.best_trial))
    warnings.warn('best_value {}'.format(searcher.best_value))

@pytest.mark.xfail
def test_simple_network_with_categorical_features():
    # (False, reason="categorical feature is not supported yet. [idea] I shoud apply embedding layers for categorical inputs.")
    searcher = Searcher()
    searcher.register_trial('hidden_size', [64, 128, 256, 512, 1024])
    searcher.register_trial('batchsize', [32, 64, 128, 256, 512, 1024])
    searcher.register_trial('lr', [0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

    searcher.register_trial('activation', [None, 'relu', 'tanh'])
    #
    n_trials = 30

    model_kwargs = dict(
        input_dim=3,
        n_train_epochs=200,
    )
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='tfdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=10,

                             model_kwargs=model_kwargs)
    assert len(searcher.result) == n_trials
    warnings.warn('results = {}'.format(searcher.result))
    warnings.warn('best_trial {}'.format(searcher.best_trial))
    warnings.warn('best_value {}'.format(searcher.best_value))

def create_cnn_model(trial):
    model = keras.Sequential([
        keras.layers.Conv2D(trial.cnn_h1, trial.cnn_k1,
                            (trial.cnn_s1, trial.cnn_s1),
                            activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(trial.pool_k1),
        keras.layers.Conv2D(trial.cnn_h2, trial.cnn_k2,
                            (trial.cnn_s2, trial.cnn_s2),
                            activation='relu'),
        keras.layers.MaxPooling2D(trial.pool_k2),
        keras.layers.Conv2D(trial.cnn_h3, trial.cnn_k3,
                            (trial.cnn_s3, trial.cnn_s3),
                            activation='relu'),
        keras.layers.MaxPooling2D(trial.pool_k3),
        keras.layers.Flatten(),
        keras.layers.Dense(trial.fc1, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    return model

def cnn_objectve(trial: Trial):
    try:
        model = create_cnn_model(trial)
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape((60000, 28, 28, 1))
        x_test = x_test.reshape((10000, 28, 28, 1))
        x_train, x_test = x_train / 255.0, x_test / 255.0

        opt = tf.keras.optimizers.SGD(learning_rate=trial.lr)
        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        model.fit(x_train, y_train, epochs=1, batch_size=trial.batchsize)
        out = model.evaluate(x_test,  y_test, verbose=2)
        accuracy = out[1]
        return accuracy
    except Exception:
        # Graph construction is failed.
        return 0.0

@pytest.mark.skipif(True, reason='heavy conputational cost.')
def test_cnn_model_with_many_trials():
    searcher = Searcher()
    searcher.register_trial('cnn_h1', [4, 8, 16, 32, 64])
    searcher.register_trial('cnn_k1', [1, 2, 3])
    searcher.register_trial('cnn_s1', [1, 2, 3])
    searcher.register_trial('pool_k1', [1, 2, 3])
    searcher.register_trial('cnn_h2', [4, 8, 16, 32, 64])
    searcher.register_trial('cnn_k2', [1, 2, 3])
    searcher.register_trial('cnn_s2', [1, 2, 3])
    searcher.register_trial('pool_k2', [1, 2, 3])
    searcher.register_trial('cnn_h3', [4, 8, 16, 32, 64])
    searcher.register_trial('cnn_k3', [1, 2, 3])
    searcher.register_trial('cnn_s3', [1, 2, 3])
    searcher.register_trial('pool_k3', [1, 2, 3])
    searcher.register_trial('fc1', [64, 128, 256, 512, 1024])
    searcher.register_trial('batchsize', [16, 32, 64, 128, 256, 512, 1024])
    searcher.register_trial('lr', [0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    n_trials = 30

    model_kwargs = dict(
        input_dim=15,
        n_train_epochs=200,
    )
    warnings.warn('CNN: the number of trials = {}'.format(len(searcher)))
    # CNN: the number of trials = 516678750
    _ = searcher.search(cnn_objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='tfdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=10,

                        model_kwargs=model_kwargs)
    assert len(searcher.result) == n_trials
    warnings.warn('CNN: results = {}'.format(searcher.result))
    warnings.warn('CNN: best_trial {}'.format(searcher.best_trial))
    warnings.warn('CNN: best_value {}'.format(searcher.best_value))

@pytest.mark.skipif(True, reason='memory error (32GB).')
def test_cnn_model_with_few_trials():
    searcher = Searcher()
    searcher.register_trial('cnn_h1', [16, 64])
    searcher.register_trial('cnn_k1', [1, 3])
    searcher.register_trial('cnn_s1', [1, 3])
    searcher.register_trial('pool_k1', [1, 3])
    searcher.register_trial('cnn_h2', [32, 64])
    searcher.register_trial('cnn_k2', [1, 2])
    searcher.register_trial('cnn_s2', [1, 2])
    searcher.register_trial('pool_k2', [2, 3])
    searcher.register_trial('cnn_h3', [64, 128])
    searcher.register_trial('cnn_k3', [1, 3])
    searcher.register_trial('cnn_s3', [1, 2])
    searcher.register_trial('pool_k3', [1, 3])
    searcher.register_trial('fc1', [64, 128, 256])
    searcher.register_trial('batchsize', [32, 64, 128])
    searcher.register_trial('lr', [0.05, 0.1])
    n_trials = 30

    model_kwargs = dict(
        input_dim=15,
        n_train_epochs=200,
    )
    warnings.warn('CNN: the number of trials = {}'.format(len(searcher)))
    # CNN: the number of trials = 73728
    # var = np.diag(np.matmul(np.matmul(predicted_bases, self.k_inv), predicted_bases.transpose()) + 1 / beta)
    # MemoryError: Unable to allocate array with shape (73718, 73718) and data type float64
    _ = searcher.search(cnn_objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='tfdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=10,

                        model_kwargs=model_kwargs)
    assert len(searcher.result) == n_trials
    warnings.warn('CNN: results = {}'.format(searcher.result))
    warnings.warn('CNN: best_trial {}'.format(searcher.best_trial))
    warnings.warn('CNN: best_value {}'.format(searcher.best_value))

@pytest.mark.heavy
def test_cnn_model_with_few_trials():
    searcher = Searcher()
    searcher.register_trial('cnn_h1', [16, 64])
    searcher.register_trial('cnn_k1', [1, 3])
    searcher.register_trial('cnn_s1', [1])
    searcher.register_trial('pool_k1', [1, 3])
    searcher.register_trial('cnn_h2', [32, 64])
    searcher.register_trial('cnn_k2', [1])
    searcher.register_trial('cnn_s2', [1])
    searcher.register_trial('pool_k2', [2, 3])
    searcher.register_trial('cnn_h3', [64, 128])
    searcher.register_trial('cnn_k3', [1, 3])
    searcher.register_trial('cnn_s3', [1])
    searcher.register_trial('pool_k3', [1, 3])
    searcher.register_trial('fc1', [64, 128, 256])
    searcher.register_trial('batchsize', [32, 64, 128])
    searcher.register_trial('lr', [0.05, 0.1])
    n_trials = 30

    model_kwargs = dict(
        input_dim=15,
        n_train_epochs=200,
    )
    warnings.warn('CNN: the number of trials = {}'.format(len(searcher)))
    _ = searcher.search(cnn_objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='tfdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=10,

                        model_kwargs=model_kwargs)
    assert len(searcher.result) == n_trials
    warnings.warn('CNN: results = {}'.format(searcher.result))
    warnings.warn('CNN: best_trial {}'.format(searcher.best_trial))
    warnings.warn('CNN: best_value {}'.format(searcher.best_value))
