import os
import unittest
import importlib

import pytest
import numpy as np

from tfdbonas.optimizer import DNGO
from tfdbonas.trial import TrialGenerator


@pytest.fixture(
    params=[
        {'n_trials': 10, 'n_remains': 990},
        {'n_trials': 500, 'n_remains': 500},
        {'n_trials': 1, 'n_remains': 999},
        {'n_trials': 1000, 'n_remains': 0},
    ]
)
def correct_tirals_test_case(request):
    return request.param

def test_dngo_random_search(correct_tirals_test_case):
    data = correct_tirals_test_case
    def objective(trial):
        return trial.lr * trial.bs

    params = [
        ['lr', [0.1 * i for i in range(10)]],
        ['bs', [64 * i for i in range(10)]],
        ['network', [64 * i for i in range(10)]],
    ]
    t = TrialGenerator()
    for inputs in params:
        t.register(inputs[0], inputs[1])
    algo = DNGO(t)
    assert len(t) == 1000
    assert data['n_remains'] == len(algo._random_search(objective, data['n_trials']))
    assert data['n_trials'] == len(algo._searched_trial_indices)
    assert data['n_trials'] == len(algo.results)


class TestDNGO(unittest.TestCase):
    @staticmethod
    def objective(trial) -> float:
        return trial.hidden1 * trial.hidden2 * trial.lr * trial.batchsize

    def setUp(self):
        params = [
            ['hidden1', [16, 32, 64, 128]],
            ['hidden2', [16, 32, 64, 128]],
            ['lr', [0.1 * i for i in range(10)]],
            ['batchsize', [64 * i for i in range(10)]],
        ]
        self.trial_generator = TrialGenerator()
        for inputs in params:
            self.trial_generator.register(inputs[0], inputs[1])

    def test_dngo_bayes_search(self):
        algo = DNGO(self.trial_generator)
        n_random = 10
        n_bayes = 10
        path = 'tfdbonas.deep_surrogate_models:SimpleNetwork'
        model_kwargs = dict(input_dim=4)
        algo._random_search(TestDNGO.objective, n_random)
        algo._bayes_search(TestDNGO.objective, n_bayes, path, model_kwargs)

    def test__calc_marginal_log_likelihood(self):
        optimizer = DNGO(self.trial_generator)
        n_random = 10
        n_bayes = 10
        path = 'tfdbonas.deep_surrogate_models:SimpleNetwork'
        theta = np.random.rand(2)
        phi = np.random.rand(10, 10)
        y_values = np.random.rand(10)
        optimizer._calc_marginal_log_likelihood(theta, phi, y_values, 10, 10)

    def test__predict(self):
        optimizer = DNGO(self.trial_generator)
        n_random = 10
        n_bayes = 10
        optimizer._deep_surrogate_model_restore_path = f'/tmp/test_model_predict{os.getpid()}.ckpt'
        path = 'tfdbonas.deep_surrogate_models:SimpleNetwork'
        theta = np.random.rand(2)
        searched_trial_indices = [1, 2, 3]
        deep_surrogate_model = self.load_class(path)()
        results = {str(i): i for i in range(3)}
        remained_trial_indices = [4, 5, 6]
        optimizer.k_inv = np.random.rand(64, 64)
        optimizer.mat = np.random.rand(64, 64)
        n_epochs = 1
        trained_bases = optimizer._train_deep_surrogate_model(searched_trial_indices,
                                                              results,
                                                              deep_surrogate_model,
                                                              n_epochs)
        mean, var = optimizer._predict(theta,
                                       remained_trial_indices,
                                       deep_surrogate_model)


    def test__train_deep_surrogate_model(self):
        optimizer = DNGO(self.trial_generator)
        path = 'tfdbonas.deep_surrogate_models:SimpleNetwork'
        theta = np.random.rand(2)
        searched_trial_indices = [1, 2, 3]
        deep_surrogate_model = self.load_class(path)()
        results = {str(i): i for i in range(3)}
        n_epochs = 1
        trained_bases = optimizer._train_deep_surrogate_model(searched_trial_indices,
                                                              results,
                                                              deep_surrogate_model,
                                                              n_epochs)

    def load_class(self, path):
        splited_path = path.split(':')
        assert len(splited_path) == 2, f'invalid input {splited_path}'
        module, class_name = splited_path
        module = importlib.import_module(module)
        return getattr(module, class_name)

    def test__predict_deep_surrogate_model(self):
        optimizer = DNGO(self.trial_generator)
        n_random = 10
        n_bayes = 10
        optimizer._deep_surrogate_model_restore_path = f'/tmp/test_model_{os.getpid()}.ckpt'
        path = 'tfdbonas.deep_surrogate_models:SimpleNetwork'
        theta = np.random.rand(2)
        searched_trial_indices = [1, 2, 3]
        deep_surrogate_model = self.load_class(path)()
        results = {str(i): i for i in range(3)}
        n_epochs = 1
        trained_bases = optimizer._train_deep_surrogate_model(searched_trial_indices,
                                                              results,
                                                              deep_surrogate_model,
                                                              n_epochs)
        trained_bases = optimizer._predict_deep_surrogate_model(searched_trial_indices, deep_surrogate_model)

    def test__calc_acq_values_ai(self):
        optimizer = DNGO(self.trial_generator)
        mean = np.random.rand(10)
        var = np.random.rand(10)
        results = {1: 1*i for i in range(1)}
        optimizer._calc_acq_value(mean, var, results)
