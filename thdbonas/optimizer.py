import math
import typing
import random
from enum import Flag, auto

import numpy as np
import scipy.optimize
import scipy.stats

from .trial import Trial, TrialGenerator
from .utils import State, load_class
from .acquistion_functions import AcquisitonFunction, AcquisitonFunctionType


class OptimizerType(Flag):
    DNGO = auto()


class DNGO:
    def __init__(self, trial_generator, acq_func_type=AcquisitonFunctionType.EI):
        self._trials_indices = list(range(len(trial_generator)))
        self.trial_generator = trial_generator
        self._state = State.NotInitialized
        self._searched_trial_indices: typing.List[int] = []
        self.results: typing.Dict[int, float] = {}
        self._deep_surrogate_model_restore_path = '/tmp/model.ckpt'
        self.acq_func = AcquisitonFunction(acq_func_type)

    def run(self, objective: typing.Callable[[Trial], float],
            n_trials: int, **kwargs):
        n_random_trials = kwargs['n_random_trials']
        deep_surrogate_model = kwargs['deep_surrogate_model']
        model_kwargs = kwargs['model_kwargs']
        _ = self._random_search(objective, n_random_trials)
        results = self._bayes_search(objective,
                                     n_trials - n_random_trials,
                                     deep_surrogate_model,
                                     model_kwargs)
        return results

    def _random_search(self,
                       objective: typing.Callable[[Trial], float],
                       n_trials: int) -> typing.List[int]:
        assert len(self._trials_indices) >= n_trials, (f'len(self._trials_indices) >= n_trials:'
                                                       f' {len(self._trials_indices)} >= {n_trials}')
        trial_indices = random.sample(self._trials_indices, n_trials)
        for i in trial_indices:
            self.results[i] = objective(self.trial_generator[i])
            self._trials_indices.remove(i)
        self._searched_trial_indices += trial_indices
        self._state = State.Initialized
        return self._trials_indices # return remained trials

    def _bayes_search(self,
                      objective: typing.Callable[[Trial], float],
                      n_trials: int,
                      deep_surrogate_model_path: str,
                      model_kwargs: typing.Dict) -> typing.List[int]:
        deep_surrogate_model_class = load_class(deep_surrogate_model_path)
        deep_surrogate_model = deep_surrogate_model_class(**model_kwargs)
        assert self._state == State.Initialized, ('not initialied: please call '
                                                  'self.random_search() before calling bayes_search.')
        assert len(self._searched_trial_indices) != 0, 'Before searching, you have to run random search.'
        for _ in range(n_trials):
            trained_bases = self._train_deep_surrogate_model(
                self._searched_trial_indices,
                self.results,
                deep_surrogate_model)
            n_samples = len(self._searched_trial_indices)
            n_features = self.trial_generator.n_features
            params = self._update_mll_params(trained_bases,
                                             self._searched_trial_indices,
                                             self.results,
                                             n_samples,
                                             n_features)
            mean, var = self._predict(params, self._trials_indices, deep_surrogate_model)
            acq_values = self._calc_acq_value(mean, var, self.results)
            next_sample_index = self._trials_indices[np.argmax(acq_values)]
            self._searched_trial_indices.append(next_sample_index)
            self._trials_indices.remove(next_sample_index)
            self.results[next_sample_index] = objective(self.trial_generator[next_sample_index])
        return self.results

    def _train_deep_surrogate_model(self,
                                    searched_trial_indices: typing.List[int],
                                    results: typing.Dict[int, float],
                                    deep_surrogate_model,
                                    n_training_epochs: int = 100):
        assert isinstance(n_training_epochs, int), f'invalid input type: type(n_training_epochs) {type(n_training_epochs)}'
        assert len(searched_trial_indices) == len(results), ('invalid inputs, searched_trial_indices[{searched_trial_indices}] '
                                                             'and results[{results}] must be the same length.')
        searched_trials = [self.trial_generator[i] for i in searched_trial_indices]
        trained_bases = deep_surrogate_model.train(searched_trials, results, n_training_epochs)
        return trained_bases

    def _predict_deep_surrogate_model(self,
                                      non_searched_trial_indices: typing.List[int],
                                      deep_surrogate_model):
        non_searched_trials = [self.trial_generator[i] for i in non_searched_trial_indices]
        predicted_bases = deep_surrogate_model.predict(non_searched_trials)
        return predicted_bases

    def _predict(self, params, remained_trial_indicees, deep_surrogate_model):
        _, beta = np.exp(params)
        predicted_bases = self._predict_deep_surrogate_model(remained_trial_indicees,
                                                             deep_surrogate_model)
        mean = np.matmul(predicted_bases, self.mat)
        var = np.diag(np.matmul(np.matmul(predicted_bases, self.k_inv), predicted_bases.transpose()) + 1 / beta)
        return mean, var

    def _calc_acq_value(self, mean, var, results):
        # TODO: current version is just for EI.
        min_val = np.float32(np.min(list(results.values())))
        return self.acq_func(mean, var, min_val)

    def _update_mll_params(self, bases, searched_trial_indices,
                           results, n_samples, n_features):

        y_values = np.array([results[i] for i in searched_trial_indices])
        params = scipy.optimize.fmin(self._calc_marginal_log_likelihood,
                                     np.random.rand(2),
                                     args=(bases, y_values, n_samples, n_features))
        return params

    def _calc_marginal_log_likelihood(self,
                                      theta,
                                      phi,
                                      y_values,
                                      n_samples,
                                      n_features):
        # TODO: input type check
        assert theta.size == 2, f"invalid input: theta => {theta}"
        assert len(theta.shape) == 1, f"invalid input: theta => {theta}"
        assert y_values.size == n_samples, f"invalid input: y_values.size => {y_values.size}"
        alpha, beta = np.exp(theta)

        # calculate K matrix
        identity = np.eye(phi.shape[1])
        phi_t = phi.transpose(1, 0)
        k_mat = beta * np.matmul(phi_t, phi) + alpha * identity

        # calculate mat
        k_inv = np.linalg.inv(k_mat)
        mat = beta * np.matmul(k_inv, phi_t)
        mat = np.matmul(mat, y_values)

        self.mat = np.float32(mat)
        self.k_inv = np.float32(k_inv)
        mll = n_features / 2. * np.log(alpha)
        mll += n_samples / 2. * np.log(beta)
        mll -= n_samples / 2. * np.log(2 * math.pi)
        mll -= beta / 2. * np.linalg.norm(y_values - np.matmul(phi, mat))
        mll -= alpha / 2. * mat.dot(mat)
        mll -= 0.5 * np.log(np.linalg.det(k_mat))
        return -mll
