#!/usr/bin/env python
import math

from thdbonas import Searcher, Trial
import numpy as np


def objectve(trial: Trial):
    # https://en.wikipedia.org/wiki/Ackley_function
    # x* = 0
    # f(x*) = 0
    o = 20 + np.e
    o += -20 * np.exp(-0.2*(trial.x ** 2 / 2 + trial.y ** 2 / 2))
    o += -np.exp(np.cos(2 * math.pi * trial.x**2) / 2 + np.cos(2 * math.pi * trial.y**2) / 2)
    return -o


if __name__ == '__main__':
    searcher = Searcher()
    searcher.register_trial('x', np.arange(-30, 30, 0.5))
    searcher.register_trial('y', np.arange(-30, 30, 0.5))
    model_kwargs = dict(
        input_dim=2, # coresponding to the number of register_trial
        n_train_epochs=200,
    )
    n_trials = 100
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='thdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=10,
                        model_kwargs=model_kwargs)
    assert len(searcher.result) == n_trials
    print('results = {}'.format(searcher.result))
    print('best_trial {}'.format(searcher.best_trial))
    print('best_value {}'.format(searcher.best_value))
