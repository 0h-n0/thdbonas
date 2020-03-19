#!/usr/bin/env python
import math

from thdbonas import Searcher, Trial
import numpy as np


def objectve(trial: Trial):
    # (x*, y*) = [1, 1]
    # f(x*, y*) = 0
    o = 100*(trial.y - trial.x**2)**2 + (trial.x - 1)**2
    return -o


if __name__ == '__main__':
    searcher = Searcher()
    searcher.register_trial('x', np.arange(-2.048, 2.048, 0.02))
    searcher.register_trial('y', np.arange(-2.048, 2.048, 0.02))
    model_kwargs = dict(
        input_dim=2, # coresponding to the number of register_trial
        n_train_epochs=200,
    )
    n_trials = 20
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='thdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=10,
                        model_kwargs=model_kwargs)
    assert len(searcher.result) == n_trials
    print('results = {}'.format(searcher.result))
    print('best_trial {}'.format(searcher.best_trial))
    print('best_value {}'.format(searcher.best_value))
