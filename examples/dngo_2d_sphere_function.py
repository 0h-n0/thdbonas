#!/usr/bin/env python
import warnings

from thdbonas import Searcher, Trial
import numpy as np


def objectve(trial: Trial):
    return -(trial.x**2 + trial.y**2)


if __name__ == '__main__':
    searcher = Searcher()
    searcher.register_trial('x', np.arange(-10, 10, 0.1))
    searcher.register_trial('y', np.arange(-10, 10, 0.1))
    model_kwargs = dict(
        input_dim=2, # coresponding to the number of register_trial
        n_train_epochs=200,
    )
    n_trials = 1000
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='thdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=5,
                        model_kwargs=model_kwargs)
    assert len(searcher.result) == n_trials
    warnings.warn('results = {}'.format(searcher.result))
    warnings.warn('best_trial {}'.format(searcher.best_trial))
    warnings.warn('best_value {}'.format(searcher.best_value))
