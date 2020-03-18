import pytest

from tfdbonas.searcher import Searcher


def test_searcher():
    def ovjective(trial):
        return trial.lr * trial.bs

    params = [
        ['lr', [0.1 * i for i in range(10)]],
        ['bs', [64 * i for i in range(10)]],
    ]

    s = Searcher()

    for p in params:
        s.register_trial(p[0], p[1])
    assert len(s) == 100

@pytest.mark.skip
def test_searcher_with_integrated_test():
    def objective(trial):
        return trial.lr

    params = [
        ['lr', [0.1 * i for i in range(10)]],
        ['bs', [64 * i for i in range(10)]],
        ['network', [64 * i for i in range(10)]],
    ]

    s = Searcher()

    for p in params:
        s.register_trial(p[0], p[1])
    assert len(s) == 1000
    s.search(objective, 100, n_random_trials=10)
