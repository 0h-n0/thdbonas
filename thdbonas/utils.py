from enum import Flag, auto
import importlib

from .trial import Trial

import numpy as np


class Result:
    def __init__(self, n_remenbers: int = 10):
        self.trials = []
        self.score = []
        self.max_score: float = 0.0

    def push(self, score: float, trial: Trial):
        pass

    def top(self) -> float:
        return 0


class State(Flag):
    Initialized = auto()
    NotInitialized = auto()

def is_float(x) -> bool:
    if isinstance(x, float):
        return True
    if isinstance(x, np.float16):
        return True
    if isinstance(x, np.float32):
        return True
    if isinstance(x, np.float64):
        return True
    return False

def load_class(path: str):
    ''' load a class from path(: str).
    ```
    path = 'hoge.hoge.hoge:HogeClass'
    HogeClass = load_class(path)
    c = HogeClass()
    ```
    '''
    splited_path = path.split(':')
    assert len(splited_path) == 2, f'invalid input {splited_path}'
    module, class_name = splited_path
    module = importlib.import_module(module)
    return getattr(module, class_name)
