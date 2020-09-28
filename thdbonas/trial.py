import typing
from enum import Flag, auto

import numpy as np


class Params(Flag):
    NN = auto()


class Trial:
    ''' this class is only accessed by TrialGenerator.
    '''
    def __init__(self):
        self._elements = {}
        del self._elements['_elements'] # remove self setattr

    def __setattr__(self, name: str, value):
        super.__setattr__(self, name, value)
        self._elements[name] = value

    def __eq__(self, other: dict or 'Trial'):
        if isinstance(other, Trial):
            return self._elements == other._elements
        elif isinstance(other, dict):
            return self._elements == other
        else:
            return NotImplemented
    def __len__(self):
        ''' return number of elements
        '''
        return len(self._elements)

    def __str__(self):
        o = "{"
        for k, v in self._elements.items():
            if k == '_elements':
                continue
            o += f"{k}: {v}, "
        o = o[:-2] # remove the last comma.
        o += "}"
        return o

    def to_dict(self):
        return self._elements

    def to_numpy(self):
        '''if elements are one values, return
        '''
        return np.array([v for k, v in self._elements.items()])


class TrialGenerator:
    def __init__(self):
        self._registered: typing.Dict[str, list] = {}
        self._registered_length: typing.Dict[str, int] = {}
        self.trial = Trial()
        self._len = 1
        self._n_features = 0

    def register(self, name: str, trials: typing.List) -> None:
        assert len(trials) != 0, "can't accept empty trials."
        if name in self._registered.keys():
            if self._registered[name] == trials:
                return
            self._len //= len(self._registered[name])
            # trials are updated
        self._registered[name] = trials
        self._registered_length[name] = len(trials)
        setattr(self.trial, name, None)
        self._len *= len(trials)
        self._n_features += 1

    @property
    def n_features(self):
        return self._n_features

    @n_features.getter
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, value):
        raise NotImplementedError

    def __getitem__(self, index: int) -> Trial:
        if index >= self._len:
            raise IndexError(f"len(self) => {self._len}, your index is invalid[{index}].")
        indices: typing.Dict[str, int] = {}
        trial = Trial()
        for idx, (k, n) in enumerate(self._registered_length.items()):
            if (idx + 1) == len(self._registered):
                # final key
                indices[k] = index % n
            else:
                remain = index % n
                indices[k] = remain
                index //= n
        for k, n in indices.items():
            setattr(trial, k, self._registered[k][n])
        return trial

    def __len__(self):
        if len(self._registered) == 0:
            return 0
        return self._len

    def __str__(self):
        o = ""
        for k, i in self._registered.items():
            o += f"{k} : {i}\n"
        return o
