from tfdbonas.trial import Trial, TrialGenerator

import pytest


class Params:
    def __init__(self, param):
        self.param = param

    def __eq__(self, other):
        if not isinstance(other, Params):
            return NotImplemented
        return self.param == other.param

    def __str__(self):
        return 'Params'

    def __repr__(self):
        return 'Params'

#### Test for Trial ####

@pytest.fixture(
        params=[
            [['hello', 1], [1, '{hello: 1}']],
            [['hello', 'str'], ['str', '{hello: str}']],
            [['hello', 0.0], [0.0, '{hello: 0.0}']],
            [['object', Params(1)], [Params(1), '{object: Params}']],
        ]
)
def setattr_one_registered_samples_for_trial(request):
    input = request.param[0]
    expected = request.param[1]
    return input, expected

@pytest.fixture(
        params=[
            [[['int', 1], 1],
             [['string', 'str'], 'str'],
             [['float', 0.0], 0.0]],
            [[['int', 1], 1],
             [['string', 'str'], 'str'],
             [['float', 0.0], 0.0]],
        ]
)
def setattr_multiple_registered_samples_for_trial(request):
    input = request.param[0]
    expected = request.param[1]
    return input, expected

def test_trial_instantiation(setattr_one_registered_samples_for_trial):
    input = setattr_one_registered_samples_for_trial[0]
    expected = setattr_one_registered_samples_for_trial[1]
    t = Trial()
    setattr(t, *input)
    assert getattr(t, input[0]) == expected[0]
    assert str(t) == expected[1]

def test_multiple_trial_instantiation(setattr_multiple_registered_samples_for_trial):
    t = Trial()
    for samples in setattr_multiple_registered_samples_for_trial:
        input = samples[0]
        setattr(t, *input)
    for samples in setattr_multiple_registered_samples_for_trial:
        input = samples[0]
        expected = samples[1]
        assert getattr(t, input[0]) == expected


#### Test for TrialGenerator ####

def test_trialgenerator_register():
    t = TrialGenerator()
    assert len(t) == 0
    params = [
        ['ints', list(range(10))],
        ['strs', [f'str{i}' for i in range(10)]],
        ['floats', [0.1*i for i in range(10)]],
    ]

    for inputs in params:
        t.register(inputs[0], inputs[1])

    assert len(t) == 1000
    assert t[0]._elements == {'ints': 0,
                              'strs': 'str0',
                              'floats': 0.0}
    tt = Trial()
    tt.ints = 0
    tt.strs = 'str0'
    tt.floats = 0.0

    assert t[0] == tt

    assert t[998]._elements == {'ints': 8,
                                'strs': 'str9',
                                'floats': 0.9}
    assert t[999]._elements == {'ints': 9,
                                'strs': 'str9',
                                'floats': 0.9}

@pytest.mark.xfail
def test_trialgenerator_register_IndexError():
    t = TrialGenerator()
    assert len(t) == 0
    params = [
        ['ints', list(range(10))],
        ['strs', [f'str{i}' for i in range(10)]],
        ['floats', [0.1*i for i in range(10)]],
    ]

    for inputs in params:
        t.register(inputs[0], inputs[1])
    print(t[1000])
