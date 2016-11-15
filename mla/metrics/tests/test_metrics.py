from __future__ import division

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from mla.metrics.base import check_data, validate_input
from mla.metrics.metrics import get_metric


def test_data_validation():
    with pytest.raises(ValueError):
        check_data([], 1)

    with pytest.raises(ValueError):
        check_data([1, 2, 3], [3, 2])

    a, b = check_data([1, 2, 3], [3, 2, 1])

    assert np.all(a == np.array([1, 2, 3]))
    assert np.all(b == np.array([3, 2, 1]))


def metric(name):
    return validate_input(get_metric(name))


def test_classification_error():
    f = metric('classification_error')
    assert f([1, 2, 3, 4], [1, 2, 3, 4]) == 0
    assert f([1, 2, 3, 4], [1, 2, 3, 5]) == 0.25
    assert f([1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]) == (1.0 / 6)


def test_absolute_error():
    f = metric('absolute_error')
    assert f([3], [5]) == [2]
    assert f([-1], [-4]) == [3]


def test_mean_absolute_error():
    f = metric('mean_absolute_error')
    assert f([1, 2, 3], [1, 2, 3]) == 0
    assert f([1, 2, 3], [3, 2, 1]) == 4 / 3


def test_squared_error():
    f = metric('squared_error')
    assert f([1], [1]) == [0]
    assert f([3], [1]) == [4]


def test_squared_log_error():
    f = metric('squared_log_error')
    assert f([1], [1]) == [0]
    assert f([3], [1]) == [np.log(2) ** 2]
    assert f([np.exp(2) - 1], [np.exp(1) - 1]) == [1.0]


def test_mean_squared_log_error():
    f = metric('mean_squared_log_error')
    assert f([1, 2, 3], [1, 2, 3]) == 0
    assert f([1, 2, 3, np.exp(1) - 1], [1, 2, 3, np.exp(2) - 1]) == 0.25


def test_root_mean_squared_log_error():
    f = metric('root_mean_squared_log_error')
    assert f([1, 2, 3], [1, 2, 3]) == 0
    assert f([1, 2, 3, np.exp(1) - 1], [1, 2, 3, np.exp(2) - 1]) == 0.5


def test_mean_squared_error():
    f = metric('mean_squared_error')
    assert f([1, 2, 3], [1, 2, 3]) == 0
    assert f(range(1, 5), [1, 2, 3, 6]) == 1


def test_root_mean_squared_error():
    f = metric('root_mean_squared_error')
    assert f([1, 2, 3], [1, 2, 3]) == 0
    assert f(range(1, 5), [1, 2, 3, 5]) == 0.5


def test_multiclass_logloss():
    f = metric('logloss')
    assert_almost_equal(f([1], [1]), 0)
    assert_almost_equal(f([1, 1], [1, 1]), 0)
    assert_almost_equal(f([1], [0.5]), -np.log(0.5))
