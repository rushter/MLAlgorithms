from __future__ import division

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from mla.metrics.base import check_data, validate_input
from mla.metrics.metrics import *


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
    assert metric('classification_error')([1, 2, 3, 4], [1, 2, 3, 4]) == 0
    assert metric('classification_error')([1, 2, 3, 4], [1, 2, 3, 5]) == 0.25
    assert metric('classification_error')([1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]) == (1.0 / 6)


def test_absolute_error():
    assert metric('absolute_error')([3], [5]) == [2]
    assert metric('absolute_error')([-1], [-4]) == [3]


def test_mean_absolute_error():
    assert metric('mean_absolute_error')([1, 2, 3], [1, 2, 3]) == 0
    assert metric('mean_absolute_error')([1, 2, 3], [3, 2, 1]) == 4 / 3


def test_squared_error():
    assert metric('squared_error')([1], [1]) == [0]
    assert metric('squared_error')([3], [1]) == [4]


def test_squared_log_error():
    assert metric('squared_log_error')([1], [1]) == [0]
    assert metric('squared_log_error')([3], [1]) == [np.log(2) ** 2]
    assert metric('squared_log_error')([np.exp(2) - 1], [np.exp(1) - 1]) == [1.0]


def test_mean_squered_error():
    assert metric('mean_squared_log_error')([1, 2, 3], [1, 2, 3]) == 0
    assert metric('mean_squared_log_error')([1, 2, 3, np.exp(1) - 1], [1, 2, 3, np.exp(2) - 1]) == 0.25


def test_root_mean_squared_log_error():
    assert metric('root_mean_squared_log_error')([1, 2, 3], [1, 2, 3]) == 0
    assert metric('root_mean_squared_log_error')([1, 2, 3, np.exp(1) - 1], [1, 2, 3, np.exp(2) - 1]) == 0.5


def test_mean_squared_error():
    assert metric('mean_squared_error')([1, 2, 3], [1, 2, 3]) == 0
    assert metric('mean_squared_error')(range(1, 5), [1, 2, 3, 6]) == 1


def test_root_mean_squared_error():
    assert metric('root_mean_squared_error')([1, 2, 3], [1, 2, 3]) == 0
    assert metric('root_mean_squared_error')(range(1, 5), [1, 2, 3, 5]) == 0.5


def test_multiclass_logloss():
    assert_almost_equal(metric('logloss')([1], [1]), 0)
    assert_almost_equal(metric('logloss')([1, 1], [1, 1]), 0)
    assert_almost_equal(metric('logloss')([1], [0.5]), -np.log(0.5))
