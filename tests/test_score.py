"""Min-Max, Score, Wertskalen und Verschmälerung gegen Handrechnung."""

import numpy as np
import pytest

import entscheidungsmatrix_generator as gen
import gwe_score as sc


def test_minmax_by_hand_and_constant_column():
    S = np.array([[90.0, 400.0], [60.0, 900.0], [30.0, 1000.0]])
    assert sc.minmax(S) == pytest.approx(np.array([[1.0, 0.0], [0.5, 5 / 6], [0.0, 1.0]]))
    assert sc.minmax(np.array([[1.0, 5.0], [2.0, 5.0]])).tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_score_and_top1_with_ties():
    assert sc.weighted_score(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.5, 0.5])).tolist() == [0.5, 0.5]
    assert sc.top1(np.array([0.5, 0.7, 0.7])) == 1


def test_nominal_bounds_and_value_functions():
    lo, hi = sc.nominal_bounds(2)
    assert lo.tolist() == [0.0, -5000.0] and hi.tolist() == [100.0, -800.0]
    S = np.array([[100.0, -800.0], [0.0, -5000.0], [50.0, -2900.0]])
    f = sc.value_functions("roh", np.zeros((3, 2)), S, 2)
    assert f == pytest.approx(np.array([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]]))
    g = np.array([[0.2, 0.7]])
    assert sc.value_functions("guete", g, S[:1], 2).tolist() == [[0.2, 0.7]]
    with pytest.raises(ValueError):
        sc.value_functions("log", g, S[:1], 2)


def test_narrow_criterion_changes_only_the_chosen_column():
    g = np.array([[0.0, 0.2], [1.0, 0.8], [0.5, 0.5]])
    out = sc.narrow_criterion(g, 0.2)
    assert out[:, 0].tolist() == pytest.approx([0.4, 0.6, 0.5])
    assert (out[:, 1] == g[:, 1]).all()
    assert (sc.narrow_criterion(g, 1.0) == g).all()
    assert out is not g and g[0, 0] == 0.0                        # Eingabe unverändert


def test_build_case_with_unit_delta_is_the_generator_unchanged():
    x = gen.make_matrix(12, 4, 0.0, gen.SplitMix64(3))
    g, raw, S, f = sc.build_case(x.goodness, x.directions, 4, 1.0, "guete")
    assert (g == x.goodness).all() and (raw == x.raw).all() and (S == x.signed).all() and (f == x.goodness).all()


def test_narrowing_reduces_the_range_of_the_first_raw_column_only():
    x = gen.make_matrix(30, 3, 0.0, gen.SplitMix64(5))
    _, raw, _, _ = sc.build_case(x.goodness, x.directions, 3, 0.1, "roh")
    assert np.ptp(raw[:, 0]) < 0.2 * np.ptp(x.raw[:, 0])
    assert (raw[:, 1:] == x.raw[:, 1:]).all()


def test_utility_loss_and_kendall_tau_b():
    u = np.array([0.2, 0.9, 0.5])
    assert sc.utility_loss(u, 1) == 0.0 and sc.utility_loss(u, 0) == pytest.approx(0.7)
    assert sc.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert sc.kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert sc.kendall_tau_b([1, 1, 1], [1, 2, 3]) == 0.0
    assert sc.kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / np.sqrt(6))
