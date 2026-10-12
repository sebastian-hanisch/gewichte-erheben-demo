"""Erhebungsmethoden gegen Handrechnung an der Mini-Instanz (conftest) und kleinen Gegenbeispielen."""

import math

import numpy as np
import pytest

import gwe_constants as C
import gwe_elicit as el
from entscheidungsmatrix_generator import SplitMix64


def test_weights_from_points():
    assert el.weights_from_points(np.array([100.0, 50.0, 50.0])).tolist() == [0.5, 0.25, 0.25]
    assert el.weights_from_points(np.zeros(4)).tolist() == [0.25] * 4


def test_rank_formulas_by_hand(w):
    ranks = np.array([1, 2, 3])
    assert el.weights_rank_sum(ranks).tolist() == pytest.approx([3 / 6, 2 / 6, 1 / 6])
    assert el.weights_rank_reciprocal(ranks).tolist() == pytest.approx([6 / 11, 3 / 11, 2 / 11])
    roc = el.weights_roc(ranks)
    assert roc.tolist() == pytest.approx([(1 + 1 / 2 + 1 / 3) / 3, (1 / 2 + 1 / 3) / 3, (1 / 3) / 3])
    assert el.tv_distance(roc, w) == pytest.approx(0.0222, abs=1e-4)
    assert el.tv_distance(el.weights_rank_sum(ranks), w) == pytest.approx(0.1)
    assert el.tv_distance(el.weights_rank_reciprocal(ranks), w) == pytest.approx(0.0818, abs=1e-4)


def test_rank_formulas_follow_the_rank_order_not_the_position():
    ranks = np.array([3, 1, 2])
    for f in (el.weights_rank_sum, el.weights_rank_reciprocal, el.weights_roc):
        v = f(ranks)
        assert v.sum() == pytest.approx(1.0)
        assert v[1] > v[2] > v[0]


def test_roc_with_one_criterion_and_two():
    assert el.weights_roc(np.array([1])).tolist() == [1.0]
    assert el.weights_roc(np.array([1, 2])).tolist() == pytest.approx([0.75, 0.25])


def test_swing_target_and_points_by_hand(w, swing_range):
    target = el.swing_target(w, swing_range)
    assert target.tolist() == pytest.approx([0.48 / 0.80, 0.27 / 0.80, 0.05 / 0.80])
    pts = el.points_swing(w, swing_range, 0.0, SplitMix64(1))
    assert pts.tolist() == [100.0, 56.0, 10.0]
    assert el.weights_from_points(pts).tolist() == pytest.approx([100 / 166, 56 / 166, 10 / 166])


def test_direct_points_without_noise_are_proportional_to_the_weights(w):
    pts = el.points_direct(w, 0.0, SplitMix64(1))
    assert pts.tolist() == [100.0, 50.0, 17.0]            # 0,1/0,6 * 100 = 16,67 -> 17


def test_swing_gives_the_largest_swing_exactly_100_even_with_noise(w, swing_range):
    for s in range(50):
        pts = el.points_swing(w, swing_range, 0.8, SplitMix64(s))
        assert pts.max() == 100.0 and pts[0] == 100.0 and (pts >= 0).all() and (pts <= 100).all()


def test_best_worst_with_consistent_judgments_returns_the_weights_exactly(w):
    v = el.weights_bwm(0, 2, np.array([1.0, 2.0, 6.0]), np.array([6.0, 3.0, 1.0]))
    assert v.tolist() == pytest.approx(w.tolist(), abs=1e-9)


def test_best_worst_judgments_without_noise_round_and_clip():
    w = np.array([0.8, 0.15, 0.05])                       # w_B / w_W = 16 -> 9
    j = el.judgments_bwm(w, 0.0, SplitMix64(1))
    assert (j["B"], j["W"]) == (0, 2)
    assert j["a_B"].tolist() == [1.0, 5.0, 9.0]            # 0,8/0,15 = 5,33 -> 5; 0,8/0,05 = 16 -> 9
    assert j["a_W"].tolist() == [9.0, 3.0, 1.0]            # 0,8/0,05 = 16 -> 9; 0,15/0,05 = 3
    assert j["clipped"] == 2                                # 16 (zum Besten) und 16 (zum Schlechtesten) über 9


def test_perceived_ranks_without_noise_follow_the_weights():
    assert el.perceived_ranks(np.array([0.1, 0.6, 0.3]), 0.0, SplitMix64(1)).tolist() == [3, 1, 2]


def test_perceived_ranks_with_noise_are_a_permutation():
    for s in range(30):
        r = el.perceived_ranks(np.array([0.3, 0.3, 0.2, 0.2]), 0.8, SplitMix64(s))
        assert sorted(r.tolist()) == [1, 2, 3, 4]


def test_elicit_dispatcher(w, swing_range):
    for method in ("gleich", "wahr", "direkt", "swing", "rangsumme", "rangrezipr", "roc", "bwm"):
        v, j = el.elicit(method, w, swing_range, 0.2, SplitMix64(3))
        assert v.sum() == pytest.approx(1.0) and (v >= 0).all(), method
    assert el.elicit("wahr", w, swing_range, 0.2, SplitMix64(3))[0].tolist() == w.tolist()
    assert el.elicit("gleich", w, swing_range, 0.2, SplitMix64(3))[0].tolist() == pytest.approx([1 / 3] * 3)
    with pytest.raises(ValueError):
        el.elicit("delphi", w, swing_range, 0.2, SplitMix64(3))


def test_judgment_counts():
    assert el.judgment_count("direkt", 5) == 5 and el.judgment_count("swing", 5) == 5
    assert el.judgment_count("bwm", 5) == 7 and el.judgment_count("bwm", 3) == 3
    assert el.judgment_count("roc", 5) == math.ceil(math.log2(120)) == 7
    assert el.judgment_count("gleich", 5) == 0 and el.judgment_count("wahr", 5) == 0


def test_tv_distance_bounds():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert el.tv_distance(a, b) == 1.0 and el.tv_distance(a, a) == 0.0


def test_methods_are_listed():
    assert set(C.METHODS) == set(C.METHOD_LABELS) == set(C.METHOD_SHORT) and set(C.RANK_METHODS) <= set(C.METHODS)
