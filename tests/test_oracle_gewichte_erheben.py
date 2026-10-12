"""Orakel-Tests gewichte-erheben-demo: Erhebungsmethoden gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12).

Anderer Rechenweg heißt hier: Definitionsschleifen, Erwartungswert über Zufallsziehungen, ein zweites LP in anderer Aufstellung, Gittersuche und Eigenschaften, die aus
der Definition folgen (Swing mit voller Spannweite, Dirichlet-Erwartung, Verteilung des Rauschens).
"""

import math

import numpy as np
import pytest

import entscheidungsmatrix_generator as gen
import gwe_elicit as el
import gwe_evaluation as ev
import gwe_score as sc
from entscheidungsmatrix_generator import SplitMix64


def loop_roc(ranks):
    m = len(ranks)
    out = []
    for r in ranks:
        out.append(sum(1.0 / j for j in range(int(r), m + 1)) / m)
    return out


def test_rank_formulas_match_the_definition_loops():
    rng = np.random.default_rng(0)
    for m in (3, 4, 5, 8):
        for _ in range(20):
            ranks = rng.permutation(m) + 1
            assert el.weights_roc(ranks).tolist() == pytest.approx(loop_roc(ranks))
            rs = [(m + 1 - r) for r in ranks]
            assert el.weights_rank_sum(ranks).tolist() == pytest.approx([x / sum(rs) for x in rs])
            rr = [1.0 / r for r in ranks]
            assert el.weights_rank_reciprocal(ranks).tolist() == pytest.approx([x / sum(rr) for x in rr])


@pytest.mark.parametrize("m", [3, 5, 8])
def test_roc_is_the_expected_ordered_weight_under_a_uniform_simplex(m):
    rng = SplitMix64(100 + m)
    draws = np.array([np.sort(ev.draw_weights(m, 1.0, rng))[::-1] for _ in range(6000)])
    formula = el.weights_roc(np.arange(1, m + 1))
    assert draws.mean(axis=0) == pytest.approx(formula, abs=0.006)


def test_roc_has_the_smallest_expected_distance_among_the_rank_formulas_for_a_uniform_simplex():
    rng = SplitMix64(7)
    m = 5
    err = {"gleich": 0.0, "rangsumme": 0.0, "rangrezipr": 0.0, "roc": 0.0}
    for _ in range(3000):
        w = ev.draw_weights(m, 1.0, rng)
        ranks = np.empty(m, dtype=int)
        ranks[np.lexsort((np.arange(m), -w))] = np.arange(1, m + 1)
        cand = {"gleich": np.full(m, 1.0 / m), "rangsumme": el.weights_rank_sum(ranks), "rangrezipr": el.weights_rank_reciprocal(ranks), "roc": el.weights_roc(ranks)}
        for k, v in cand.items():
            err[k] += float(((v - w) ** 2).sum())
    assert err["roc"] < err["rangrezipr"] < err["rangsumme"] < err["gleich"]


def test_roc_is_worse_than_the_rank_sum_when_weights_are_almost_equal():
    rng = SplitMix64(8)
    m = 5
    tv = {"rangsumme": 0.0, "roc": 0.0}
    for _ in range(3000):
        w = ev.draw_weights(m, 5.0, rng)
        ranks = np.empty(m, dtype=int)
        ranks[np.lexsort((np.arange(m), -w))] = np.arange(1, m + 1)
        tv["rangsumme"] += el.tv_distance(el.weights_rank_sum(ranks), w)
        tv["roc"] += el.tv_distance(el.weights_roc(ranks), w)
    assert tv["roc"] > tv["rangsumme"]                      # Gegenprobe: ROC ist nicht überall die beste Rangformel


@pytest.mark.parametrize("alpha", [0.5, 1.0, 5.0])
def test_drawn_weights_follow_the_dirichlet_moments(alpha):
    rng = SplitMix64(11)
    m = 4
    W = np.array([ev.draw_weights(m, alpha, rng) for _ in range(8000)])
    assert (W.sum(axis=1) == pytest.approx(1.0)) and (W >= 0).all()
    assert W.mean(axis=0) == pytest.approx([1 / m] * m, abs=0.01)                       # Erwartungswert alpha / (m alpha)
    var = (1 / m) * (1 - 1 / m) / (m * alpha + 1)                                       # Varianz einer Komponente der Dirichlet-Verteilung
    assert W.var(axis=0) == pytest.approx([var] * m, rel=0.12)


def test_best_worst_lp_is_optimal_against_random_candidates_and_a_grid():
    """Optimalität des LP: kein zufälliger Kandidat aus dem Simplex (20 000 Dirichlet-Ziehungen, davon die Hälfte nahe der Lösung) und kein Gitterpunkt hat einen
    kleineren Wert von xi = max(|w_B - a_Bj w_j|, |w_j - a_jW w_W|) als die LP-Lösung."""
    rng = SplitMix64(3)
    np_rng = np.random.default_rng(3)
    grid = np.array([(i / 40, j / 40, (40 - i - j) / 40) for i in range(41) for j in range(41 - i)])
    for _ in range(12):
        w = ev.draw_weights(3, 1.0, rng)
        j = el.judgments_bwm(w, 0.3, rng)
        v = el.weights_bwm(j["B"], j["W"], j["a_B"], j["a_W"])
        B, W = j["B"], j["W"]

        def xi(X):
            X = np.atleast_2d(X)
            terms = []
            for k in range(3):
                if k != B:
                    terms.append(np.abs(X[:, B] - j["a_B"][k] * X[:, k]))
                if k != W:
                    terms.append(np.abs(X[:, k] - j["a_W"][k] * X[:, W]))
            return np.max(terms, axis=0)

        near = np.clip(v + 0.05 * np_rng.standard_normal((10000, 3)), 0, None)
        near = near / near.sum(axis=1, keepdims=True)
        candidates = np.vstack([np_rng.dirichlet(np.ones(3), size=10000), near, grid])
        assert xi(v)[0] <= xi(candidates).min() + 1e-9
        assert v.sum() == pytest.approx(1.0) and (v >= 0).all()


def test_swing_with_full_range_and_no_noise_is_proportional_to_the_weights():
    rng = np.random.default_rng(1)
    for _ in range(20):
        w = rng.dirichlet(np.ones(5))
        pts = el.points_swing(w, np.ones(5), 0.0, SplitMix64(1))
        got = el.weights_from_points(pts)
        assert np.abs(got - w).max() < 0.006                # Rundung auf ganze Punkte


def test_swing_weights_are_exact_for_a_linear_in_raw_decision_maker_up_to_rounding():
    """Entscheider linear in den Rohwerten, Min-Max, Swing-Gewichte ohne Rauschen: dieselbe Rangfolge wie der Nutzen (bis auf die Rundung auf Punkte)."""
    agree = 0
    for s in range(60):
        c = ev.make_case(20, 5, s, 0.2, "roh")
        res = ev.evaluate_methods(c, 0.0, s, ("wahr", "swing"))
        assert res["swing"].tau > 0.97
        agree += res["swing"].top1 == sc.top1(c.utility)
    assert agree >= 55                                       # fast immer Platz 1; die Rundung auf 100 Punkte kippt einzelne Gleichstände


def test_true_weights_miss_the_best_more_often_than_swing_for_a_narrow_criterion():
    wins = 0
    for s in range(120):
        c = ev.make_case(20, 5, s, 0.05, "roh")
        res = ev.evaluate_methods(c, 0.0, s, ("wahr", "swing"))
        wins += res["swing"].hit and not res["wahr"].hit
    assert wins > 15                                         # Gegenprobe zum vorigen Test: die wahren Gewichte sind hier nicht die passenden


def test_judgment_noise_is_lognormal_with_the_chosen_sigma():
    """Zweites Kriterium: w / w_max = 0,2, also 20 Punkte mal exp(sigma eps); die Obergrenze 100 wird bei sigma = 0,4 nie erreicht (5 / 0,4 = 4 Standardabweichungen
    im Logarithmus, ln 5 = 1,61 = 4,0 sigma), die Rundung auf ganze Punkte ist klein gegen 20."""
    rng = SplitMix64(5)
    sigma = 0.4
    w = np.array([1.0, 0.2])
    col = np.array([el.points_direct(w, sigma, rng)[1] for _ in range(20000)])
    assert (col > 0).all() and (col < 100).all()
    logs = np.log(col / 20.0)
    assert logs.std() == pytest.approx(sigma, abs=0.02)
    assert logs.mean() == pytest.approx(0.0, abs=0.01)


def test_rounding_and_clipping_at_the_edges():
    pts = el.points_direct(np.array([0.9, 0.1]), 3.0, SplitMix64(2))
    assert pts.min() >= 0 and pts.max() <= 100 and (pts == np.round(pts)).all()
    j = el.judgments_bwm(np.array([0.7, 0.2, 0.1]), 3.0, SplitMix64(2))
    assert j["a_B"].min() >= 1 and j["a_B"].max() <= 9 and j["a_W"].max() <= 9


def test_narrowing_keeps_the_dominance_structure_of_the_other_columns():
    x = gen.make_matrix(15, 3, 0.0, gen.SplitMix64(4))
    g = sc.narrow_criterion(x.goodness, 1.0)
    assert (g == x.goodness).all()                           # delta = 1: unverändert, daher dieselbe Dominanz (Stück-1-Eigenschaft als Gegenprobe)


def test_mini_instance_values_by_independent_loop(w):
    ranks = [1, 2, 3]
    roc = loop_roc(ranks)
    assert roc == pytest.approx([0.6111111, 0.2777778, 0.1111111], abs=1e-6)
    tv = 0.5 * sum(abs(a - b) for a, b in zip(roc, w))
    assert tv == pytest.approx(0.0222222, abs=1e-6)
    assert math.isclose(sum(roc), 1.0)
