"""Jede Zahl aus README.md und App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SWEEP_SEEDS (200 feste Matrizen), nie über einen einzelnen Seed. Prozentwerte mit Toleranz 0,5 Punkte, Gewichtsabstände mit 0,01,
τ mit 0,02, damit Rundung und Bibliotheksversionen (CI installiert ungepinnt) nichts kippen. Der Standardfehler einer Trefferquote ist 1,3 bis 3,5 Punkte:
Unterschiede unter etwa 6 Punkten sind NICHT belastbar und stehen nirgends als Aussage.
"""

import pytest

import gwe_constants as C
import gwe_evaluation as ev


@pytest.fixture(scope="module")
def methods():
    return ev.experiment_methods()


@pytest.fixture(scope="module")
def narrow():
    return ev.experiment_narrow()


@pytest.fixture(scope="module")
def rank():
    return ev.experiment_rank_formulas()


@pytest.fixture(scope="module")
def noise():
    return ev.experiment_noise()


@pytest.fixture(scope="module")
def clip():
    return ev.bwm_clip_stats()


def cell(rows, **keys):
    hits = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(hits) == 1, (keys, len(hits))
    return hits[0]


def pct(x):
    return 100 * x


@pytest.mark.parametrize("scale,m,method,hit", [
    ("guete", 3, "gleich", 47.0), ("guete", 3, "wahr", 85.0), ("guete", 3, "direkt", 86.5), ("guete", 3, "swing", 82.5), ("guete", 3, "bwm", 80.0),
    ("guete", 3, "rangsumme", 65.0), ("guete", 3, "rangrezipr", 64.0), ("guete", 3, "roc", 73.5),
    ("guete", 5, "gleich", 46.0), ("guete", 5, "wahr", 67.0), ("guete", 5, "direkt", 66.5), ("guete", 5, "swing", 66.0), ("guete", 5, "bwm", 66.5),
    ("guete", 5, "rangsumme", 58.5), ("guete", 5, "rangrezipr", 57.5), ("guete", 5, "roc", 58.5),
    ("roh", 3, "gleich", 51.5), ("roh", 3, "wahr", 94.0), ("roh", 3, "direkt", 87.5), ("roh", 3, "swing", 92.5), ("roh", 3, "bwm", 85.0),
    ("roh", 3, "rangsumme", 68.0), ("roh", 3, "rangrezipr", 67.5), ("roh", 3, "roc", 74.5),
    ("roh", 5, "gleich", 56.0), ("roh", 5, "wahr", 96.5), ("roh", 5, "direkt", 90.5), ("roh", 5, "swing", 95.0), ("roh", 5, "bwm", 93.5),
    ("roh", 5, "rangsumme", 73.5), ("roh", 5, "rangrezipr", 80.0), ("roh", 5, "roc", 81.5),
])
def test_methods_first_place(methods, scale, m, method, hit):
    assert pct(cell(methods, scale=scale, m=m, method=method)["hit"]) == pytest.approx(hit, abs=0.5)


def test_rank_formulas_trail_the_ratio_methods_in_every_cell(methods):
    for scale in C.SCALES:
        for m in C.EXP_M:
            ratio_best = max(cell(methods, scale=scale, m=m, method=k)["hit"] for k in ("direkt", "swing", "bwm"))
            rank_best = max(cell(methods, scale=scale, m=m, method=k)["hit"] for k in C.RANK_METHODS)
            assert 0.07 <= ratio_best - rank_best <= 0.19, (scale, m)


def test_everything_beats_equal_weights_by_a_wide_margin(methods):
    for scale in C.SCALES:
        for m in C.EXP_M:
            eq = cell(methods, scale=scale, m=m, method="gleich")["hit"]
            for k in ("direkt", "swing", "bwm", "wahr"):
                assert cell(methods, scale=scale, m=m, method=k)["hit"] > eq + 0.15


def test_judgment_counts_in_the_table(methods):
    row = {r["method"]: r["judgments"] for r in methods if r["m"] == 5 and r["scale"] == "roh"}
    assert row == {"gleich": 0, "wahr": 0, "direkt": 5, "swing": 5, "rangsumme": 7, "rangrezipr": 7, "roc": 7, "bwm": 7}


def test_utility_loss_of_equal_weights_is_an_order_of_magnitude_larger(methods):
    assert cell(methods, scale="roh", m=5, method="gleich")["loss"] == pytest.approx(0.033, abs=0.005)
    assert cell(methods, scale="roh", m=5, method="swing")["loss"] < 0.003


@pytest.mark.parametrize("scale,delta,method,hit", [
    ("roh", 1.0, "wahr", 96.5), ("roh", 1.0, "direkt", 90.5), ("roh", 1.0, "swing", 95.0),
    ("roh", 0.5, "wahr", 86.5), ("roh", 0.5, "direkt", 82.0), ("roh", 0.5, "swing", 93.0),
    ("roh", 0.2, "wahr", 76.0), ("roh", 0.2, "direkt", 73.0), ("roh", 0.2, "swing", 90.5),
    ("roh", 0.05, "wahr", 67.0), ("roh", 0.05, "direkt", 64.5), ("roh", 0.05, "swing", 91.5), ("roh", 0.05, "roc", 60.5), ("roh", 0.05, "gleich", 47.5),
    ("guete", 1.0, "wahr", 67.0), ("guete", 0.05, "wahr", 50.0), ("guete", 0.05, "direkt", 48.5), ("guete", 0.05, "swing", 69.0),
])
def test_one_narrow_criterion(narrow, scale, delta, method, hit):
    assert pct(cell(narrow, scale=scale, delta=delta, method=method)["hit"]) == pytest.approx(hit, abs=0.5)


def test_true_and_direct_weights_fall_with_the_range_and_swing_does_not(narrow):
    for scale in C.SCALES:
        wahr = [cell(narrow, scale=scale, delta=d, method="wahr")["hit"] for d in C.DELTA_CHOICES]
        swing = [cell(narrow, scale=scale, delta=d, method="swing")["hit"] for d in C.DELTA_CHOICES]
        assert wahr[0] - wahr[-1] > 0.15
        assert min(swing) >= swing[0] - 0.06                       # Swing hält sein Niveau (Güte-Welt: steigt sogar)
    assert cell(narrow, scale="roh", delta=0.05, method="swing")["hit"] > cell(narrow, scale="roh", delta=0.05, method="wahr")["hit"] + 0.2


@pytest.mark.parametrize("alpha,method,tv", [
    (0.5, "gleich", 0.434), (0.5, "rangsumme", 0.266), (0.5, "rangrezipr", 0.216), (0.5, "roc", 0.184),
    (1.0, "gleich", 0.329), (1.0, "rangsumme", 0.166), (1.0, "rangrezipr", 0.149), (1.0, "roc", 0.128),
    (5.0, "gleich", 0.157), (5.0, "rangsumme", 0.088), (5.0, "rangrezipr", 0.138), (5.0, "roc", 0.172),
])
def test_rank_formulas_against_the_weight_distribution(rank, alpha, method, tv):
    assert cell(rank, alpha=alpha, method=method)["tv"] == pytest.approx(tv, abs=0.01)


def test_roc_is_best_for_uniform_weights_and_not_for_almost_equal_ones(rank):
    for a in (0.5, 1.0):
        assert cell(rank, alpha=a, method="roc")["tv"] < min(cell(rank, alpha=a, method=k)["tv"] for k in ("rangsumme", "rangrezipr", "gleich"))
    assert cell(rank, alpha=5.0, method="roc")["tv"] > cell(rank, alpha=5.0, method="gleich")["tv"]
    assert cell(rank, alpha=5.0, method="roc")["tv"] > cell(rank, alpha=5.0, method="rangsumme")["tv"]


def test_roc_formula_matches_the_simulated_expectation():
    for r in ev.roc_expectation_check():
        assert r["max_deviation"] < 0.002


@pytest.mark.parametrize("sigma,method,hit", [
    (0.0, "direkt", 96.5), (0.0, "swing", 98.5), (0.0, "roc", 82.0), (0.0, "bwm", 93.5),
    (0.4, "direkt", 81.0), (0.4, "swing", 86.5), (0.4, "roc", 76.5), (0.4, "bwm", 84.0),
    (0.8, "direkt", 69.0), (0.8, "swing", 77.0), (0.8, "roc", 71.0), (0.8, "bwm", 79.0),
])
def test_noise(noise, sigma, method, hit):
    assert pct(cell(noise, sigma=sigma, method=method)["hit"]) == pytest.approx(hit, abs=0.5)


def test_every_method_degrades_with_noise(noise):
    for k in C.EXP_NOISE_METHODS:
        hits = [cell(noise, sigma=s, method=k)["hit"] for s in C.SIGMA_CHOICES]
        assert hits[0] > hits[-1] + 0.1


@pytest.mark.parametrize("alpha,m,share,anyc", [(1.0, 3, 22.2, 38.9), (1.0, 5, 24.8, 66.1), (1.0, 8, 27.8, 86.9), (0.5, 3, 43.0, 67.3), (0.5, 8, 48.8, 98.5)])
def test_best_worst_clipping(clip, alpha, m, share, anyc):
    r = cell(clip, alpha=alpha, m=m)
    assert pct(r["ratio_share"]) == pytest.approx(share, abs=1.5)
    assert pct(r["any_clipped"]) == pytest.approx(anyc, abs=2.0)


def test_verdicts_use_the_documented_numbers(methods, narrow, rank, noise, clip):
    assert "8 bis 18 Punkte" in ev.verdict_methods(methods)[1]
    assert "in 67 %" in ev.verdict_narrow(narrow)[1] and "in 92 %" in ev.verdict_narrow(narrow)[1]
    assert "0,17" in ev.verdict_rank(rank)[1] and "schlechter als gleiche Gewichte" in ev.verdict_rank(rank)[1]
    assert "25 %" in ev.verdict_noise(noise, clip)[1]
