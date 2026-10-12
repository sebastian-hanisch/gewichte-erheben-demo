"""Auswertung: Fall der App und Form der Experimente (kleine Seed-Mengen, damit die Suite schnell bleibt)."""

import numpy as np
import pytest

import gwe_constants as C
import gwe_evaluation as ev
import gwe_score as sc
from entscheidungsmatrix_generator import SplitMix64

FEW = C.SWEEP_SEEDS[:12]


def test_analysis_is_consistent():
    a = ev.analyse(ev.Settings(n=12, m=4, method="bwm", sigma=0.4, delta=0.5, scale="guete", alpha=5.0, seed=7))
    assert set(a.results) == set(ev.ALL_METHODS)
    assert a.chosen is a.results["bwm"]
    assert a.best == sc.top1(a.case.utility)
    for k, r in a.results.items():
        assert r.weights.sum() == pytest.approx(1.0) and (r.weights >= 0).all(), k
        assert r.hit == (r.top1 == a.best)
        assert r.loss == pytest.approx(a.case.utility.max() - a.case.utility[r.top1])
    assert a.results["wahr"].tv == 0.0 and a.results["wahr"].hit
    assert a.target.sum() == pytest.approx(1.0)


def test_the_choice_of_a_method_never_changes_the_judgments_of_the_others():
    a = ev.analyse(ev.Settings(method="direkt", seed=9))
    b = ev.analyse(ev.Settings(method="roc", seed=9))
    for k in ev.ALL_METHODS:
        assert a.results[k].weights.tolist() == b.results[k].weights.tolist()


def test_same_settings_give_the_same_analysis():
    a, b = ev.analyse(ev.Settings(seed=5)), ev.analyse(ev.Settings(seed=5))
    assert a.case.S.tolist() == b.case.S.tolist() and a.results["swing"].weights.tolist() == b.results["swing"].weights.tolist()


def test_alpha_one_uses_the_generators_weights_and_other_alphas_draw_their_own():
    base = ev.make_case(10, 4, 3, alpha=1.0)
    assert base.w.tolist() == base.matrix.weights.tolist()
    other = ev.make_case(10, 4, 3, alpha=5.0)
    assert other.w.tolist() != base.w.tolist() and other.w.sum() == pytest.approx(1.0)
    assert other.S.tolist() == base.S.tolist()                          # die Matrix ist von alpha unabhängig
    with pytest.raises(ValueError):
        ev.draw_weights(3, 2.0, SplitMix64(1))


def test_value_range_follows_the_narrowing():
    wide = ev.make_case(40, 5, 4, delta=1.0, scale="roh")
    narrow = ev.make_case(40, 5, 4, delta=0.05, scale="roh")
    assert narrow.value_range[0] < 0.2 * wide.value_range[0]
    assert narrow.value_range[1:].tolist() == pytest.approx(wide.value_range[1:].tolist())


def test_methods_experiment_shape_and_reference_rows():
    rows = ev.experiment_methods(ms=(3,), seeds=FEW)
    assert len(rows) == 2 * 1 * len(ev.ALL_METHODS)
    ref = [r for r in rows if r["method"] == "wahr"]
    assert all(r["tv"] == 0.0 and r["judgments"] == 0 for r in ref)
    for r in rows:
        assert 0 <= r["hit"] <= 1 and -1 <= r["tau"] <= 1 and r["loss"] >= 0
    assert ev.verdict_methods(rows)[0] == "info"


def test_narrow_experiment_shape_and_swing_beats_direct_at_the_narrowest():
    rows = ev.experiment_narrow(deltas=(1.0, 0.05), seeds=C.SWEEP_SEEDS[:60])
    assert len(rows) == 2 * 2 * len(C.EXP_NARROW_METHODS)
    swing = [r["hit"] for r in rows if r["scale"] == "roh" and r["delta"] == 0.05 and r["method"] == "swing"][0]
    direct = [r["hit"] for r in rows if r["scale"] == "roh" and r["delta"] == 0.05 and r["method"] == "direkt"][0]
    assert swing > direct
    assert ev.verdict_narrow(rows)[0] == "warning"


def test_rank_formulas_experiment_and_the_alpha_ordering():
    rows = ev.experiment_rank_formulas(draws=600)
    assert len(rows) == 3 * 4
    tv = {(r["alpha"], r["method"]): r["tv"] for r in rows}
    assert tv[(1.0, "roc")] < tv[(1.0, "rangsumme")] < tv[(1.0, "gleich")]
    assert tv[(5.0, "roc")] > tv[(5.0, "rangsumme")]
    assert ev.verdict_rank(rows)[0] in ("warning", "info")


def test_roc_expectation_check_is_small():
    for r in ev.roc_expectation_check(ms=(3,), draws=3000):
        assert r["max_deviation"] < 0.01


def test_noise_experiment_degrades_with_sigma():
    rows = ev.experiment_noise(sigmas=(0.0, 0.8), seeds=C.SWEEP_SEEDS[:60])
    for k in C.EXP_NOISE_METHODS:
        lo = [r["hit"] for r in rows if r["sigma"] == 0.0 and r["method"] == k][0]
        hi = [r["hit"] for r in rows if r["sigma"] == 0.8 and r["method"] == k][0]
        assert hi < lo, k


def test_bwm_clip_statistics_grow_with_m_and_with_unequal_weights():
    rows = ev.bwm_clip_stats(ms=(3, 5, 8), draws=1500)
    get = {(r["alpha"], r["m"]): r for r in rows}
    assert get[(1.0, 8)]["any_clipped"] > get[(1.0, 3)]["any_clipped"]
    assert get[(0.5, 3)]["ratio_share"] > get[(1.0, 3)]["ratio_share"]
    noise_rows = ev.experiment_noise(sigmas=(0.2,), seeds=C.SWEEP_SEEDS[:20])
    assert "Skala" in ev.verdict_noise(noise_rows, rows)[1] or "abgeschnitten" in ev.verdict_noise(noise_rows, rows)[1]


def test_german_decimal_comma():
    assert ev.de(0.5, 1) == "0,5" and ev.de(12.0, 0) == "12"


def test_worst_case_at_the_slider_maximum_terminates():
    a = ev.analyse(ev.Settings(n=C.N_MAX, m=C.M_MAX, method="bwm", sigma=0.8, delta=0.05, scale="guete", alpha=0.5, seed=3))
    assert all(np.isfinite(r.tau) for r in a.results.values())
