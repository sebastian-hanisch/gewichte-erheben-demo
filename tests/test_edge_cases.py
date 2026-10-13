"""Grenzfälle, die der Mutationstest (DEMO-PLAYBOOK §12) gefunden hat: genau auf der Schwelle, Zahl der Ziehungen, Ausgabe der Verdikte.

Alle Eingaben sind klein und von Hand gerechnet. Wo eine Funktion Zufall zieht, ersetzen die Tests `draw_weights` durch eine feste Folge, damit die Zahl der
Ziehungen und ihr Mittel exakt prüfbar sind (ein Mittel über 2 Ziehungen unterscheidet sich von dem über 1).
"""

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

import gwe_constants as C
import gwe_elicit as el
import gwe_evaluation as ev
from entscheidungsmatrix_generator import SplitMix64


# ----------------------------------------------------------------------------- gwe_elicit

def test_points_with_a_small_positive_sum_are_not_mistaken_for_all_zero():
    # Nur eine Summe <= 0 heißt "alle Punkte 0". Summe 1 oder 0,75 sind echte Punkte und ergeben Gewichte, keine Gleichverteilung.
    assert el.weights_from_points(np.array([1.0, 0.0, 0.0])).tolist() == [1.0, 0.0, 0.0]
    assert el.weights_from_points(np.array([0.5, 0.25])).tolist() == pytest.approx([2 / 3, 1 / 3])


def test_roc_rejects_a_rank_above_m_instead_of_returning_a_weight():
    # m = 2: gültig sind die Ränge 1 und 2. Rang 3 gibt es nicht; die Tabelle hat genau m Einträge, der Zugriff schlägt fehl (kein stilles Gewicht 0).
    # Eine Zeile mehr in der Tabelle (range(1, m + 2)) bliebe bei gültigen Rängen unsichtbar und ließe Rang 3 still das Gewicht 0 bekommen.
    with pytest.raises((IndexError, ValueError)):
        el.weights_roc(np.array([1, 3]))


@pytest.mark.parametrize("w, a_B, a_W, clipped", [
    # Sigma 0: Rohwerte = Verhältnisse. Geschnitten wird nach dem Runden (numpy rundet die Hälfte zur geraden Zahl: 9,5 -> 10, 8,5 -> 8).
    ([19.0, 2.0, 1.0], [1.0, 9.0, 9.0], [9.0, 2.0, 1.0], 3),     # B/2 = 9,5 -> 10 -> 9 (geschnitten!), B/1 = 19, 19 zum Schlechtesten: drei
    ([20.0, 9.5, 1.0], [1.0, 2.0, 9.0], [9.0, 9.0, 1.0], 3),     # 9,5/1 = 9,5 -> 10 -> 9 (geschnitten), dazu 20 und 20: drei
    ([18.0, 2.0, 1.0], [1.0, 9.0, 9.0], [9.0, 2.0, 1.0], 2),     # 18/2 = 9: Rohwert 9 liegt auf der Skala, nichts geschnitten; nur die beiden 18
    ([20.0, 9.0, 1.0], [1.0, 2.0, 9.0], [9.0, 9.0, 1.0], 2),     # 9/1 = 9 auf der Skala; nur die beiden 20
    ([18.8, 2.0, 1.0], [1.0, 9.0, 9.0], [9.0, 2.0, 1.0], 2),     # 18,8/2 = 9,4 -> 9: knapp unter der Rundungsgrenze, nicht geschnitten
])
def test_best_worst_clipped_counts_exactly_the_judgments_the_scale_cut(w, a_B, a_W, clipped):
    j = el.judgments_bwm(np.array(w), 0.0, SplitMix64(1))
    assert (j["B"], j["W"]) == (0, 2)
    assert j["a_B"].tolist() == a_B and j["a_W"].tolist() == a_W
    assert j["clipped"] == clipped


# ----------------------------------------------------------------------------- feste Ziehungen für die Experimente

W1 = np.array([0.6, 0.3, 0.1])        # Ränge 1, 2, 3 (Mini-Instanz aus conftest)
W2 = np.array([0.2, 0.5, 0.3])        # Ränge 3, 1, 2


@pytest.fixture
def fixed_draws(monkeypatch):
    """`draw_weights` liefert W1, W2, W1, W2, ...; `calls` zählt die Ziehungen."""
    seq = itertools.cycle([W1, W2])
    state = SimpleNamespace(calls=0)

    def fake(m, alpha, rng):
        state.calls += 1
        return next(seq).copy()

    monkeypatch.setattr(ev, "draw_weights", fake)
    return state


def test_rank_formula_experiment_averages_over_exactly_the_requested_draws(fixed_draws):
    rows = ev.experiment_rank_formulas(alphas=(1.0, 5.0), m=3, draws=2)
    assert fixed_draws.calls == 2 * 2                                   # zwei Alphas mal zwei Ziehungen
    tv = {(r["alpha"], r["method"]): r["tv"] for r in rows}
    # W1: TV gleich 4/15, ROC 1/45, Rangsumme 1/10, Rangreziprok 9/110 (conftest). W2 (Ränge 3, 1, 2): gleich 1/6, ROC 1/9, Rangsumme 1/30, Rangreziprok 1/22.
    expect = {"gleich": (4 / 15 + 1 / 6) / 2, "roc": (1 / 45 + 1 / 9) / 2, "rangsumme": (1 / 10 + 1 / 30) / 2, "rangrezipr": (9 / 110 + 1 / 22) / 2}
    for a in (1.0, 5.0):
        for k, v in expect.items():
            assert tv[(a, k)] == pytest.approx(v), (a, k)


def test_roc_expectation_check_averages_over_exactly_the_requested_draws(fixed_draws):
    (row,) = ev.roc_expectation_check(ms=(3,), draws=2)
    assert fixed_draws.calls == 2
    # absteigend geordnet: (0,6; 0,3; 0,1) und (0,5; 0,3; 0,2), Mittel (0,55; 0,30; 0,15); ROC-Formel (11/18; 5/18; 2/18)
    assert row["first_sample"] == pytest.approx(0.55)
    assert row["first_formula"] == pytest.approx(11 / 18)
    assert row["max_deviation"] == pytest.approx(11 / 18 - 0.55)        # 0,0611 (die Abweichungen der anderen: 0,0222 und 0,0389)


def test_bwm_clip_stats_threshold_is_strictly_above_9_and_draws_are_counted(monkeypatch):
    # Zwei feste Ziehungen (nicht normiert, das ist der Funktion gleich): (9; 2; 1) und (20; 4; 1), B = Index 0, W = Index 2.
    # Ziehung 1: w_B / w_j = (4,5; 9,0), w_j / w_W = (9; 2): 9 ist NICHT über 9, nichts abgeschnitten -> Anteil 0, kein Abschnitt.
    # Ziehung 2: w_B / w_j = (5; 20) -> Anteil 1/2; w_j / w_W = (20; 4) -> mindestens ein Abschnitt.
    seq = itertools.cycle([np.array([9.0, 2.0, 1.0]), np.array([20.0, 4.0, 1.0])])
    calls = []

    def fake(m, alpha, rng):
        calls.append(m)
        return next(seq)

    monkeypatch.setattr(ev, "draw_weights", fake)
    (row,) = ev.bwm_clip_stats(alphas=(1.0,), ms=(3,), draws=2)
    assert len(calls) == 2
    assert row["alpha"] == 1.0 and row["m"] == 3
    assert row["ratio_share"] == pytest.approx((0.0 + 0.5) / 2)
    assert row["any_clipped"] == pytest.approx((0.0 + 1.0) / 2)


def test_narrow_standard_error_is_the_binomial_one(monkeypatch):
    # Vier feste Matrizen, Treffer je Methode vorgegeben: swing 3 von 4 (p = 0,75), direkt 1 von 4 (p = 0,25), wahr 4 von 4, gleich 0 von 4.
    # se = sqrt(p (1 - p) / n): sqrt(0,75 * 0,25 / 4) = sqrt(3) / 8 = 0,2165 für beide mittleren; 0 an den Rändern.
    pattern = {"swing": [True, True, True, False], "direkt": [True, False, False, False], "wahr": [True] * 4, "gleich": [False] * 4}
    seeds = (10, 11, 12, 13)
    monkeypatch.setattr(ev, "make_case", lambda *a, **k: None)
    monkeypatch.setattr(ev, "evaluate_methods",
                        lambda case, sigma, seed, methods: {k: SimpleNamespace(hit=pattern[k][seeds.index(seed)]) for k in methods})
    rows = ev.experiment_narrow(deltas=(1.0,), methods=tuple(pattern), seeds=seeds)
    assert len(rows) == 2 * len(pattern)
    got = {r["method"]: (r["hit"], r["se"]) for r in rows}
    assert got["swing"] == (0.75, pytest.approx(math.sqrt(3) / 8))
    assert got["direkt"] == (0.25, pytest.approx(math.sqrt(3) / 8))
    assert got["wahr"] == (1.0, 0.0) and got["gleich"] == (0.0, 0.0)


# ----------------------------------------------------------------------------- Verdikte mit synthetischen Zeilen

def _method_rows(table):
    """{(scale, m): (direkt, swing, bwm, rangsumme, rangrezipr, roc)} -> Zeilen wie experiment_methods; dazu die Referenz "wahr" mit Trefferquote 1."""
    rows = []
    for (scale, m), hits in table.items():
        for k, h in zip(("direkt", "swing", "bwm", "rangsumme", "rangrezipr", "roc"), hits):
            rows.append({"scale": scale, "m": m, "method": k, "hit": h})
        rows.append({"scale": scale, "m": m, "method": "wahr", "hit": 1.0})
    return rows


def test_verdict_methods_reports_smallest_and_largest_gap_in_points():
    rows = _method_rows({
        ("guete", 3): (0.90, 0.80, 0.70, 0.30, 0.20, 0.25),     # beste Verhältnismethode 0,90, beste Rangformel 0,30: 60 Punkte
        ("guete", 5): (0.50, 0.95, 0.60, 0.25, 0.30, 0.20),     # 0,95 - 0,30: 65
        ("roh", 3): (0.40, 0.50, 0.90, 0.10, 0.15, 0.20),       # 0,90 - 0,20: 70 (größte Lücke)
        ("roh", 5): (0.80, 0.70, 0.60, 0.25, 0.10, 0.20),       # 0,80 - 0,25: 55 (kleinste Lücke)
    })
    kind, text = ev.verdict_methods(rows)
    assert kind == "info"
    assert text.startswith("Die Rangformeln liegen 55 bis 70 Punkte hinter der besten Verhältnismethode (direkt, Swing, Best-Worst);")


def _rank_rows(roc_hi, eq_hi, roc_mid=0.13):
    rows = []
    for a, roc, eq in ((0.5, 0.18, 0.43), (1.0, roc_mid, 0.33), (5.0, roc_hi, eq_hi)):
        rows += [{"alpha": a, "method": "roc", "tv": roc}, {"alpha": a, "method": "gleich", "tv": eq}]
    return rows


@pytest.mark.parametrize("roc_hi, eq_hi, kind, phrase", [
    (0.17, 0.16, "warning", "ROC mit 0,17 schlechter als gleiche Gewichte (0,16)"),
    (0.15, 0.15, "info", "ROC mit 0,15 nicht schlechter als gleiche Gewichte (0,15)"),     # genau gleich: nicht schlechter
    (0.10, 0.16, "info", "ROC mit 0,10 nicht schlechter als gleiche Gewichte (0,16)"),
])
def test_verdict_rank_warns_only_when_roc_is_strictly_worse_than_equal_weights(roc_hi, eq_hi, kind, phrase):
    k, text = ev.verdict_rank(_rank_rows(roc_hi, eq_hi))
    assert k == kind
    assert phrase in text
    assert "Abstand von 0,13 die beste Rangformel" in text and "(α = 5)" in text


def test_verdict_noise_uses_the_largest_sigma_and_the_clip_row_for_m_5():
    rows = []
    for sg, swing, roc in ((0.0, 0.98, 0.82), (0.4, 0.86, 0.76), (0.8, 0.77, 0.71)):
        rows += [{"sigma": sg, "method": "swing", "hit": swing}, {"sigma": sg, "method": "roc", "hit": roc}]
    clip = [{"alpha": 1.0, "m": 3, "ratio_share": 0.222, "any_clipped": 0.39}, {"alpha": 0.5, "m": 5, "ratio_share": 0.49, "any_clipped": 0.9},
            {"alpha": 1.0, "m": 5, "ratio_share": 0.248, "any_clipped": 0.66}]
    kind, text = ev.verdict_noise(rows, clip)
    assert kind == "info"
    assert text.startswith("Bei σ = 0,8 trifft Swing Platz 1 in 77 %, ROC in 71 %.")
    assert "25 % der Verhältnisse zum Besten größer als 9" in text
