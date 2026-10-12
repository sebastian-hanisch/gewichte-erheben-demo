"""Urteilsmodell und Erhebungsmethoden für Gewichte. Nur numpy (scipy nur im Best-Worst-LP), jede Regel eine eigene Funktion.

Das Urteilsmodell ist ANGENOMMEN, nicht am Menschen gemessen: Der Entscheider kennt seine Gewichte w; jede Methode fragt etwas anderes und bekommt ein Urteil, das mit
lognormalem Rauschen (Streuung sigma des Logarithmus eines Verhältnisses) und der Rundung der Methode versehen ist. Zufall nur über einen übergebenen `SplitMix64`.
"""

from __future__ import annotations

import math

import numpy as np

import gwe_constants as C


def weights_from_points(points: np.ndarray) -> np.ndarray:
    """Punkte (>= 0) auf Gewichte mit Summe 1; sind alle Punkte 0, sind die Gewichte gleich."""
    p = np.asarray(points, dtype=float)
    total = p.sum()
    return np.full(len(p), 1.0 / len(p)) if total <= 0 else p / total


def points_direct(w: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Direkte Wichtigkeit: Punkte 0 bis 100 ~ w_k / max(w), mit Rauschen und ganzzahlig gerundet. Kein Bezug zur Spannweite der Alternativen."""
    w = np.asarray(w, dtype=float)
    noise = np.exp(sigma * rng.normal(len(w)))
    return np.clip(np.round(C.POINTS_MAX * (w / w.max()) * noise), 0, C.POINTS_MAX)


def swing_target(w: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    """Schwenk-Gewichte ohne Rauschen: w_k mal Spannweite des Wertes (worst -> best) in der Stichprobe, auf Summe 1."""
    s = np.asarray(w, dtype=float) * np.asarray(value_range, dtype=float)
    return s / s.sum()


def points_swing(w: np.ndarray, value_range: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Swing: der größte Schwenk bekommt 100 Punkte (Definition der Methode), die übrigen relativ dazu, mit Rauschen und ganzzahlig gerundet."""
    s = np.asarray(w, dtype=float) * np.asarray(value_range, dtype=float)
    ratio = s / s.max()
    noisy = np.minimum(1.0, ratio * np.exp(sigma * rng.normal(len(s))))
    noisy[int(np.argmax(s))] = 1.0
    return np.round(C.POINTS_MAX * noisy)


def perceived_ranks(w: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Rang 1..m je Kriterium nach der wahrgenommenen Wichtigkeit w_k * exp(sigma * eps); Gleichstand: kleinere Nummer zuerst."""
    w = np.asarray(w, dtype=float)
    perceived = w * np.exp(sigma * rng.normal(len(w)))
    order = np.lexsort((np.arange(len(w)), -perceived))
    ranks = np.empty(len(w), dtype=int)
    ranks[order] = np.arange(1, len(w) + 1)
    return ranks


def weights_rank_sum(ranks: np.ndarray) -> np.ndarray:
    """Rangsumme: w_r proportional zu m + 1 - r."""
    r = np.asarray(ranks)
    x = (len(r) + 1 - r).astype(float)
    return x / x.sum()


def weights_rank_reciprocal(ranks: np.ndarray) -> np.ndarray:
    """Rangreziprok: w_r proportional zu 1 / r."""
    x = 1.0 / np.asarray(ranks, dtype=float)
    return x / x.sum()


def weights_roc(ranks: np.ndarray) -> np.ndarray:
    """Rangzentroid (ROC): w_r = (1/m) * Summe_{j=r..m} 1/j; der Erwartungswert des r-größten Gewichts bei Gleichverteilung über dem Simplex."""
    r = np.asarray(ranks)
    m = len(r)
    tail = np.array([sum(1.0 / j for j in range(i, m + 1)) for i in range(1, m + 1)]) / m
    return tail[r - 1]


def judgments_bwm(w: np.ndarray, sigma: float, rng) -> dict:
    """Best-Worst: der Entscheider kennt Bestes B und Schlechtestes W (nach w); Verhältnisse a_B[j] = w_B / w_j und a_W[j] = w_j / w_W, mit Rauschen,
    auf ganze Zahlen 1 bis 9 gerundet und abgeschnitten. `clipped` zählt die Urteile, deren Rohwert über 9 lag."""
    w = np.asarray(w, dtype=float)
    m = len(w)
    B, W = int(np.argmax(w)), int(np.argmin(w))
    raw_B = w[B] / w * np.exp(sigma * rng.normal(m))
    raw_W = w / w[W] * np.exp(sigma * rng.normal(m))
    a_B = np.clip(np.round(raw_B), 1, C.BWM_MAX)
    a_W = np.clip(np.round(raw_W), 1, C.BWM_MAX)
    a_B[B], a_W[W] = 1.0, 1.0
    clipped = int((np.delete(raw_B, B) > C.BWM_MAX + 0.5).sum() + (np.delete(raw_W, W) > C.BWM_MAX + 0.5).sum())
    return {"B": B, "W": W, "a_B": a_B, "a_W": a_W, "clipped": clipped}


def weights_bwm(B: int, W: int, a_B: np.ndarray, a_W: np.ndarray) -> np.ndarray:
    """Lineares Best-Worst: minimiere xi unter |w_B - a_Bj w_j| <= xi, |w_j - a_jW w_W| <= xi, Summe w = 1, w >= 0 (LP, scipy)."""
    from scipy.optimize import linprog

    a_B = np.asarray(a_B, dtype=float)
    a_W = np.asarray(a_W, dtype=float)
    m = len(a_B)
    rows = []
    for j in range(m):
        if j != B:
            e = np.zeros(m)
            e[B], e[j] = 1.0, -a_B[j]
            rows += [np.r_[e, -1.0], np.r_[-e, -1.0]]
        if j != W:
            e = np.zeros(m)
            e[j], e[W] = 1.0, -a_W[j]
            rows += [np.r_[e, -1.0], np.r_[-e, -1.0]]
    res = linprog(np.r_[np.zeros(m), 1.0], A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=[[1.0] * m + [0.0]], b_eq=[1.0],
                  bounds=[(0, None)] * (m + 1), method="highs")
    v = np.maximum(res.x[:m], 0.0)
    return v / v.sum()


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Halber L1-Abstand zweier Gewichtsvektoren (Totalvariation), zwischen 0 und 1."""
    return float(0.5 * np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())


def elicit(method: str, w: np.ndarray, value_range: np.ndarray, sigma: float, rng) -> tuple[np.ndarray, dict]:
    """Gewichte der Methode und ihre Urteile (für die Anzeige). `wahr` ist die Referenz: die Gewichte des Entscheiders selbst."""
    m = len(w)
    if method == "wahr":
        return np.asarray(w, dtype=float).copy(), {}
    if method == "gleich":
        return np.full(m, 1.0 / m), {}
    if method == "direkt":
        pts = points_direct(w, sigma, rng)
        return weights_from_points(pts), {"points": pts}
    if method == "swing":
        pts = points_swing(w, value_range, sigma, rng)
        return weights_from_points(pts), {"points": pts}
    if method in C.RANK_METHODS:
        ranks = perceived_ranks(w, sigma, rng)
        f = {"rangsumme": weights_rank_sum, "rangrezipr": weights_rank_reciprocal, "roc": weights_roc}[method]
        return f(ranks), {"ranks": ranks}
    if method == "bwm":
        j = judgments_bwm(w, sigma, rng)
        return weights_bwm(j["B"], j["W"], j["a_B"], j["a_W"]), j
    raise ValueError(f"unbekannte Erhebungsmethode: {method}")


def judgment_count(method: str, m: int) -> int:
    """Zahl der Einzelurteile: direkt und Swing eines je Kriterium, Best-Worst 2m - 3. Die Rangformeln brauchen EINE Rangordnung der Kriterien; als Maß
    dient die untere Schranke ceil(log2(m!)) der Vergleiche, die ein Sortieren von m Elementen mindestens braucht."""
    if method in C.RANK_METHODS:
        return math.ceil(math.log2(math.factorial(m)))
    return {"gleich": 0, "wahr": 0, "direkt": m, "swing": m, "bwm": 2 * m - 3}[method]
