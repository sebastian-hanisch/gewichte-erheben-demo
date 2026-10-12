"""Gewichtete Summe mit Min-Max, Nutzen des Entscheiders auf beiden Wertskalen, Verschmälerung eines Kriteriums. Nur numpy; eigene Kopie der kleinen Helfer
aus gewichtete-summe-demo (kein Import über Repos)."""

from __future__ import annotations

import numpy as np

from entscheidungsmatrix_generator import CRITERIA, to_raw


def minmax(S: np.ndarray) -> np.ndarray:
    """Min-Max-Normierung je Kriterium über die Stichprobe; konstante Spalte -> lauter Nullen."""
    S = np.asarray(S, dtype=float)
    rng = S.max(axis=0) - S.min(axis=0)
    return (S - S.min(axis=0)) / np.where(rng == 0, 1.0, rng)


def weighted_score(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(N, dtype=float) @ np.asarray(w, dtype=float)


def top1(score: np.ndarray) -> int:
    """Index des höchsten Scores; bei Gleichstand die kleinste Nummer."""
    return int(np.argmax(score))


def nominal_bounds(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nenn-Spannweite je Kriterium im vorzeichenbehafteten Raum: ↑ [lo, hi], ↓ [-hi, -lo]."""
    lo = np.array([CRITERIA[k][3] if CRITERIA[k][1] > 0 else -CRITERIA[k][4] for k in range(m)], dtype=float)
    hi = np.array([CRITERIA[k][4] if CRITERIA[k][1] > 0 else -CRITERIA[k][3] for k in range(m)], dtype=float)
    return lo, hi


def narrow_criterion(goodness: np.ndarray, delta: float, column: int = 0) -> np.ndarray:
    """Staucht die Güte eines Kriteriums um 0,5 auf den Anteil `delta` der Streuung: g' = 0,5 + delta (g - 0,5). delta = 1 ändert nichts."""
    g = np.array(goodness, dtype=float, copy=True)
    g[:, column] = 0.5 + delta * (g[:, column] - 0.5)
    return g


def value_functions(scale: str, goodness: np.ndarray, S: np.ndarray, m: int) -> np.ndarray:
    """Wert je Alternative und Kriterium auf der Wertskala des Entscheiders: Güte (Quantil) oder Rohwert bezogen auf die feste Nenn-Spannweite."""
    if scale == "guete":
        return np.asarray(goodness, dtype=float)
    if scale == "roh":
        lo, hi = nominal_bounds(m)
        return (np.asarray(S, dtype=float) - lo) / (hi - lo)
    raise ValueError(f"unbekannte Wertskala: {scale}")


def build_case(goodness: np.ndarray, directions: np.ndarray, m: int, delta: float, scale: str):
    """Aus Güte und Richtungen: (Güte nach Verschmälerung, Rohwerte, vorzeichenbehaftete Rohwerte, Wert je Kriterium)."""
    g = narrow_criterion(goodness, delta)
    raw = to_raw(g, m)
    S = raw * directions
    return g, raw, S, value_functions(scale, g, S, m)


def utility_loss(utility: np.ndarray, chosen: int) -> float:
    return float(np.max(utility) - utility[chosen])


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)
