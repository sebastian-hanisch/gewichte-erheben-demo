"""Gemeinsamer Entscheidungsmatrix-Generator der Linie Mehrkriterien-Ranking (WORTGLEICHE Kopie in allen Repos der Linie, kein Modul-Präfix).

Version 1 (2026-10-12). Nur numpy und math, kein Streamlit. Änderungen an diesem Modul betreffen alle Repos der Linie: Schnittstelle, Namen
und Reihenfolge des Zufallsverbrauchs nur ändern, wenn alle Kopien nachgezogen werden (tools/portfolio_audit.py, SHARED_GROUPS).

Modell: n Alternativen, m Kriterien. Die Güte der Alternativen je Kriterium ist gleichverteilt auf [0, 1] (höher = besser) und über eine
Gauß-Copula mit Gleichkorrelation rho zwischen den Kriterien verbunden. Daraus entstehen Rohwerte mit Richtung und Einheit (↑ mehr ist besser,
↓ weniger ist besser) über je eine streng monotone Abbildung; bei `levels` > 0 sind es ganzzahlige Bewertungsstufen 1 bis levels (alle ↑).
Ein simulierter Entscheider kennt verborgene Gewichte (gleichverteilt über dem Simplex) und eine Nutzenform; sein Nutzen ist das Orakel.

Zufall nur über einen übergebenen `SplitMix64`; Verbrauch je Instanz in fester Reihenfolge: Normalverteilte der Copula (zwei Blöcke
n*m Gleichverteilte, Box-Muller), danach m Werte für die Gewichte des Entscheiders. Plattformhinweis: Die Rohwerte laufen über `math.erf`,
`log` und `cos` der C-Bibliothek und können sich zwischen Plattformen in der letzten Stelle unterscheiden; Tests dürfen Float-Werte
daher nur mit Toleranz prüfen, Ordnungsaussagen sind davon nicht betroffen.
"""

import math
from dataclasses import dataclass

import numpy as np

GENERATOR_VERSION = 1
FORMS = ("linear", "konkav", "cobb", "zielwert")
FORM_LABELS = {
    "linear": "Linear: gewichtete Summe der Güten",
    "konkav": "Konkav: Wurzel der Güten, gewichtet",
    "cobb": "Cobb-Douglas: gewichtetes Mittel der Logarithmen",
    "zielwert": "Zielwert: Kriterium 1 hat ein Optimum in der Mitte",
}

# Name, Richtung (+1 mehr ist besser, -1 weniger ist besser), Einheit, Untergrenze, Obergrenze, Nachkommastellen.
# Skala: ↑-Kriterien (Punkte) sind linear, ↓-Kriterien (Euro, Minuten) logarithmisch; das ist die einzige Kombination, die die Linie braucht.
CRITERIA = (
    ("Kriterium 1", +1, "Punkte", 0.0, 100.0, 2),
    ("Kriterium 2", -1, "Euro", 800.0, 5000.0, 0),
    ("Kriterium 3", +1, "Punkte", 0.0, 100.0, 2),
    ("Kriterium 4", -1, "Minuten", 5.0, 600.0, 1),
    ("Kriterium 5", +1, "Punkte", 0.0, 100.0, 2),
    ("Kriterium 6", -1, "Euro", 800.0, 5000.0, 0),
    ("Kriterium 7", +1, "Punkte", 0.0, 100.0, 2),
    ("Kriterium 8", -1, "Minuten", 5.0, 600.0, 1),
)
M_MAX = len(CRITERIA)

_GAMMA = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_ERF = np.frompyfunc(math.erf, 1, 1)


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        """Die nächsten `size` 64-Bit-Werte; der Strom rückt um `size` Schritte weiter."""
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilte Werte in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormalverteilte Werte (Box-Muller, nur der Kosinus-Zweig); verbraucht 2 * size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)

    def permutation(self, n: int) -> np.ndarray:
        return np.argsort(self.uniform(n), kind="stable")


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standardnormal-Verteilungsfunktion über math.erf."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + _ERF(z / math.sqrt(2.0)).astype(float))


def rho_lower(m: int) -> float:
    """Untergrenze der Gleichkorrelation einer m x m Korrelationsmatrix: -1/(m-1); die Grenze selbst ist singulär (m = 1: -1)."""
    return -1.0 if m <= 1 else -1.0 / (m - 1)


def copula_uniforms(n: int, m: int, rho: float, rng: SplitMix64) -> np.ndarray:
    """n x m Güten, Randverteilungen gleichverteilt auf (0, 1), Gleichkorrelation rho der zugrunde liegenden Normalverteilten."""
    if not (rho_lower(m) < rho < 1.0):
        raise ValueError(f"rho = {rho} liegt nicht in ({rho_lower(m):.3f}, 1) für m = {m}")
    z = rng.normal(n * m).reshape(n, m)
    if rho != 0.0:
        corr = (1.0 - rho) * np.eye(m) + rho * np.ones((m, m))
        z = z @ np.linalg.cholesky(corr).T
    return np.clip(normal_cdf(z), 1e-12, 1.0 - 1e-12)


def quantize(goodness: np.ndarray, levels: int) -> np.ndarray:
    """Bewertungsstufen 1 bis levels aus Güten in [0, 1]: gleich breite Klassen."""
    return np.minimum(np.floor(goodness * levels), levels - 1) + 1.0


def to_raw(goodness: np.ndarray, m: int) -> np.ndarray:
    """Rohwerte je Kriterium aus der Güte (streng monoton in der Güte; ↓-Kriterien fallen mit steigender Güte), auf die Anzeigegenauigkeit gerundet.

    ↑: linear von Untergrenze (Güte 0) bis Obergrenze (Güte 1). ↓: logarithmisch von der Obergrenze (Güte 0, schlecht) bis zur Untergrenze (Güte 1, gut).
    """
    raw = np.empty_like(goodness, dtype=float)
    for k in range(m):
        _, direction, _, lo, hi, decimals = CRITERIA[k]
        g = goodness[:, k]
        v = lo + (hi - lo) * g if direction > 0 else hi * (lo / hi) ** g
        raw[:, k] = np.round(v, decimals)
    return raw


def directions(m: int, levels: int = 0) -> np.ndarray:
    """Richtung je Kriterium (+1 mehr ist besser, -1 weniger ist besser); auf Bewertungsstufen sind alle +1."""
    if levels:
        return np.ones(m)
    return np.array([CRITERIA[k][1] for k in range(m)], dtype=float)


def decision_maker_weights(m: int, rng: SplitMix64) -> np.ndarray:
    """Verborgene Gewichte des Entscheiders, gleichverteilt über dem Simplex (normierte Exponentialverteilte)."""
    e = -np.log1p(-rng.uniform(m))
    return e / e.sum()


def utility(goodness: np.ndarray, weights: np.ndarray, form: str = "linear") -> np.ndarray:
    """Nutzen je Alternative aus Güten in [0, 1]. linear, konkav und cobb sind streng monoton in jeder Güte; zielwert ersetzt die Güte
    des Kriteriums 1 durch 1 - |2 (g - 0,5)| (Optimum in der Mitte, nicht monoton)."""
    g = np.asarray(goodness, dtype=float)
    if form == "linear":
        return g @ weights
    if form == "konkav":
        return np.sqrt(g) @ weights
    if form == "cobb":
        return np.log(np.clip(g, 1e-12, None)) @ weights
    if form == "zielwert":
        h = g.copy()
        h[:, 0] = 1.0 - np.abs(2.0 * (g[:, 0] - 0.5))
        return h @ weights
    raise ValueError(f"unbekannte Nutzenform: {form}")


@dataclass(frozen=True)
class Matrix:
    """Eine Entscheidungsmatrix mit simuliertem Entscheider.

    goodness: n x m Güte in [0, 1] (nach Quantisierung: Klassenmitte), Grundlage des Nutzens.
    raw:      n x m Rohwerte (Anzeige), mit Richtung und Einheit; bei levels > 0 Bewertungsstufen 1 bis levels.
    signed:   n x m Rohwerte mit Vorzeichen der Richtung (höher = besser); die Dominanz wird hier gerechnet.
    """

    n: int
    m: int
    rho: float
    levels: int
    goodness: np.ndarray
    raw: np.ndarray
    signed: np.ndarray
    directions: np.ndarray
    names: tuple
    units: tuple
    weights: np.ndarray
    form: str
    utility: np.ndarray


def make_matrix(n: int, m: int, rho: float, rng: SplitMix64, levels: int = 0, form: str = "linear") -> Matrix:
    """Erzeugt eine Instanz. `levels` = 0: stetige Rohwerte; sonst ganzzahlige Bewertungsstufen 1 bis levels."""
    if not (1 <= m <= M_MAX):
        raise ValueError(f"m muss zwischen 1 und {M_MAX} liegen")
    if form not in FORMS:
        raise ValueError(f"unbekannte Nutzenform: {form}")
    u = copula_uniforms(n, m, rho, rng)
    weights = decision_maker_weights(m, rng)
    if levels:
        raw = quantize(u, levels)
        goodness = (raw - 0.5) / levels
        names = tuple(CRITERIA[k][0] for k in range(m))
        units = ("Stufe",) * m
    else:
        goodness = u
        raw = to_raw(u, m)
        names = tuple(CRITERIA[k][0] for k in range(m))
        units = tuple(CRITERIA[k][2] for k in range(m))
    d = directions(m, levels)
    return Matrix(n, m, float(rho), int(levels), goodness, raw, raw * d, d, names, units, weights, form, utility(goodness, weights, form))
