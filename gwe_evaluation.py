"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte. Nur numpy (scipy über gwe_elicit), kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import gwe_constants as C
import gwe_elicit as el
import gwe_score as sc
from entscheidungsmatrix_generator import SplitMix64, make_matrix

ALL_METHODS = ("gleich", "wahr", "direkt", "swing", "rangsumme", "rangrezipr", "roc", "bwm")   # feste Reihenfolge des Zufallsverbrauchs
JUDGMENT_OFFSET = 20011       # Strom der Urteile: SplitMix64(seed + JUDGMENT_OFFSET), unabhängig vom Strom der Matrix
WEIGHT_OFFSET = 10007         # Strom der wahren Gewichte, wenn alpha != 1


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    method: str = C.DEFAULT_METHOD
    sigma: float = C.DEFAULT_SIGMA
    delta: float = C.DEFAULT_DELTA
    scale: str = C.DEFAULT_SCALE
    alpha: float = C.DEFAULT_ALPHA
    seed: int = C.DEFAULT_SEED


def draw_weights(m: int, alpha: float, rng: SplitMix64) -> np.ndarray:
    """Wahre Gewichte aus Dirichlet(alpha) für alpha in {0,5; 1; 5}: Gamma(0,5) = Z^2 / 2, Gamma(1) = Exponential, Gamma(5) = Summe von fünf Exponentialverteilten."""
    if alpha == 0.5:
        g = rng.normal(m) ** 2 / 2.0
    elif alpha == 1.0:
        g = -np.log1p(-rng.uniform(m))
    elif alpha == 5.0:
        g = (-np.log1p(-rng.uniform(5 * m))).reshape(5, m).sum(axis=0)
    else:
        raise ValueError(f"alpha muss 0,5, 1 oder 5 sein, nicht {alpha}")
    return g / g.sum()


@dataclass
class Case:
    """Eine Matrix mit simuliertem Entscheider: Rohwerte nach Verschmälerung, Wert je Kriterium auf seiner Wertskala, wahre Gewichte, Nutzen."""
    matrix: object
    goodness: np.ndarray
    raw: np.ndarray
    S: np.ndarray
    f: np.ndarray
    w: np.ndarray
    utility: np.ndarray

    @property
    def value_range(self) -> np.ndarray:
        """Spannweite des Wertes je Kriterium in der Stichprobe (worst -> best)."""
        return self.f.max(axis=0) - self.f.min(axis=0)


def make_case(n: int, m: int, seed: int, delta: float = 1.0, scale: str = "roh", alpha: float = 1.0) -> Case:
    x = make_matrix(n, m, 0.0, SplitMix64(seed))
    w = x.weights if alpha == 1.0 else draw_weights(m, alpha, SplitMix64(seed + WEIGHT_OFFSET))
    g, raw, S, f = sc.build_case(x.goodness, x.directions, m, delta, scale)
    return Case(x, g, raw, S, f, w, f @ w)


@dataclass
class Result:
    weights: np.ndarray
    judgments: dict
    top1: int
    hit: bool
    tau: float
    loss: float
    tv: float
    count: int


def evaluate_methods(case: Case, sigma: float, seed: int, methods=ALL_METHODS) -> dict:
    """Alle Methoden auf derselben Matrix: gleicher Urteilsstrom, feste Reihenfolge (die Wahl einer Methode ändert die Urteile der anderen nie)."""
    rng = SplitMix64(seed + JUDGMENT_OFFSET)
    N = sc.minmax(case.S)
    best = sc.top1(case.utility)
    out = {}
    for k in ALL_METHODS:
        w_hat, j = el.elicit(k, case.w, case.value_range, sigma, rng)
        if k not in methods:
            continue
        score = sc.weighted_score(N, w_hat)
        t = sc.top1(score)
        out[k] = Result(w_hat, j, t, t == best, sc.kendall_tau_b(score, case.utility), sc.utility_loss(case.utility, t),
                        el.tv_distance(w_hat, case.w), el.judgment_count(k, case.matrix.m))
    return out


@dataclass
class Analysis:
    settings: Settings
    case: Case
    results: dict
    best: int
    target: np.ndarray              # Swing-Ziel: w_k mal Spannweite des Wertes, auf Summe 1
    chosen: Result = None
    extra: dict = field(default_factory=dict)


def analyse(settings: Settings) -> Analysis:
    case = make_case(settings.n, settings.m, settings.seed, settings.delta, settings.scale, settings.alpha)
    results = evaluate_methods(case, settings.sigma, settings.seed)
    return Analysis(settings, case, results, sc.top1(case.utility), el.swing_target(case.w, case.value_range), results[settings.method])


# ----------------------------------------------------------------------------- Experiment 1: Methoden gegeneinander

def experiment_methods(ms=C.EXP_M, n=C.EXP_N, sigma=C.EXP_SIGMA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for scale in C.SCALES:
        for m in ms:
            acc = {k: ([], [], [], []) for k in ALL_METHODS}
            for s in seeds:
                res = evaluate_methods(make_case(n, m, s, 1.0, scale), sigma, s)
                for k, r in res.items():
                    acc[k][0].append(r.hit)
                    acc[k][1].append(r.tau)
                    acc[k][2].append(r.loss)
                    acc[k][3].append(r.tv)
            for k in ALL_METHODS:
                rows.append({"scale": scale, "m": m, "method": k, "hit": float(np.mean(acc[k][0])), "tau": float(np.mean(acc[k][1])),
                             "loss": float(np.mean(acc[k][2])), "tv": float(np.mean(acc[k][3])), "judgments": el.judgment_count(k, m)})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: ein Kriterium kaum verschieden

def experiment_narrow(deltas=C.DELTA_CHOICES, m=C.EXP_NARROW_M, n=C.EXP_N, sigma=C.EXP_SIGMA, methods=C.EXP_NARROW_METHODS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for scale in C.SCALES:
        for d in deltas:
            hits = {k: [] for k in methods}
            for s in seeds:
                res = evaluate_methods(make_case(n, m, s, d, scale), sigma, s, methods)
                for k in methods:
                    hits[k].append(res[k].hit)
            for k in methods:
                rows.append({"scale": scale, "delta": d, "method": k, "hit": float(np.mean(hits[k])), "se": float(np.sqrt(np.mean(hits[k]) * (1 - np.mean(hits[k])) / len(seeds)))})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: Rangformeln und die Verteilung der Gewichte

def experiment_rank_formulas(alphas=C.ALPHA_CHOICES, m=C.EXP_RANK_M, draws=C.EXP_RANK_DRAWS) -> list[dict]:
    rows = []
    for a in alphas:
        rng = SplitMix64(31337 + int(10 * a))
        errs = {k: [] for k in ("gleich", "rangsumme", "rangrezipr", "roc")}
        for _ in range(draws):
            w = draw_weights(m, a, rng)
            ranks = np.empty(m, dtype=int)
            ranks[np.lexsort((np.arange(m), -w))] = np.arange(1, m + 1)
            cand = {"gleich": np.full(m, 1.0 / m), "rangsumme": el.weights_rank_sum(ranks), "rangrezipr": el.weights_rank_reciprocal(ranks), "roc": el.weights_roc(ranks)}
            for k, v in cand.items():
                errs[k].append(el.tv_distance(v, w))
        for k, e in errs.items():
            rows.append({"alpha": a, "method": k, "tv": float(np.mean(e))})
    return rows


def roc_expectation_check(ms=(3, 5, 8), draws=10000) -> list[dict]:
    """Größte Abweichung zwischen der ROC-Formel und dem Mittel der geordneten Gewichte bei Gleichverteilung über dem Simplex."""
    rows = []
    for m in ms:
        rng = SplitMix64(4242 + m)
        acc = np.zeros(m)
        for _ in range(draws):
            acc += np.sort(draw_weights(m, 1.0, rng))[::-1]
        formula = el.weights_roc(np.arange(1, m + 1))
        rows.append({"m": m, "max_deviation": float(np.abs(acc / draws - formula).max()), "first_formula": float(formula[0]), "first_sample": float(acc[0] / draws)})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Rauschen und die Skala 1 bis 9

def experiment_noise(sigmas=C.SIGMA_CHOICES, m=5, n=C.EXP_N, scale="roh", methods=C.EXP_NOISE_METHODS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for sg in sigmas:
        hits = {k: [] for k in methods}
        tvs = {k: [] for k in methods}
        for s in seeds:
            res = evaluate_methods(make_case(n, m, s, 1.0, scale), sg, s, methods)
            for k in methods:
                hits[k].append(res[k].hit)
                tvs[k].append(res[k].tv)
        for k in methods:
            rows.append({"sigma": sg, "method": k, "hit": float(np.mean(hits[k])), "tv": float(np.mean(tvs[k]))})
    return rows


def bwm_clip_stats(alphas=(1.0, 0.5), ms=C.EXP_CLIP_M, draws=C.EXP_CLIP_DRAWS) -> list[dict]:
    """Anteil der Verhältnisse w_B / w_j über 9 und Anteil der Instanzen mit mindestens einem Urteil über 9 (ohne Rauschen)."""
    rows = []
    for a in alphas:
        for m in ms:
            rng = SplitMix64(9001 + m + int(10 * a))
            share, anyc = [], []
            for _ in range(draws):
                w = draw_weights(m, a, rng)
                B, W = int(np.argmax(w)), int(np.argmin(w))
                rb = np.delete(w[B] / w, B)
                rw = np.delete(w / w[W], W)
                share.append(float((rb > C.BWM_MAX).mean()))
                anyc.append(bool((rb > C.BWM_MAX).any() or (rw > C.BWM_MAX).any()))
            rows.append({"alpha": a, "m": m, "ratio_share": float(np.mean(share)), "any_clipped": float(np.mean(anyc))})
    return rows


# ----------------------------------------------------------------------------- Verdikte (live aus den Zeilen)

def _row(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def verdict_methods(rows: list[dict]) -> tuple[str, str]:
    gaps = []
    for scale in C.SCALES:
        for m in sorted({r["m"] for r in rows}):
            ratio_best = max(_row(rows, scale=scale, m=m, method=k)["hit"] for k in ("direkt", "swing", "bwm"))
            rank_best = max(_row(rows, scale=scale, m=m, method=k)["hit"] for k in C.RANK_METHODS)
            gaps.append(100 * (ratio_best - rank_best))
    return "info", (f"Die Rangformeln liegen {de(min(gaps), 0)} bis {de(max(gaps), 0)} Punkte hinter der besten Verhältnismethode (direkt, Swing, Best-Worst); diese liegen "
                    "untereinander im Rauschen. Dafür braucht eine Rangformel nur eine Rangordnung der Kriterien.")


def verdict_narrow(rows: list[dict]) -> tuple[str, str]:
    lo = min(d for d in {r["delta"] for r in rows})
    w = _row(rows, scale="roh", delta=lo, method="wahr")
    d = _row(rows, scale="roh", delta=lo, method="direkt")
    s = _row(rows, scale="roh", delta=lo, method="swing")
    return "warning", (f"Streut das erste Kriterium nur noch zu {de(100 * lo, 0)} %, trifft die Summe mit den wahren Gewichten Platz 1 in {de(100 * w['hit'], 0)} % der Fälle "
                       f"und mit direkt angegebener Wichtigkeit in {de(100 * d['hit'], 0)} %; Swing-Gewichte, die den Schwenk in der Stichprobe bewerten, in {de(100 * s['hit'], 0)} %.")


def verdict_rank(rows: list[dict]) -> tuple[str, str]:
    hi = max(r["alpha"] for r in rows)
    roc = _row(rows, alpha=hi, method="roc")["tv"]
    eq = _row(rows, alpha=hi, method="gleich")["tv"]
    mid = _row(rows, alpha=1.0, method="roc")["tv"]
    kind = "warning" if roc > eq else "info"
    return kind, (f"Bei gleichverteilten Gewichten (α = 1) ist ROC mit einem Abstand von {de(mid, 2)} die beste Rangformel. Liegen die Gewichte nahe beieinander (α = {de(hi, 0)}), "
                  f"ist ROC mit {de(roc, 2)} {'schlechter' if roc > eq else 'nicht schlechter'} als gleiche Gewichte ({de(eq, 2)}).")


def verdict_noise(rows: list[dict], clip: list[dict]) -> tuple[str, str]:
    top = max(r["sigma"] for r in rows)
    swing = _row(rows, sigma=top, method="swing")["hit"]
    roc = _row(rows, sigma=top, method="roc")["hit"]
    c = _row(clip, alpha=1.0, m=5)
    return "info", (f"Bei σ = {de(top, 1)} trifft Swing Platz 1 in {de(100 * swing, 0)} %, ROC in {de(100 * roc, 0)} %. Beim Best-Worst sind bei gleichverteilten Gewichten "
                    f"{de(100 * c['ratio_share'], 0)} % der Verhältnisse zum Besten größer als 9 und werden auf der Skala abgeschnitten.")
