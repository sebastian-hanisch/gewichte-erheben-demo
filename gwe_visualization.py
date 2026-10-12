"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import gwe_constants as C
from gwe_evaluation import Analysis, de

ARROW = {1: "↑", -1: "↓"}
SHORT = dict(C.METHOD_SHORT, wahr="wahre Gewichte")


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def criterion_names(a: Analysis) -> list[str]:
    inst = a.case.matrix
    return [f"K{k + 1} {ARROW[int(inst.directions[k])]} {inst.units[k]}" for k in range(inst.m)]


def build_weights(a: Analysis) -> go.Figure:
    """Wahre Gewichte des Entscheiders, die zur gezeigten Spannweite passenden (Swing-Ziel) und die der gewählten Methode."""
    x = [f"K{k + 1}" for k in range(a.case.matrix.m)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=a.case.w, name="wahre Gewichte (Entscheider)", marker_color=C.COLOR_TRUE))
    fig.add_trace(go.Bar(x=x, y=a.target, name="passend zur Spannweite (Swing-Ziel)", marker_color=C.COLOR_FIT))
    fig.add_trace(go.Bar(x=x, y=a.chosen.weights, name=f"erhoben: {SHORT[a.settings.method]}", marker_color=C.COLOR_ELICITED))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Gewicht", range=[0, 1])
    return _fixed(fig, 320)


def build_judgments(a: Analysis) -> pd.DataFrame:
    """Die Urteile der gewählten Methode je Kriterium."""
    m = a.case.matrix.m
    j = a.chosen.judgments
    cols = {"Kriterium": [f"K{k + 1}" for k in range(m)], "wahres Gewicht": a.case.w, "Spannweite des Wertes": a.case.value_range}
    meth = a.settings.method
    if meth in ("direkt", "swing"):
        cols["Punkte"] = j["points"]
    elif meth in C.RANK_METHODS:
        cols["Rang (wahrgenommen)"] = j["ranks"]
    elif meth == "bwm":
        cols["Verhältnis zum Besten"] = j["a_B"]
        cols["Verhältnis zum Schlechtesten"] = j["a_W"]
    cols["erhobenes Gewicht"] = a.chosen.weights
    return pd.DataFrame(cols)


def style_judgments(df: pd.DataFrame):
    fmt = {c: (lambda v: de(v, 3)) for c in ("wahres Gewicht", "Spannweite des Wertes", "erhobenes Gewicht")}
    for c in ("Punkte", "Rang (wahrgenommen)", "Verhältnis zum Besten", "Verhältnis zum Schlechtesten"):
        if c in df.columns:
            fmt[c] = lambda v: de(v, 0)
    return df.style.format(fmt)


def build_method_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.case.matrix.n)
    rows = []
    for k, r in a.results.items():
        rows.append({"Methode": SHORT[k], "Urteile": r.count, "Platz 1": names[r.top1], "Trifft den Besten": "ja" if r.hit else "nein",
                     "τ gegen Nutzen": de(r.tau, 2), "Nutzenverlust": de(r.loss, 3), "Abstand zu w": de(r.tv, 3)})
    return pd.DataFrame(rows)


def _bars(rows: list[dict], x_key: str, x_labels: dict, y_key: str, methods, ytitle: str, height=360) -> go.Figure:
    fig = go.Figure()
    for k in methods:
        rs = [r for r in rows if r["method"] == k]
        fig.add_trace(go.Bar(x=[x_labels[r[x_key]] for r in rs], y=[100 * r[y_key] for r in rs], name=SHORT[k], marker_color=C.METHOD_COLORS[k]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=ytitle, range=[0, 102])
    return _fixed(fig, height)


def build_methods(rows: list[dict]) -> go.Figure:
    methods = [k for k in ("gleich", "wahr", "direkt", "swing", "rangsumme", "rangrezipr", "roc", "bwm") if any(r["method"] == k for r in rows)]
    rs = [dict(r, key=(r["scale"], r["m"])) for r in rows]
    labels = {(s, m): f"{'Güte' if s == 'guete' else 'Rohwerte'}, m = {m}" for s in C.SCALES for m in sorted({r["m"] for r in rows})}
    return _bars(rs, "key", labels, "hit", methods, "Platz 1 getroffen (%)")


def build_narrow(rows: list[dict], scale: str = "roh") -> go.Figure:
    fig = go.Figure()
    for k in C.EXP_NARROW_METHODS:
        rs = sorted([r for r in rows if r["scale"] == scale and r["method"] == k], key=lambda r: -r["delta"])
        fig.add_trace(go.Scatter(x=[de(r["delta"], 2) for r in rs], y=[100 * r["hit"] for r in rs], mode="lines+markers", name=SHORT[k],
                                 line=dict(color=C.METHOD_COLORS[k])))
    fig.update_xaxes(title="Anteil der Streuung im ersten Kriterium (1 = unverändert)")
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    return _fixed(fig, 340)


def build_rank_formulas(rows: list[dict]) -> go.Figure:
    labels = {a: f"α = {de(a, 1)}" for a in sorted({r["alpha"] for r in rows})}
    fig = go.Figure()
    for k in ("gleich", "rangsumme", "rangrezipr", "roc"):
        rs = [r for r in rows if r["method"] == k]
        fig.add_trace(go.Bar(x=[labels[r["alpha"]] for r in rs], y=[r["tv"] for r in rs], name=SHORT[k], marker_color=C.METHOD_COLORS[k]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Abstand zu den wahren Gewichten (TV, kleiner ist besser)", rangemode="tozero")
    return _fixed(fig, 340)


def build_noise(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for k in C.EXP_NOISE_METHODS:
        rs = [r for r in rows if r["method"] == k]
        fig.add_trace(go.Scatter(x=[r["sigma"] for r in rs], y=[100 * r["hit"] for r in rs], mode="lines+markers", name=SHORT[k], line=dict(color=C.METHOD_COLORS[k])))
    fig.update_xaxes(title="Rauschen σ der Urteile", tickvals=list(C.SIGMA_CHOICES))
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    return _fixed(fig, 340)
