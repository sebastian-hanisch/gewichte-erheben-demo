"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import gwe_constants as C
from gwe_evaluation import Analysis, de

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "–": "-", "—": "-", "τ": "tau", "σ": "sigma", "δ": "delta", "α": "alpha", "≥": ">=", "−": "-", "€": "EUR", "·": "*"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    inst = a.case.matrix
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Gewichte erheben: direkt, Swing, Rangordnung"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Alternativen n = {s.n}, Kriterien m = {s.m}, Methode: {C.METHOD_SHORT[s.method]}, Rauschen sigma = {de(s.sigma, 1)}, Streuung des ersten Kriteriums "
                            f"delta = {de(s.delta, 2)}, Wertskala: {C.SCALE_LABELS[s.scale]}, Gewichtsverteilung alpha = {de(s.alpha, 1)}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    r = a.chosen
    pdf.multi_cell(0, 5, _s(f"Gewählte Methode: Platz 1 {'getroffen' if r.hit else 'nicht getroffen'}, tau {de(r.tau, 2)}, Nutzenverlust {de(r.loss, 3)}, "
                            f"Abstand zu den wahren Gewichten {de(r.tv, 3)}, Urteile {r.count}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head = ["Kriterium", "wahres Gewicht", "Spannweite", "Swing-Ziel", "erhobenes Gewicht"]
    widths = [28, 36, 36, 36, 44]
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for k in range(inst.m):
        row = [f"K{k + 1} ({'+' if inst.directions[k] > 0 else '-'}) {inst.units[k]}", de(a.case.w[k], 3), de(a.case.value_range[k], 3), de(a.target[k], 3), de(r.weights[k], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Alle Methoden auf derselben Matrix"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for k, res in a.results.items():
        pdf.cell(0, 5, _s(f"{C.METHOD_SHORT.get(k, 'wahre Gewichte')}: Platz 1 {'ja' if res.hit else 'nein'}, tau {de(res.tau, 2)}, Verlust {de(res.loss, 3)}, "
                          f"Abstand {de(res.tv, 3)}, Urteile {res.count}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Das Urteilsmodell ist angenommen (Rauschen, Rundung), nicht am Menschen gemessen. Quelle: gewichte-erheben-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
