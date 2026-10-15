"""Gewichte erheben: direkt, Swing, Rangordnung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Linie Mehrkriterien-Ranking (Kind der gewichteten Summe). Dieselbe neutrale Entscheidungsmatrix; ein simulierter Entscheider kennt seine Gewichte,
und jede Erhebungsmethode bekommt ein verrauschtes Urteil. Das Urteilsmodell ist angenommen, nicht am Menschen gemessen. Kern: gwe_elicit.py, gwe_score.py;
Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import gwe_constants as C
from gwe_evaluation import (
    Settings,
    analyse,
    bwm_clip_stats,
    de,
    experiment_methods,
    experiment_narrow,
    experiment_noise,
    experiment_rank_formulas,
    roc_expectation_check,
    verdict_methods,
    verdict_narrow,
    verdict_noise,
    verdict_rank,
)
from gwe_pdf_export import build_pdf
from gwe_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from gwe_visualization import (
    SHORT,
    alt_names,
    build_judgments,
    build_method_table,
    build_methods,
    build_narrow,
    build_noise,
    build_rank_formulas,
    build_weights,
    style_judgments,
)

st.set_page_config(page_title="Gewichte erheben – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Die vier Experimente rechnen über 200 feste Matrizen …")
def _experiments():
    return {
        "methods": experiment_methods(),
        "narrow": experiment_narrow(),
        "rank": experiment_rank_formulas(),
        "roc": roc_expectation_check(),
        "noise": experiment_noise(),
        "clip": bwm_clip_stats(),
    }


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎚️ Gewichte erheben: direkt, Swing, Rangordnung")
st.markdown(
    """
**Die „wahren“ Gewichte sind nicht das Ziel, das Ziel sind Gewichte zur gezeigten Spannweite.** Dies ist das dritte Stück der Linie **Mehrkriterien-Ranking**: Die gewichtete
Summe braucht Gewichte, und ein Entscheider nennt sie nicht einfach. Die Demo erhebt sie auf fünf Wegen (**direkt**, **Swing**, drei **Rangformeln** und **Best-Worst**)
und misst jeden an der **Entscheidung**, die die Summe damit trifft. Der Entscheider ist simuliert, ebenso sein **Urteil**: Rauschen und Rundung sind angenommen, nicht am
Menschen gemessen.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So werden Gewichte erhoben", expanded=True):
    st.markdown(
        """
1. **Direkt:** Punkte 0 bis 100 je Kriterium für die Wichtigkeit. Ein Urteil je Kriterium; die Frage bezieht sich nicht auf die Alternativen, die zur Wahl stehen.
2. **Swing:** Wie viel ist der Schwenk vom schlechtesten zum besten Wert **in dieser Stichprobe** wert? Der größte Schwenk bekommt 100 Punkte. Ein Urteil je Kriterium;
   die Gewichte gelten für die gezeigte Spannweite.
3. **Rangformeln:** Nur die Rangordnung der Kriterien; **Rangsumme**, **Rangreziprok** und **ROC** (Rangzentroid) machen daraus feste Gewichte. Eine Rangordnung genügt.
4. **Best-Worst:** Wie viel wichtiger ist das beste Kriterium als jedes andere, und jedes andere als das schlechteste (Skala 1 bis 9)? 2m − 3 Urteile, ein lineares LP.
5. **Gemessen wird an der Entscheidung:** Platz 1 der gewichteten Summe (Min-Max) gegen den Besten des Entscheiders, Kendalls τ der Rangfolge und der Nutzenverlust.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen zur Wahl stehen.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist; Best-Worst braucht mindestens drei.")
    delta = st.selectbox("Streuung des ersten Kriteriums", C.DELTA_CHOICES, key="delta_select", format_func=lambda d: "unverändert" if d == 1.0 else f"{de(100 * d, 0)} % der Streuung",
                         help="Wie stark sich die Alternativen im ersten Kriterium unterscheiden. Bei kleiner Streuung ist die Spannweite in der Stichprobe schmal: dort trennen sich "
                              "Swing-Gewichte von direkt angegebener Wichtigkeit.")
    st.markdown("**Erhebung**")
    method = st.selectbox("Methode", C.METHODS, key="method_select", format_func=lambda k: C.METHOD_LABELS[k])
    sigma = st.selectbox("Rauschen σ der Urteile", C.SIGMA_CHOICES, key="sigma_select", format_func=lambda s: de(s, 1),
                         help="Streuung des Logarithmus jedes Verhältnisurteils (lognormal). 0: exakte Urteile; 0,8: ein Verhältnis von 2 wird oft als 1 oder 4 genannt.")
    st.markdown("**Entscheider (simuliert)**")
    scale = st.selectbox("Wertskala", C.SCALES, key="scale_select", format_func=lambda k: C.SCALE_LABELS[k],
                         help="Auf welcher Skala der verborgene Entscheider linear bewertet.")
    alpha = st.selectbox("Verteilung der wahren Gewichte α", C.ALPHA_CHOICES, key="alpha_select", format_func=lambda a: de(a, 1),
                         help="Dirichlet-Parameter: 1 gleichverteilt über dem Simplex, 0,5 ungleicher (ein Kriterium dominiert), 5 fast gleiche Gewichte.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "method_select": method, "sigma_select": float(sigma), "delta_select": float(delta),
                   "scale_select": scale, "alpha_select": float(alpha), "seed_input": int(seed)})

settings = Settings(int(n), int(m), method, float(sigma), float(delta), scale, float(alpha), int(seed))
a = _analysis(settings)
names = alt_names(a.case.matrix.n)
r = a.chosen
ref = a.results["wahr"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Platz 1 der Summe", names[r.top1], delta=f"bester nach Nutzen: {names[a.best]}", delta_color="off")
c2.metric("Trifft den Besten", "ja" if r.hit else "nein", delta=f"τ gegen Nutzen: {de(r.tau, 2)}", delta_color="off",
          help="Ob Platz 1 der gewichteten Summe (Min-Max) mit den erhobenen Gewichten der Alternative entspricht, die der simulierte Entscheider am höchsten bewertet; "
               "τ ist Kendalls τ-b der ganzen Rangfolge gegen den verborgenen Nutzen.")
c3.metric("Nutzenverlust", de(r.loss, 3), delta=f"mit wahren Gewichten: {de(ref.loss, 3)}", delta_color="off",
          help="Nutzen der besten Alternative minus Nutzen der gewählten; 0 heißt: richtig gewählt.")
c4.metric("Urteile", f"{r.count}", delta=f"Abstand zu w: {de(r.tv, 3)}", delta_color="off",
          help="Zahl der Einzelurteile (bei den Rangformeln die untere Schranke der Vergleiche für eine Rangordnung). Der Abstand zu den wahren Gewichten (halber L1-Abstand) ist "
               "für Swing kein Fehlermaß: Swing-Gewichte sollen von w abweichen, weil sie die Spannweite enthalten.")

st.plotly_chart(build_weights(a), width="stretch", key="weights_chart")
tab_j, tab_m = st.tabs(["Urteile der gewählten Methode", "Alle Methoden im Vergleich"])
with tab_j:
    if method in ("gleich",):
        st.info("Gleiche Gewichte brauchen kein Urteil.")
    else:
        st.dataframe(style_judgments(build_judgments(a)), hide_index=True, width="stretch")
        if method == "bwm":
            st.caption(f"Bestes Kriterium: K{a.chosen.judgments['B'] + 1}, schlechtestes: K{a.chosen.judgments['W'] + 1}; "
                       f"{a.chosen.judgments['clipped']} Urteil(e) lagen über 9 und wurden auf 9 abgeschnitten.")
        if method == "swing":
            st.caption("Die Punkte beziehen sich auf die Spannweite der Alternativen: „wahres Gewicht“ mal „Spannweite des Wertes“, auf 100 Punkte für den größten Schwenk gebracht.")
with tab_m:
    st.dataframe(build_method_table(a), hide_index=True, width="stretch")
    st.caption("Dieselbe Matrix, derselbe Entscheider, jede Methode mit eigenem Urteil (aus demselben Urteilsstrom, die Wahl der Methode ändert die anderen nie).")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="gewichte-erheben.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Treffen die erhobenen Gewichte den Besten?")
if r.hit and ref.hit:
    st.success(f"{SHORT[method]} trifft den Besten ({names[a.best]}), ebenso die wahren Gewichte.")
elif r.hit and not ref.hit:
    st.success(f"{SHORT[method]} trifft den Besten ({names[a.best]}), die wahren Gewichte verfehlen ihn: Sie gelten auf der Skala des Entscheiders, nicht auf der "
               "Spannweite der gezeigten Alternativen.")
elif (not r.hit) and ref.hit:
    st.warning(f"{SHORT[method]} verfehlt den Besten ({names[a.best]}, gewählt: {names[r.top1]}), die wahren Gewichte treffen ihn.")
else:
    st.warning(f"Weder {SHORT[method]} noch die wahren Gewichte treffen den Besten ({names[a.best]}, gewählt: {names[r.top1]}): bei diesem Fall liegt es an der Skala.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption("200 feste Matrizen (SWEEP_SEEDS), n = 20, Min-Max, unabhängig vom Regler n. Die vier Experimente rechnen zusammen rund 12 Sekunden und starten erst auf Klick.")
if st.button("▶ Experimente rechnen", key="exp_start"):
    st.session_state["exp_rows"] = _experiments()
exp = st.session_state.get("exp_rows")
if exp:
    st.markdown("### Experiment 1: Welche Methode trifft am besten?")
    st.caption("Rauschen σ = 0,2, wahre Gewichte gleichverteilt, Entscheider in der Güte oder in den Rohwerten, 3 und 5 Kriterien. Standardfehler 1,3 bis 3,5 Punkte: "
               "Unterschiede unter etwa 6 Punkten sind nicht belastbar.")
    st.plotly_chart(build_methods(exp["methods"]), width="stretch", key="methods_chart")
    verdict(verdict_methods(exp["methods"]))
    cost = {r_["method"]: r_["judgments"] for r_ in exp["methods"] if r_["m"] == 5 and r_["scale"] == "roh"}
    st.caption("Urteile bei 5 Kriterien: " + ", ".join(f"{SHORT[k]} {cost[k]}" for k in ("direkt", "swing", "rangsumme", "bwm")) +
               " (Rangformeln: untere Schranke der Vergleiche für eine Rangordnung). Die Paarvergleichsmatrix würde 10 Vergleiche brauchen.")

    st.markdown("### Experiment 2: Wenn ein Kriterium kaum streut")
    st.caption("Das erste Kriterium wird auf einen Anteil seiner Streuung gestaucht; Entscheider in den Rohwerten, 5 Kriterien, σ = 0,2. Der Entscheider selbst ändert sich nicht.")
    st.plotly_chart(build_narrow(exp["narrow"], "roh"), width="stretch", key="narrow_chart")
    verdict(verdict_narrow(exp["narrow"]))
    g = [x for x in exp["narrow"] if x["scale"] == "guete" and x["delta"] == min(d["delta"] for d in exp["narrow"])]
    gd = {x["method"]: x for x in g}
    st.caption(f"Entscheider in der Güte, 5 % der Streuung: wahre Gewichte {de(100 * gd['wahr']['hit'], 0)} %, direkt {de(100 * gd['direkt']['hit'], 0)} %, "
               f"Swing {de(100 * gd['swing']['hit'], 0)} % (die Güte-Skala ist bei den logarithmischen Kriterien nicht linear in den Rohwerten, daher bleibt Swing hier unter 100 %).")

    st.markdown("### Experiment 3: Rangformeln und die Verteilung der Gewichte")
    st.caption("Exakte Rangfolge (σ = 0), 5 Kriterien, 2 000 Ziehungen je α. ROC ist der Erwartungswert der geordneten Gewichte bei Gleichverteilung über dem Simplex.")
    st.plotly_chart(build_rank_formulas(exp["rank"]), width="stretch", key="rank_chart")
    verdict(verdict_rank(exp["rank"]))
    rc = exp["roc"]
    st.caption("Die ROC-Formel gegen das Mittel der geordneten Gewichte bei 10 000 Ziehungen: größte Abweichung " +
               ", ".join(f"m = {x['m']}: {de(x['max_deviation'], 4)}" for x in rc) + ".")

    st.markdown("### Experiment 4: Rauschen und die Skala 1 bis 9")
    st.caption("Entscheider in den Rohwerten, 5 Kriterien. Unten: Anteil der Verhältnisse zum besten Kriterium über 9 und Anteil der Fälle mit mindestens einem abgeschnittenen Urteil.")
    st.plotly_chart(build_noise(exp["noise"]), width="stretch", key="noise_chart")
    verdict(verdict_noise(exp["noise"], exp["clip"]))
    st.dataframe([{"α": de(c["alpha"], 1), "Kriterien": c["m"], "Verhältnisse > 9": f"{de(100 * c['ratio_share'], 0)} %",
                   "Fälle mit mindestens einem Abschnitt": f"{de(100 * c['any_clipped'], 0)} %"} for c in exp["clip"]], hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Urteilsmodell stimmt** (lognormales Rauschen, gleiches σ für alle Methoden, Rundung auf ganze Punkte) | Es ist angenommen, nicht gemessen. Echte Urteile haben Anker, runde Zahlen und systematische Verzerrungen; die Demo sagt nichts über Menschen. | – |
| **Direkte Wichtigkeit ist bereichsunempfindlich** | Das ist die Annahme hinter dem Befund zur schmalen Spannweite (Fischer 1995, nur aus der Zusammenfassung); in der Simulation ist sie fest eingebaut. | – |
| **Der Entscheider kennt seine Gewichte** | Hier sind es Zahlen der Simulation; ein Entscheider hat Gewichte nicht fertig im Kopf, sondern konstruiert sie in der Erhebung. | Aus Beispielen lernen |
| **Die Gewichte sind gleichverteilt (oder Dirichlet)** | ROC ist nur dann die beste Rangformel; bei fast gleichen Gewichten ist es gleiche Gewichte nicht überlegen (Experiment 3). | Unsichere Gewichte |
| **Die Konsistenz der Urteile** | Die Simulation kann sie nicht prüfen; ob Best-Worst konsistenter ist als eine Paarvergleichsmatrix, ist eine Frage an Menschen. | AHP |
| **Eine Normierung (Min-Max)** | Die Wahl der Normierung ist im vorigen Stück gemessen; hier ist sie fest. | Gewichtete Summe |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gewichtete Summe.** Score $U_i = \sum_k w_k\, N_k(s_{ik})$ mit der Min-Max-Normierung $N_k$ und $w_k \ge 0$, $\sum_k w_k = 1$.

**Swing.** Der Schwenk des Kriteriums $k$ ist $w_k\, \Delta_k$ mit der Spannweite $\Delta_k = \max_i f_k(s_{ik}) - \min_i f_k(s_{ik})$ des Wertes in der Stichprobe; die
Swing-Gewichte sind $w_k \Delta_k / \sum_j w_j \Delta_j$. Bei einem in den Rohwerten linearen Entscheider und Min-Max-Normierung sind sie genau die passenden.

**Rangformeln** für Rang $r = 1, \dots, m$: Rangsumme $w_r \propto m + 1 - r$; Rangreziprok $w_r \propto 1/r$; ROC $w_r = \tfrac1m \sum_{j=r}^{m} 1/j$. ROC ist der Erwartungswert
des $r$-größten Gewichts bei Gleichverteilung über dem Simplex (hier gegen Simulation geprüft).

**Best-Worst (linear).** Mit Verhältnissen $a_{Bj}$ (bestes zu $j$) und $a_{jW}$ ($j$ zum schlechtesten): minimiere $\xi$ unter $|w_B - a_{Bj} w_j| \le \xi$,
$|w_j - a_{jW} w_W| \le \xi$, $\sum w = 1$, $w \ge 0$; 2m − 3 Urteile.

**Urteilsmodell.** Ein Verhältnis $\rho$ wird als $\rho \cdot e^{\sigma \varepsilon}$ mit $\varepsilon \sim N(0,1)$ genannt, bei Punkten auf ganze Zahlen 0 bis 100, bei Best-Worst auf
ganze Zahlen 1 bis 9 gerundet und abgeschnitten.

**Gewichtsabstand.** $\mathrm{TV}(\hat w, w) = \tfrac12 \sum_k |\hat w_k - w_k|$.

**Literatur** (nur aus Zusammenfassungen, nicht im Original geprüft): F. H. Barron, B. E. Barrett,
*Decision quality using ranked attribute weights*, Management Science 42(11), 1996, S. 1515–1523 (ROC genauer als die anderen Rangformeln); G. W. Fischer, *Range sensitivity of
attribute weights in multiattribute value models*, Organizational Behavior and Human Decision Processes 62(3), 1995, S. 252–266 (direkte Wichtigkeit bereichsunempfindlich).
J. Rezaei, *Best-worst multi-criteria decision-making method*, Omega 53, 2015, S. 49–57.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html)."
)
