# Gewichte erheben: direkt, Swing, Rangordnung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-gewichte-erheben-demo.streamlit.app/)**

Drittes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind der
[gewichteten Summe](https://github.com/sebastian-hanisch/gewichtete-summe-demo). Dort steht, dass ein Gewicht zu einer Skala gehört und dass gleiche Gewichte nur in knapp
der Hälfte der Fälle Platz 1 treffen. Dieses Stück fragt, **woher die Gewichte kommen**: Es erhebt sie auf fünf Wegen von einem simulierten Entscheider und misst jeden an der
Entscheidung, die die Summe damit trifft.

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der **Entscheider** kennt
seine Gewichte w; seine **Urteile** sind ein **angenommenes** Modell (Rauschen, Rundung), nicht am Menschen gemessen.

## Warum dieses Problem

**Die „wahren“ Gewichte sind nicht das Ziel, das Ziel sind Gewichte zur gezeigten Spannweite.** Ein Entscheider nennt seine Gewichte nicht einfach; man fragt ihn. Fragt man nach
Wichtigkeit, bekommt man etwas, das nicht an die Alternativen gebunden ist; fragt man nach dem Schwenk vom schlechtesten zum besten Wert **in der Stichprobe**, bekommt man
Gewichte für genau diese Spannweite. Die Kernfrage der Demo: Wie gut treffen die Methoden den Besten, und wann trennen sie sich?

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zahlen sind neu gemessen.** Die Phase 0 rechnete das Urteilsrauschen mit `numpy.default_rng`; die Demo nutzt den SplitMix64-Strom des Generators. Die Werte verschieben
  sich um bis zu vier Punkte (Rohwerte, 5 Kriterien: direkt 90,5 % statt 94,0 %, Swing 95,0 % statt 92,5 %, ROC 81,5 % statt 84,0 %). Dadurch liegen direkt und Swing bei
  gleichmäßigen Kriterien in der Demo etwas weiter auseinander als in der Phase 0 (4,5 Punkte bei 1,5 bis 2,1 Punkten Standardfehler): kein belastbarer Unterschied.
- **Die Verteilung der wahren Gewichte kommt aus SplitMix64**, nicht aus `numpy.random.Generator.dirichlet` (dessen Strom über Versionen nicht stabil ist): α = 1 über
  Exponentialverteilte, α = 0,5 über Z²/2, α = 5 als Summe von fünf Exponentialverteilten.
- **Die Messgröße ist die Entscheidung.** Der Gewichtsabstand steht in der Tabelle, aber Swing soll von w abweichen (Spannweite): Für Swing ist er kein Fehlermaß.
- **Die Rangformeln zählen als eine Rangordnung,** gemessen an der unteren Schranke ⌈log₂ m!⌉ der Vergleiche (7 bei 5 Kriterien); die Phase 0 hatte m − 1 gezählt.
- **Die Konsistenz von Best-Worst gegenüber der Paarvergleichsmatrix ist nicht simuliert.** Sie ist ein Merkmal menschlicher Urteile; die Demo zeigt nur die Zahl der Urteile
  und das Abschneiden auf der Skala 1 bis 9.
- **Die Experimente starten auf Klick** (rund 12 Sekunden), nicht beim Laden.
- **Der Standard-Seed ist 1**; die Presets sind auf Seeds abgestimmt, bei denen der Effekt sichtbar ist (`test_every_preset_shows_the_effect_it_is_named_for`).

## Modell

- **Matrix:** n Alternativen (5 bis 40), m Kriterien (3 bis 8), unabhängig, Rohwerte mit Richtung und Einheit; optional ein Kriterium auf den Anteil δ seiner Streuung gestaucht.
- **Entscheider:** wahre Gewichte aus Dirichlet(α) mit α = 0,5 / 1 / 5; Nutzen linear in der Güte oder linear in den Rohwerten (feste Nenn-Spannweite je Kriterium).
- **Urteile (angenommen):** *direkt*: Punkte 0 bis 100 ∝ w, ohne Bezug zur Spannweite; *Swing*: Punkte für den Schwenk worst → best **in der Stichprobe**, ∝ w · Spannweite, der
  größte Schwenk bekommt 100; *Rangformeln*: wahrgenommene Wichtigkeit w · exp(σε) liefert die Rangordnung, daraus Rangsumme, Rangreziprok oder ROC; *Best-Worst*:
  Verhältnisse zum besten und zum schlechtesten Kriterium auf ganze Zahlen 1 bis 9 gerundet und abgeschnitten, lineares LP. Rauschen lognormal mit Streuung σ des Logarithmus.
- **Summe:** gewichtete Summe auf Min-Max-normierten Rohwerten (die Wahl der Normierung steht im Vorgänger).
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), n = 20, unabhängig vom Regler n.

Formeln im 📐-Expander der App.

## Methodik

1. **Alle Methoden auf derselben Matrix** mit demselben Urteilsstrom; die Wahl der Methode ändert die Urteile der anderen nie.
2. **Messung an der Entscheidung:** Platz 1 gegen den Besten des Entscheiders, Kendalls τ-b der Rangfolge, Nutzenverlust.
3. **Zwei Welten:** der Entscheider denkt in der Güte oder in den Rohwerten.
4. **Verschmälerung:** ein Kriterium streut nur noch zu einem Anteil δ; der Entscheider selbst bleibt unverändert.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Mittel über 200 feste Matrizen, n = 20, σ = 0,2 wenn nicht anders vermerkt, `tests/test_claims.py`. **Der Standardfehler einer Trefferquote ist 1,3 bis 3,5
Punkte; Unterschiede unter etwa 6 Punkten sind nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Welche Methode trifft Platz 1? (Welt Güte m = 3 / 5, Rohwerte m = 3 / 5) | gleiche Gewichte 47,0 / 46,0 / 51,5 / 56,0 %; wahre Gewichte 85,0 / 67,0 / 94,0 / 96,5 %; direkt 86,5 / 66,5 / 87,5 / 90,5 %; Swing 82,5 / 66,0 / 92,5 / 95,0 %; Best-Worst 80,0 / 66,5 / 85,0 / 93,5 %; **Rangsumme 65,0 / 58,5 / 68,0 / 73,5 %; Rangreziprok 64,0 / 57,5 / 67,5 / 80,0 %; ROC 73,5 / 58,5 / 74,5 / 81,5 %**. | `test_methods_first_place` |
| Wie weit liegen die Rangformeln zurück? | 8 bis 18 Punkte hinter der besten Verhältnismethode (direkt, Swing, Best-Worst); diese liegen untereinander im Rauschen. | `test_rank_formulas_trail…` |
| Was kostet jede Methode? | Urteile bei 5 Kriterien: direkt 5, Swing 5, Best-Worst 7 (2m − 3), Rangformeln eine Rangordnung (mindestens 7 Vergleiche); die Paarvergleichsmatrix 10. | `test_judgment_counts…` |
| Wenn ein Kriterium kaum streut (Welt Rohwerte, 5 Kriterien; δ = 1 / 0,5 / 0,2 / 0,05) | **wahre Gewichte 96,5 / 86,5 / 76,0 / 67,0 %; direkt 90,5 / 82,0 / 73,0 / 64,5 %; Swing 95,0 / 93,0 / 90,5 / 91,5 %**; ROC 81,5 / 74,5 / 68,0 / 60,5 %; gleich 56,0 / 53,5 / 54,0 / 47,5 %. | `test_one_narrow_criterion` |
| Dasselbe in der Welt Güte (δ = 1 / 0,05) | wahre Gewichte 67,0 / 50,0 %, direkt 66,5 / 48,5 %, Swing 66,0 / 69,0 % (die Güte ist bei den logarithmischen Kriterien nicht linear in den Rohwerten, daher bleibt Swing unter 100 %). | `test_one_narrow_criterion` |
| Rangformeln gegen die Verteilung der Gewichte (5 Kriterien, exakte Rangfolge, Abstand TV zu w) | α = 0,5 / 1 / 5: gleich 0,434 / 0,329 / 0,157; Rangsumme 0,266 / 0,166 / **0,088**; Rangreziprok 0,216 / 0,149 / 0,138; ROC **0,184 / 0,128** / 0,172. Bei fast gleichen Gewichten ist ROC schlechter als gleiche Gewichte und als die Rangsumme. | `test_rank_formulas_against_the_weight_distribution` |
| Ist ROC der Erwartungswert der geordneten Gewichte? | Ja, bei Gleichverteilung über dem Simplex: größte Abweichung Formel gegen Simulation 0,001 (m = 3, 5, 8; 10 000 Ziehungen). | `test_roc_formula_matches…` |
| Rauschen σ (Welt Rohwerte, 5 Kriterien; σ = 0 / 0,4 / 0,8) | direkt 96,5 / 81,0 / 69,0 %; Swing 98,5 / 86,5 / 77,0 %; Best-Worst 93,5 / 84,0 / 79,0 %; ROC 82,0 / 76,5 / 71,0 %. Jede Methode verliert mehr als zehn Punkte bis σ = 0,8. | `test_noise` |
| Wie oft schneidet die Skala 1 bis 9 ab? (gleichverteilte Gewichte, 3 / 5 / 8 Kriterien) | 22 / 25 / 28 % der Verhältnisse zum Besten sind größer als 9; in 39 / 66 / 87 % der Fälle mindestens ein Urteil. Bei α = 0,5: 43 bis 49 % und 67 / 90 / 98,5 %. | `test_best_worst_clipping` |

## Ehrliche Grenzen

- **Das Urteilsmodell ist angenommen.** Lognormales Rauschen mit gleichem σ für alle Methoden, Rundung auf ganze Punkte, Bereichsunempfindlichkeit der direkten Angabe nach der
  Literatur (Fischer 1995, nur die Zusammenfassung gelesen), nicht nach eigener Messung. Echte Menschen urteilen anders: Anker, runde Zahlen, systematische Verzerrungen.
- **Kleine Unterschiede sind Rauschen.** Direkt, Swing, Best-Worst und die wahren Gewichte liegen bei gleichmäßigen Kriterien innerhalb von zwei Standardfehlern.
- **Gleichverteilte Gewichte begünstigen ROC.** Das Experiment 3 zeigt die Gegenrichtung, aber nur für drei Werte von α.
- **Eine Verschmälerung, ein Generator.** Wie sich mehrere schmale Kriterien verhalten, ist nicht gemessen; n = 20, 3 bis 5 Kriterien.
- **Keine Konsistenz, kein Interview.** Die Zahl der Urteile ist ein Aufwandsmaß, kein Maß für Mühe oder Fehleranfälligkeit.
- **Eine Normierung (Min-Max).** Mit anderen Normierungen verschieben sich die Zahlen; das steht im Vorgänger.

## Verifikation

- **Kern:** Rangformeln, Swing und Best-Worst gegen Handrechnung auf der Mini-Instanz (`tests/conftest.py`: w = (0,6; 0,3; 0,1), ROC 0,6111 / 0,2778 / 0,1111 mit Abstand
  0,0222, Rangsumme 0,1, Rangreziprok 0,0818, Swing-Punkte 100 / 56 / 10, Best-Worst exakt w) und gegen Randfälle (Abschneiden bei 16-fachem Verhältnis, Gleichstände).
- **Orakel** (`tests/test_oracle_gewichte_erheben.py`): Rangformeln gegen Definitionsschleifen; ROC gegen den Erwartungswert der geordneten Gewichte bei Gleichverteilung und als
  kleinster erwarteter Abstand unter den Rangformeln (Gegenprobe: bei α = 5 ist ROC schlechter als die Rangsumme); die gezogenen Gewichte gegen die Dirichlet-Momente;
  Best-Worst-LP gegen 20 000 zufällige und nahe Kandidaten sowie ein Gitter; Swing mit voller Spannweite ∝ w; Swing-Gewichte treffen bei linearem Entscheider (Rohwerte) und
  schmaler Spannweite, wahre Gewichte nicht; das Urteilsrauschen gegen die Lognormalverteilung.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), jede Methode, alle Presets, Permalink, PDF und Experimente erst nach Klick, Footer,
  Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `gwe_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `gwe_elicit.py` | Urteilsmodell und Erhebungsmethoden (direkt, Swing, Rangformeln, Best-Worst) |
| `gwe_score.py` | Min-Max, Score, Wertskalen des Entscheiders, Verschmälerung, Kendalls τ-b |
| `gwe_evaluation.py` | Fall der App, vier Experimente über feste Matrizen, Verdikte |
| `gwe_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `gwe_visualization.py` | Tabellen und Plotly-Abbildungen (Achsen gesperrt) |
| `gwe_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; `scipy` für das Best-Worst-LP) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanz, Erhebung, Orakel, Generator, Auswertung, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Die Paarvergleichsmatrix und der Eigenvektor** (AHP): zehn Vergleiche bei fünf Kriterien, mit Konsistenzmaß; das Verfahren hat sein eigenes Stück der Linie.
- **Konsistenz menschlicher Urteile:** Die Simulation kann sie nicht prüfen.
- **Mehrere Entscheider:** der Entscheider ist hier ein einzelnes, verborgenes Orakel; die Gruppe steht in einem eigenen Stück.
- **Unsichere Gewichte als Bereich** (alle verträglichen Gewichte): ROC ist deren Schwerpunkt bei Rangvorgabe, die allgemeine Fassung steht beim SMAA.
- **Lernen aus Beispielurteilen über Alternativen:** dort werden Wertfunktionen abgeleitet, nicht Wichtigkeiten erfragt.
- **Andere Normierungen als Min-Max:** stehen im Vorgänger.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html).
