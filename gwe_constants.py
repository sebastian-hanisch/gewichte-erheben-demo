"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 5, 40, 20           # Score und Min-Max: O(n m); Best-Worst: ein LP mit m + 1 Variablen je Fall
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5             # Best-Worst braucht mindestens drei Kriterien
SEED_MAX, DEFAULT_SEED = 999999, 1

METHODS = ("gleich", "direkt", "swing", "rangsumme", "rangrezipr", "roc", "bwm")
METHOD_LABELS = {
    "gleich": "Gleiche Gewichte (keine Erhebung)",
    "direkt": "Direkte Wichtigkeit (Punkte 0 bis 100)",
    "swing": "Swing (Schwenk worst → best in der Stichprobe)",
    "rangsumme": "Rangsumme (nur die Rangfolge der Kriterien)",
    "rangrezipr": "Rangreziprok (nur die Rangfolge der Kriterien)",
    "roc": "ROC, Rangzentroid (nur die Rangfolge der Kriterien)",
    "bwm": "Best-Worst (Verhältnisse 1 bis 9, lineares LP)",
}
METHOD_SHORT = {"gleich": "gleich", "direkt": "direkt", "swing": "Swing", "rangsumme": "Rangsumme", "rangrezipr": "Rangreziprok", "roc": "ROC", "bwm": "Best-Worst"}
DEFAULT_METHOD = "swing"
RANK_METHODS = ("rangsumme", "rangrezipr", "roc")

SIGMA_CHOICES = (0.0, 0.1, 0.2, 0.4, 0.8)     # Streuung des Logarithmus jedes Verhältnisurteils
DEFAULT_SIGMA = 0.2
DELTA_CHOICES = (1.0, 0.5, 0.2, 0.05)         # Anteil der Streuung des ersten Kriteriums (1 = unverändert)
DEFAULT_DELTA = 1.0
ALPHA_CHOICES = (0.5, 1.0, 5.0)               # Verteilung der wahren Gewichte: Dirichlet(alpha); kleiner = ungleicher
DEFAULT_ALPHA = 1.0

SCALES = ("guete", "roh")
SCALE_LABELS = {
    "guete": "Linear in der Güte (Quantil innerhalb des Kriteriums)",
    "roh": "Linear in den Rohwerten (feste Nenn-Spannweite)",
}
DEFAULT_SCALE = "roh"

POINTS_MAX = 100                               # Punkteskala für direkte Wichtigkeit und Swing
BWM_MAX = 9                                    # Skala der Verhältnisurteile beim Best-Worst

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 20                                     # Experimente: feste Instanz mit 20 Alternativen, unabhängig vom Regler
EXP_M = (3, 5)
EXP_SIGMA = 0.2
EXP_NARROW_M = 5
EXP_NARROW_METHODS = ("gleich", "wahr", "direkt", "swing", "roc")
EXP_RANK_M = 5
EXP_RANK_DRAWS = 2000
EXP_NOISE_METHODS = ("direkt", "swing", "roc", "bwm")
EXP_CLIP_M = (3, 5, 8)
EXP_CLIP_DRAWS = 5000                          # Ziehungen je Zelle der Abschneide-Statistik (Standardfehler unter 1 Punkt)

COLOR_TRUE = "#D68A2E"
COLOR_FIT = "#8A96A6"
COLOR_ELICITED = "#2F6B65"
COLOR_LINE = "#4c78a8"
COLOR_WARN = "#c0392b"
METHOD_COLORS = {"gleich": "#8A96A6", "wahr": "#D68A2E", "direkt": "#4c78a8", "swing": "#2F6B65", "rangsumme": "#B8860B",
                 "rangrezipr": "#9b6b9b", "roc": "#c0392b", "bwm": "#5b7f95"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"n": DEFAULT_N, "m": DEFAULT_M, "method": DEFAULT_METHOD, "sigma": DEFAULT_SIGMA, "delta": 1.0, "scale": DEFAULT_SCALE,
                                      "alpha": 1.0, "seed": DEFAULT_SEED},
    "Ein Kriterium kaum verschieden": {"n": 20, "m": 5, "method": "direkt", "sigma": 0.2, "delta": 0.05, "scale": "roh", "alpha": 1.0, "seed": 3},
    "Fast gleiche Gewichte": {"n": 20, "m": 5, "method": "roc", "sigma": 0.0, "delta": 1.0, "scale": "guete", "alpha": 5.0, "seed": 24},
    "Viel Rauschen": {"n": 20, "m": 5, "method": "swing", "sigma": 0.8, "delta": 1.0, "scale": "roh", "alpha": 1.0, "seed": 5},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "20 Alternativen, 5 Kriterien, Swing-Erhebung mit mäßigem Rauschen; der Entscheider bewertet linear in den Rohwerten.",
    "Ein Kriterium kaum verschieden": "Die Alternativen unterscheiden sich im ersten Kriterium fast nicht: direkt angegebene Wichtigkeit und selbst die wahren Gewichte verfehlen den Besten, Swing nicht.",
    "Fast gleiche Gewichte": "Die wahren Gewichte liegen nahe beieinander (Dirichlet α = 5): ROC ist hier schlechter als gleiche Gewichte.",
    "Viel Rauschen": "Das Urteilsrauschen ist groß (σ = 0,8): jede Erhebung verliert, die Rangformeln am wenigsten.",
}
