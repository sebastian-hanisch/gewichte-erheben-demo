"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import gwe_constants as C
import gwe_evaluation as ev
import gwe_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["method"] in C.METHODS and p["scale"] in C.SCALES and p["sigma"] in C.SIGMA_CHOICES, name
        assert p["delta"] in C.DELTA_CHOICES and p["alpha"] in C.ALPHA_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.METHOD_LABELS) == set(C.METHODS) == set(C.METHOD_SHORT)
    assert set(C.SCALE_LABELS) == set(C.SCALES)
    assert set(C.METHODS) <= set(C.METHOD_COLORS) and "wahr" in C.METHOD_COLORS


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis("Standardfall (Voreinstellung)")
    assert a.chosen.hit and a.results["wahr"].hit and not a.results["gleich"].hit
    a = _analysis("Ein Kriterium kaum verschieden")
    assert a.results["swing"].hit and not a.results["direkt"].hit and not a.results["wahr"].hit
    a = _analysis("Fast gleiche Gewichte")
    assert a.results["gleich"].tv < a.results["roc"].tv                 # ROC liegt weiter von w entfernt als gleiche Gewichte
    assert a.results["gleich"].hit and not a.results["roc"].hit
    a = _analysis("Viel Rauschen")
    assert a.settings.sigma == 0.8 and not a.chosen.hit and _analysis("Viel Rauschen", sigma=0.0).chosen.hit
