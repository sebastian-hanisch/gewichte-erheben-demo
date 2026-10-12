"""Generator der Linie (wortgleich in allen Repos): Strom, Copula, Rohwerte, Entscheider."""

import numpy as np
import pytest

import entscheidungsmatrix_generator as gen

MASK = (1 << 64) - 1


def splitmix_reference(seed, count):
    """Unabhängige Python-Fassung des SplitMix64-Stroms (Steele, Lea, Flood 2014)."""
    state, out = seed & MASK, []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out.append(z ^ (z >> 31))
    return out


@pytest.mark.parametrize("seed", [0, 1, 35, 123456789, 2**63 + 5])
def test_splitmix_stream_matches_the_reference(seed):
    got = gen.SplitMix64(seed).u64(25).tolist()
    assert got == splitmix_reference(seed, 25)


def test_the_stream_does_not_depend_on_how_it_is_drawn():
    a = gen.SplitMix64(7)
    b = gen.SplitMix64(7)
    assert np.concatenate([a.uniform(3), a.uniform(4)]).tolist() == b.uniform(7).tolist()


def test_uniform_is_in_the_half_open_unit_interval():
    u = gen.SplitMix64(3).uniform(50000)
    assert u.min() >= 0.0 and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01


def test_normal_has_mean_zero_and_unit_variance():
    z = gen.SplitMix64(11).normal(60000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02


def test_normal_cdf_known_values():
    assert gen.normal_cdf(np.array([0.0]))[0] == pytest.approx(0.5)
    assert gen.normal_cdf(np.array([1.959963984540054]))[0] == pytest.approx(0.975, abs=1e-9)
    assert gen.normal_cdf(np.array([-1.0, 1.0])).sum() == pytest.approx(1.0)


def test_copula_marginals_are_uniform():
    u = gen.copula_uniforms(20000, 3, 0.4, gen.SplitMix64(5))
    for k in range(3):
        col = np.sort(u[:, k])
        ks = np.max(np.abs(col - (np.arange(1, len(col) + 1) - 0.5) / len(col)))
        assert ks < 0.012, ks


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.6, 0.9])
def test_copula_spearman_correlation_follows_the_gaussian_formula(rho):
    u = gen.copula_uniforms(40000, 2, rho, gen.SplitMix64(9))
    ranks = np.argsort(np.argsort(u, axis=0), axis=0)
    spearman = np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1]
    assert spearman == pytest.approx(6 / np.pi * np.arcsin(rho / 2), abs=0.02)


def test_equal_correlation_below_the_lower_bound_is_rejected():
    assert gen.rho_lower(2) == -1.0 and gen.rho_lower(5) == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="liegt nicht in"):
        gen.copula_uniforms(5, 5, -0.25, gen.SplitMix64(1))
    with pytest.raises(ValueError, match="liegt nicht in"):
        gen.copula_uniforms(5, 3, 1.0, gen.SplitMix64(1))


def test_matrix_is_deterministic_per_seed_and_differs_between_seeds():
    a = gen.make_matrix(10, 4, 0.2, gen.SplitMix64(35))
    b = gen.make_matrix(10, 4, 0.2, gen.SplitMix64(35))
    c = gen.make_matrix(10, 4, 0.2, gen.SplitMix64(36))
    assert (a.raw == b.raw).all() and (a.utility == b.utility).all()
    assert not (a.raw == c.raw).all()


def test_raw_values_follow_direction_and_stay_in_range():
    g = np.linspace(0.01, 0.99, 50)[:, None] * np.ones((1, 4))
    raw = gen.to_raw(g, 4)
    assert (np.diff(raw[:, 0]) >= 0).all() and (np.diff(raw[:, 1]) <= 0).all()      # Kriterium 1 ↑, Kriterium 2 ↓
    assert (np.diff(raw[:, 2]) >= 0).all() and (np.diff(raw[:, 3]) <= 0).all()
    for k in range(4):
        _, _, _, lo, hi, _ = gen.CRITERIA[k]
        assert raw[:, k].min() >= lo and raw[:, k].max() <= hi


def test_signed_matrix_turns_every_criterion_into_higher_is_better():
    x = gen.make_matrix(30, 4, 0.0, gen.SplitMix64(2))
    assert (x.directions == np.array([1, -1, 1, -1])).all()
    for k in range(4):
        order_g = np.argsort(x.goodness[:, k])
        # steigende Güte: signed steigt (nicht streng: die Rohwerte sind gerundet)
        assert (np.diff(x.signed[order_g, k]) >= 0).all()


def test_levels_give_integer_ratings_and_all_directions_up():
    x = gen.make_matrix(40, 3, 0.0, gen.SplitMix64(4), levels=3)
    assert set(np.unique(x.raw)) <= {1.0, 2.0, 3.0}
    assert (x.directions == 1).all() and (x.signed == x.raw).all() and x.units == ("Stufe",) * 3
    assert set(np.unique(x.goodness)) <= {1 / 6, 0.5, 5 / 6}


def test_decision_maker_weights_are_positive_and_sum_to_one():
    w = gen.decision_maker_weights(6, gen.SplitMix64(8))
    assert (w > 0).all() and w.sum() == pytest.approx(1.0)


def test_monotone_utility_forms_increase_in_every_criterion():
    w = np.array([0.2, 0.5, 0.3])
    base = np.array([[0.4, 0.4, 0.4]])
    for form in ("linear", "konkav", "cobb"):
        for k in range(3):
            up = base.copy()
            up[0, k] += 0.1
            assert gen.utility(up, w, form)[0] > gen.utility(base, w, form)[0], (form, k)


def test_target_form_is_not_monotone_in_the_first_criterion():
    w = np.array([0.5, 0.3, 0.2])
    g = np.array([[0.1, 0.5, 0.5], [0.5, 0.5, 0.5], [0.9, 0.5, 0.5]])
    u = gen.utility(g, w, "zielwert")
    assert u[1] > u[0] and u[1] > u[2]


def test_unknown_inputs_raise():
    with pytest.raises(ValueError):
        gen.make_matrix(5, 9, 0.0, gen.SplitMix64(1))
    with pytest.raises(ValueError):
        gen.make_matrix(5, 3, 0.0, gen.SplitMix64(1), form="quadrat")
    with pytest.raises(ValueError):
        gen.utility(np.zeros((2, 2)), np.ones(2), "quadrat")


def test_a_single_criterion_is_allowed_and_has_no_correlation_bound():
    assert gen.rho_lower(1) == -1.0 and gen.rho_lower(2) == -1.0
    x = gen.make_matrix(7, 1, 0.0, gen.SplitMix64(3))
    assert x.raw.shape == (7, 1) and x.names == ("Kriterium 1",) and x.units == ("Punkte",)


def test_quantize_covers_the_closed_unit_interval_and_all_levels():
    assert gen.quantize(np.array([0.0, 0.3333, 0.34, 0.6667, 0.99, 1.0]), 3).tolist() == [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    assert gen.quantize(np.array([1.0]), 5).tolist() == [5.0]


def test_units_and_names_have_one_entry_per_criterion():
    for m in (1, 3, gen.M_MAX):
        x = gen.make_matrix(4, m, 0.0, gen.SplitMix64(m))
        assert len(x.units) == m and len(x.names) == m
        assert x.units == tuple(gen.CRITERIA[k][2] for k in range(m))


def test_down_criteria_are_logarithmic_and_hit_both_bounds():
    raw = gen.to_raw(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]), 2)
    assert raw[0].tolist() == [0.0, 5000.0] and raw[1].tolist() == [100.0, 800.0]
    assert raw[2, 1] == pytest.approx(round(5000.0 * (800.0 / 5000.0) ** 0.5), abs=1.0)      # geometrische Mitte, nicht arithmetische
