from dataclasses import replace

import numpy as np
import pytest

from covid19_model import Params, simulate_seird, simulate_sir, intervention, analysis, stochastic, sensitivity

P = Params(N=1_000_000, beta=0.5, sigma=1 / 5.2, gamma=1 / 7)


def test_population_is_conserved():
    res = simulate_seird(P, 300, beta_fn=intervention(P.beta, 0.5, 30))
    total = sum(res[k] for k in ("S", "E", "I", "R", "D"))
    assert np.allclose(total, P.N, rtol=1e-6)


def test_sir_conserves_population():
    res = simulate_sir(P, 200)
    assert np.allclose(res["S"] + res["I"] + res["R"], P.N, rtol=1e-6)


def test_r0_next_generation_matrix_matches_closed_form():
    assert analysis.next_generation_r0(P.beta, P.sigma, P.gamma) == pytest.approx(P.r0)


def test_final_size_equation_matches_simulation():
    res = simulate_seird(P, 1000, e0=1, i0=1)
    assert res["C"][-1] / P.N == pytest.approx(analysis.final_size(P.r0), abs=1e-3)


def test_no_epidemic_below_threshold():
    assert analysis.final_size(0.9) == 0.0
    p = replace(P, beta=0.1)
    res = simulate_seird(p, 300, e0=10, i0=10)
    assert res["C"][-1] / p.N < 1e-3


def test_early_growth_rate_matches_simulation():
    p = replace(P, N=1e10)                     # keep S ~ N so the linear regime lasts
    res = simulate_seird(p, 80, e0=1, i0=1)
    inc = res.daily_incidence()
    slope = np.polyfit(np.arange(40, 70), np.log(inc[40:70]), 1)[0]
    assert slope == pytest.approx(analysis.early_growth_rate(p.beta, p.sigma, p.gamma), abs=2e-3)


def test_vaccination_reduces_attack_rate():
    base = simulate_seird(P, 400)["C"][-1]
    vax = simulate_seird(replace(P, vax_rate=0.01, vax_start=20), 400)["C"][-1]
    assert vax < 0.5 * base


def test_ifr_only_affects_deaths_not_transmission():
    a = simulate_seird(replace(P, ifr=0.002), 300)
    b = simulate_seird(replace(P, ifr=0.02), 300)
    assert np.allclose(a["C"], b["C"], rtol=1e-6)
    assert b["D"][-1] > a["D"][-1]


def test_extinction_probability_matches_branching_theory():
    ps = Params(N=50_000, beta=0.35, sigma=1 / 5.2, gamma=1 / 7)
    _, total = stochastic.simulate(ps, t_max=300, i0=1, runs=2000, seed=11)
    sim = stochastic.minor_outbreak_fraction(total, ps.N)
    assert sim == pytest.approx(stochastic.extinction_probability(ps), abs=0.04)


def test_prcc_recovers_monotone_relationships():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(500, 3))
    y = np.exp(3 * X[:, 0]) - 2 * X[:, 1] + 0.05 * rng.normal(size=500)
    r = sensitivity.prcc(X, y)
    assert r[0] > 0.95 and r[1] < -0.8 and abs(r[2]) < 0.15
