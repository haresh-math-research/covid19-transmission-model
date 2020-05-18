"""Fit the SEIRD model to daily case counts by maximum likelihood, with profile-likelihood CIs.

Observation model:  y_t ~ NegBin(mean = rho * incidence_t, dispersion k)
  * rho (reporting fraction) is held fixed: with N also fixed it is confounded with E0,
    so it cannot be estimated from case counts alone. State this in any write-up.
  * sigma, gamma are fixed from the literature (latent / infectious periods).
Estimated: beta0, intervention effect, initial exposed E0 (with I0 = E0).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom

from .models import Params, simulate_seird, intervention


@dataclass(frozen=True)
class Setup:
    p: Params                 # fixed quantities (N, sigma, gamma)
    t_int: float              # intervention date (assumed known)
    ramp: float = 2.0
    rho: float = 0.25         # reporting fraction
    k: float = 20.0           # NB dispersion (larger -> closer to Poisson)


def expected_cases(theta, setup: Setup, T: int) -> np.ndarray:
    beta, effect, e0 = theta
    p = replace(setup.p, beta=beta)
    res = simulate_seird(p, T, e0=e0, i0=e0, beta_fn=intervention(beta, effect, setup.t_int, setup.ramp),
                         rtol=1e-6, atol=1e-6)
    return setup.rho * res.daily_incidence()


def synthetic_data(theta, setup: Setup, T: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = expected_cases(theta, setup, T)
    return rng.negative_binomial(setup.k, setup.k / (setup.k + mu))


def _to_theta(z):
    return np.array([np.exp(z[0]), 1 / (1 + np.exp(-z[1])), np.exp(z[2])])


def _from_theta(theta):
    b, eff, e0 = theta
    return np.array([np.log(b), np.log(eff / (1 - eff)), np.log(e0)])


def loglik(theta, y: np.ndarray, setup: Setup) -> float:
    mu = expected_cases(theta, setup, len(y)) + 1e-9
    return float(nbinom.logpmf(y, setup.k, setup.k / (setup.k + mu)).sum())


def fit(y: np.ndarray, setup: Setup, starts=None):
    """Multi-start Nelder-Mead in unconstrained space. Returns (theta_hat, loglik)."""
    starts = starts or [(0.35, 0.4, 3.0), (0.6, 0.7, 10.0), (0.45, 0.5, 1.0)]
    best = None
    for s in starts:
        r = minimize(lambda z: -loglik(_to_theta(z), y, setup), _from_theta(s),
                     method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-4, maxiter=800))
        if best is None or r.fun < best.fun:
            best = r
    return _to_theta(best.x), -best.fun


def profile_r0(theta_hat, ll_hat, y, setup: Setup, rel_range=0.25, n=21):
    """Profile likelihood for R0 = beta/gamma. Re-optimises (effect, E0) at each fixed beta.
    Returns (R0 grid, profile loglik, 95% CI) using the chi-square(1) cutoff 3.84/2."""
    gamma = setup.p.gamma
    grid = theta_hat[0] * np.linspace(1 - rel_range, 1 + rel_range, n)
    z0 = _from_theta(theta_hat)[1:]
    prof = []
    for b in grid:
        f = lambda z: -loglik(_to_theta(np.r_[np.log(b), z]), y, setup)
        r = minimize(f, z0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3, maxiter=300))
        prof.append(-r.fun)
    prof = np.array(prof)
    r0 = grid / gamma
    inside = prof >= ll_hat - 1.92
    lo = np.interp(ll_hat - 1.92, prof[: n // 2 + 1], r0[: n // 2 + 1]) if not inside[0] else np.nan
    hi = np.interp(ll_hat - 1.92, prof[n // 2:][::-1], r0[n // 2:][::-1]) if not inside[-1] else np.nan
    return r0, prof, (lo, hi)
