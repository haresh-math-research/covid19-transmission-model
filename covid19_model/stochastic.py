"""Stochastic (chain-binomial) SEIR with daily time steps, vectorised across runs.

Why bother: the ODE model can never go extinct. With a handful of index cases a
real epidemic dies out by chance with probability > 0, and the ODEs miss this.
For the discrete-time model below the extinction probability has a closed form
from branching-process theory, which we use for validation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .models import Params


def simulate(p: Params, t_max: int = 250, i0: int = 1, e0: int = 0,
             runs: int = 2000, seed: int = 0):
    """Returns (I_traj [t_max+1, runs], total_infected [runs])."""
    rng = np.random.default_rng(seed)
    N = int(p.N)
    S = np.full(runs, N - i0 - e0, dtype=np.int64)
    E = np.full(runs, e0, dtype=np.int64)
    I = np.full(runs, i0, dtype=np.int64)
    p_inc, p_rec = 1 - np.exp(-p.sigma), 1 - np.exp(-p.gamma)
    traj = np.empty((t_max + 1, runs), dtype=np.int64)
    traj[0] = I
    for t in range(1, t_max + 1):
        lam = 1 - np.exp(-p.beta * I / N)          # per-susceptible infection probability
        new_E = rng.binomial(S, lam)
        new_I = rng.binomial(E, p_inc)
        new_R = rng.binomial(I, p_rec)
        S, E, I = S - new_E, E + new_E - new_I, I + new_I - new_R
        traj[t] = I
    return traj, N - S


def discrete_r0(p: Params) -> float:
    """R0 of the daily chain-binomial model: beta / (1 - exp(-gamma)).
    (A geometric infectious period has mean 1/(1-e^-gamma) steps, slightly above 1/gamma.)"""
    return p.beta / (1 - np.exp(-p.gamma))


def extinction_probability(p: Params) -> float:
    """P(minor outbreak) from one index case: smallest root of s = G(s), where
    G is the offspring pgf: Poisson(beta) infections per infectious day, geometric duration.
        G(s) = q e^{beta(s-1)} / (1 - (1-q) e^{beta(s-1)}),   q = 1 - e^{-gamma}
    """
    q = 1 - np.exp(-p.gamma)
    G = lambda s: q * np.exp(p.beta * (s - 1)) / (1 - (1 - q) * np.exp(p.beta * (s - 1)))
    if discrete_r0(p) <= 1:
        return 1.0
    return brentq(lambda s: G(s) - s, 0.0, 1 - 1e-9)


def minor_outbreak_fraction(total_infected: np.ndarray, N: float, cutoff: float = 0.005) -> float:
    return float(np.mean(total_infected < cutoff * N))
