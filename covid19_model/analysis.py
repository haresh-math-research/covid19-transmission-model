"""Analytical results and epidemic summary statistics."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .models import Params, Result


def next_generation_r0(beta: float, sigma: float, gamma: float) -> float:
    """R0 as the spectral radius of the next-generation matrix K = F V^-1
    for the infected subsystem (E, I) of the SEIR model (van den Driessche & Watmough, 2002)."""
    F = np.array([[0.0, beta], [0.0, 0.0]])          # new infections
    V = np.array([[sigma, 0.0], [-sigma, gamma]])    # transitions
    K = F @ np.linalg.inv(V)
    return float(np.max(np.abs(np.linalg.eigvals(K))))


def herd_immunity_threshold(r0: float) -> float:
    return max(0.0, 1.0 - 1.0 / r0)


def early_growth_rate(beta: float, sigma: float, gamma: float) -> float:
    """Malthusian growth rate r: positive root of (r + sigma)(r + gamma) = beta * sigma."""
    return 0.5 * (-(sigma + gamma) + np.sqrt((sigma - gamma) ** 2 + 4 * beta * sigma))


def doubling_time(r: float) -> float:
    return np.log(2) / r


def final_size(r0: float, s0: float = 1.0) -> float:
    """Fraction of the population ever infected, solving z = s0 (1 - exp(-R0 z)).

    s0 < 1 accounts for initial immunity (e.g. prior vaccination)."""
    if s0 * r0 <= 1.0:
        return 0.0
    f = lambda z: z - s0 * (1.0 - np.exp(-r0 * z))
    return brentq(f, 1e-12, s0)


def effective_r(res: Result, p: Params, beta_fn=None) -> np.ndarray:
    """R_t = beta(t)/gamma * S(t)/N."""
    beta_t = np.array([beta_fn(t) for t in res.t]) if beta_fn else p.beta
    return beta_t / p.gamma * res["S"] / p.N


def summarise(res: Result, p: Params) -> dict:
    I = res["I"]
    k = int(np.argmax(I))
    return {
        "R0": p.r0,
        "peak_infectious": float(I[k]),
        "peak_day": float(res.t[k]),
        "peak_prevalence_%": 100 * float(I[k]) / p.N,
        "attack_rate_%": 100 * float(res["C"][-1]) / p.N if "C" in res.names
                         else 100 * float(res["R"][-1]) / p.N,
        "deaths": float(res["D"][-1]) if "D" in res.names else float("nan"),
    }
