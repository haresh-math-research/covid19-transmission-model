"""Deterministic compartmental models.

SIR    : S -> I -> R                                   (the original baseline)
SEIRD  : S -> E -> I -> {R, D}, plus vaccination (S -> R) and a time-varying
         transmission rate beta(t) to represent non-pharmaceutical interventions.

State order for SEIRD is (S, E, I, R, D, C) where C is *cumulative* incidence
(new infectious onsets), which is what surveillance data actually resembles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

BetaFn = Callable[[float], float]


@dataclass(frozen=True)
class Params:
    N: float = 1_000_000.0      # population size
    beta: float = 0.5           # transmission rate (per day)
    sigma: float = 1 / 5.2      # 1 / mean latent period (per day)
    gamma: float = 1 / 7.0      # 1 / mean infectious period (per day)
    ifr: float = 0.008          # infection fatality ratio
    vax_rate: float = 0.0       # per-susceptible daily vaccination rate
    vax_eff: float = 0.9        # all-or-nothing vaccine efficacy against infection
    vax_start: float = 0.0      # day vaccination begins

    @property
    def r0(self) -> float:
        """Basic reproduction number (same for SIR and SEIRD: beta / gamma)."""
        return self.beta / self.gamma


@dataclass
class Result:
    t: np.ndarray
    y: np.ndarray               # shape (n_states, n_times)
    names: tuple

    def __getitem__(self, key: str) -> np.ndarray:
        return self.y[self.names.index(key)]

    def daily_incidence(self) -> np.ndarray:
        """New infectious onsets per day (requires the cumulative state 'C')."""
        return np.diff(self["C"])


# --------------------------------------------------------------------------- SIR
def sir_rhs(t, y, p: Params):
    S, I, R = y
    infection = p.beta * S * I / p.N
    return [-infection, infection - p.gamma * I, p.gamma * I]


def simulate_sir(p: Params, t_max: float = 160, i0: float = 1.0) -> Result:
    t = np.arange(0, t_max + 1)
    sol = solve_ivp(sir_rhs, (0, t_max), [p.N - i0, i0, 0.0], args=(p,),
                    t_eval=t, method="LSODA", rtol=1e-8, atol=1e-8)
    return Result(sol.t, sol.y, ("S", "I", "R"))


# ------------------------------------------------------------------------- SEIRD
def seird_rhs(t, y, p: Params, beta_fn: Optional[BetaFn]):
    S, E, I, R, D, C = y
    beta_t = beta_fn(t) if beta_fn is not None else p.beta
    infection = beta_t * S * I / p.N
    nu = p.vax_rate * p.vax_eff if t >= p.vax_start else 0.0
    vaccinated = nu * S
    return [
        -infection - vaccinated,
        infection - p.sigma * E,
        p.sigma * E - p.gamma * I,
        (1 - p.ifr) * p.gamma * I + vaccinated,
        p.ifr * p.gamma * I,
        p.sigma * E,
    ]


def simulate_seird(p: Params, t_max: float = 300, e0: float = 1.0, i0: float = 1.0,
                   beta_fn: Optional[BetaFn] = None,
                   rtol: float = 1e-8, atol: float = 1e-8) -> Result:
    t = np.arange(0, t_max + 1)
    y0 = [p.N - e0 - i0, e0, i0, 0.0, 0.0, 0.0]
    sol = solve_ivp(seird_rhs, (0, t_max), y0, args=(p, beta_fn), t_eval=t,
                    method="LSODA", rtol=rtol, atol=atol)
    return Result(sol.t, sol.y, ("S", "E", "I", "R", "D", "C"))


def intervention(beta0: float, effect: float, t_start: float, ramp: float = 3.0) -> BetaFn:
    """Smooth (logistic) reduction of transmission by a fraction `effect` around t_start.

    beta(t) = beta0 * (1 - effect * sigmoid((t - t_start) / ramp))
    """
    def beta_fn(t: float) -> float:
        return beta0 * (1.0 - effect / (1.0 + np.exp(-(t - t_start) / ramp)))
    return beta_fn
