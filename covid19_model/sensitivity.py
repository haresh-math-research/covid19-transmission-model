"""Global sensitivity analysis: Latin hypercube sampling + partial rank correlation (PRCC)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.stats import qmc, rankdata

from .models import Params, simulate_seird

PRIORS = {                     # (low, high) uniform ranges
    "beta":  (0.30, 0.70),
    "sigma": (1 / 7, 1 / 3),
    "gamma": (1 / 10, 1 / 4),
    "ifr":   (0.002, 0.02),
}


def prcc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Partial rank correlation of each column of X with y, controlling for the other columns."""
    Xr = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    yr = rankdata(y)
    out = []
    for j in range(Xr.shape[1]):
        others = np.column_stack([np.ones(len(yr)), np.delete(Xr, j, axis=1)])
        rx = Xr[:, j] - others @ np.linalg.lstsq(others, Xr[:, j], rcond=None)[0]
        ry = yr - others @ np.linalg.lstsq(others, yr, rcond=None)[0]
        out.append(np.corrcoef(rx, ry)[0, 1])
    return np.array(out)


def run_lhs(base: Params, n: int = 400, t_max: int = 250, seed: int = 0):
    names = list(PRIORS)
    lo, hi = np.array([PRIORS[k] for k in names]).T
    X = qmc.scale(qmc.LatinHypercube(d=len(names), seed=seed).random(n), lo, hi)
    outs = {"peak_prevalence": [], "attack_rate": [], "deaths": []}
    for row in X:
        p = replace(base, **dict(zip(names, row)))
        res = simulate_seird(p, t_max, rtol=1e-6, atol=1e-6)
        outs["peak_prevalence"].append(res["I"].max() / p.N)
        outs["attack_rate"].append(res["C"][-1] / p.N)
        outs["deaths"].append(res["D"][-1])
    return names, X, {k: np.array(v) for k, v in outs.items()}
