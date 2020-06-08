"""Reproduces every figure and number in the README.  Run:  python run_demo.py"""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from covid19_model import Params, simulate_sir, simulate_seird, intervention
from covid19_model import analysis, calibration, sensitivity, stochastic

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- 1. SIR vs SEIRD
p = Params(N=1_000_000, beta=0.5, sigma=1 / 5.2, gamma=1 / 7)
sir = simulate_sir(p, 250, i0=2)
seird = simulate_seird(p, 250, e0=1, i0=1)
print(f"R0 = {p.r0:.2f}  (next-generation matrix: {analysis.next_generation_r0(p.beta, p.sigma, p.gamma):.2f})")
print(f"Herd immunity threshold = {100*analysis.herd_immunity_threshold(p.r0):.1f}%")
r = analysis.early_growth_rate(p.beta, p.sigma, p.gamma)
print(f"Early growth rate r = {r:.3f}/day, doubling time {analysis.doubling_time(r):.1f} days "
      f"(SIR would predict r = {p.beta - p.gamma:.3f}/day)")
print(f"Final size (analytic) = {100*analysis.final_size(p.r0):.2f}%   "
      f"SEIRD simulated = {100*seird['C'][-1]/p.N:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(sir.t, sir["I"] / p.N * 100, label="SIR", lw=2)
ax[0].plot(seird.t, seird["I"] / p.N * 100, label="SEIRD (infectious)", lw=2)
ax[0].plot(seird.t, seird["E"] / p.N * 100, "--", label="SEIRD (exposed)", lw=1.5)
ax[0].set(xlabel="day", ylabel="% of population", title="Latent period delays and lowers the peak")
ax[0].legend()
for k in ("S", "E", "I", "R", "D"):
    ax[1].plot(seird.t, seird[k] / p.N * 100, label=k, lw=2)
ax[1].set(xlabel="day", ylabel="% of population", title="SEIRD compartments"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / "01_sir_vs_seird.png"); plt.close(fig)

# ------------------------------------------------- 2. Interventions & vaccination
scenarios = {
    "No intervention": (p, None),
    "Lockdown day 30 (-55%)": (p, intervention(p.beta, 0.55, 30)),
    "Lockdown day 30 + vaccination from day 90": (replace(p, vax_rate=0.01, vax_start=90),
                                                  intervention(p.beta, 0.55, 30)),
}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
print("\nScenario summary")
for name, (pp, bf) in scenarios.items():
    res = simulate_seird(pp, 400, e0=5, i0=5, beta_fn=bf)
    s = analysis.summarise(res, pp)
    print(f"  {name:45s} peak {s['peak_prevalence_%']:.2f}% on day {s['peak_day']:.0f}, "
          f"attack {s['attack_rate_%']:.1f}%, deaths {s['deaths']:.0f}")
    ax[0].plot(res.t, res["I"] / pp.N * 100, lw=2, label=name)
    ax[1].plot(res.t, analysis.effective_r(res, pp, bf), lw=2, label=name)
ax[1].axhline(1, color="k", ls=":")
ax[0].set(xlabel="day", ylabel="% infectious", title="Intervention scenarios"); ax[0].legend(fontsize=8)
ax[1].set(xlabel="day", ylabel="$R_t$", title="Effective reproduction number")
fig.tight_layout(); fig.savefig(OUT / "02_scenarios.png"); plt.close(fig)

# ------------------------------------------------ 3. Calibration on synthetic data
setup = calibration.Setup(Params(N=2_000_000, sigma=1 / 5.2, gamma=1 / 7), t_int=35, rho=0.25, k=20)
truth = (0.50, 0.55, 5.0)
T = 120
y = calibration.synthetic_data(truth, setup, T, seed=1)
theta, ll = calibration.fit(y, setup)
r0_grid, prof, (lo, hi) = calibration.profile_r0(theta, ll, y, setup)
print(f"\nCalibration (synthetic truth: beta={truth[0]}, effect={truth[1]}, E0={truth[2]})")
print(f"  MLE: beta={theta[0]:.3f}, effect={theta[1]:.3f}, E0={theta[2]:.1f}")
print(f"  R0 = {theta[0]/setup.p.gamma:.2f}, 95% profile-likelihood CI [{lo:.2f}, {hi:.2f}]  (truth {truth[0]/setup.p.gamma:.2f})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(np.arange(1, T + 1), y, color="0.75", label="synthetic observed cases")
ax[0].plot(np.arange(1, T + 1), calibration.expected_cases(theta, setup, T), "r", lw=2, label="fitted mean")
ax[0].axvline(setup.t_int, color="k", ls=":"); ax[0].set(xlabel="day", ylabel="reported cases/day", title="Fit to data")
ax[0].legend()
ax[1].plot(r0_grid, prof - ll, lw=2); ax[1].axhline(-1.92, color="r", ls="--", label="95% cutoff")
ax[1].axvline(truth[0] / setup.p.gamma, color="g", ls=":", label="true $R_0$")
ax[1].set(xlabel="$R_0$", ylabel="profile log-likelihood (rel. to max)", title="Profile likelihood for $R_0$")
ax[1].legend(); fig.tight_layout(); fig.savefig(OUT / "03_calibration.png"); plt.close(fig)

# ------------------------------------------------------- 4. Sensitivity (PRCC)
names, X, outs = sensitivity.run_lhs(p, n=400, seed=0)
fig, ax = plt.subplots(figsize=(7, 4)); w = 0.25
print("\nPRCC (rows: parameter; cols: peak prevalence, attack rate, deaths)")
table = {k: sensitivity.prcc(X, v) for k, v in outs.items()}
for j, n in enumerate(names):
    print(f"  {n:6s}", "  ".join(f"{table[k][j]:+.2f}" for k in outs))
for i, (k, v) in enumerate(table.items()):
    ax.bar(np.arange(len(names)) + i * w, v, w, label=k)
ax.set_xticks(np.arange(len(names)) + w); ax.set_xticklabels(names)
ax.axhline(0, color="k", lw=0.5); ax.set(ylabel="PRCC", title="Global sensitivity (Latin hypercube, n=400)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "04_sensitivity.png"); plt.close(fig)

# --------------------------------------------------- 5. Stochastic vs deterministic
ps = Params(N=50_000, beta=0.35, sigma=1 / 5.2, gamma=1 / 7)
traj, total = stochastic.simulate(ps, t_max=300, i0=1, runs=3000, seed=3)
minor = stochastic.minor_outbreak_fraction(total, ps.N)
q = stochastic.extinction_probability(ps)
print(f"\nStochastic (N={ps.N:.0f}, R0={ps.r0:.1f}, discrete R0={stochastic.discrete_r0(ps):.2f}, 1 index case)")
print(f"  P(minor outbreak): simulated {minor:.3f}, branching-process theory {q:.3f}, naive 1/R0 = {1/ps.r0:.3f}")
# A daily chain-binomial has geometric stage durations, so the matching ODE uses per-step
# probabilities 1-exp(-rate) as its rates (giving R0 = beta / (1 - e^-gamma)).
ps_ode = replace(ps, sigma=1 - np.exp(-ps.sigma), gamma=1 - np.exp(-ps.gamma))
det = simulate_seird(ps_ode, 300, e0=0, i0=1)
fs = analysis.final_size(ps_ode.r0)
print(f"  Major-outbreak attack rate: simulated mean {100*np.mean(total[total >= 0.005*ps.N])/ps.N:.2f}%, "
      f"final-size equation with matched R0 {100*fs:.2f}%")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
major = total >= 0.005 * ps.N
ax[0].plot(traj[:, ~major][:, :100], color="tab:red", alpha=0.3, lw=0.8)
ax[0].plot(traj[:, major][:, :80], color="tab:blue", alpha=0.15, lw=0.8)
ax[0].plot(det.t, det["I"], "k", lw=2, label="matched ODE")
ax[0].set(xlabel="day", ylabel="infectious", title=f"Stochastic runs (red: fizzled, {100*minor:.0f}%)"); ax[0].legend()
ax[1].hist(total[major] / ps.N * 100, bins=40, color="tab:blue", alpha=0.7)
ax[1].axvline(100 * fs, color="k", ls="--", label="final-size equation (matched $R_0$)")
ax[1].set(xlabel="final attack rate (%) among major outbreaks", ylabel="runs", title="Final size distribution"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / "05_stochastic.png"); plt.close(fig)
print("\nFigures written to", OUT.resolve())
