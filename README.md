# COVID-19 Transmission Modelling (SIR → SEIRD)

Compartmental epidemic models in Python, with the analysis a methods section would actually need:
analytic results, calibration with uncertainty, global sensitivity analysis, and a stochastic model.

```
pip install -r requirements.txt
python run_demo.py          # prints all numbers below and writes figures/
pytest                      # 10 tests
```

## What's in the box

| Module | Contents |
|---|---|
| `models.py` | SIR; SEIRD with vaccination and a time-varying transmission rate β(t) for interventions; cumulative-incidence state for fitting to case data |
| `analysis.py` | R₀ via the next-generation matrix, early growth rate and doubling time, final-size equation, herd-immunity threshold, R_t |
| `calibration.py` | Negative-binomial maximum likelihood fit to daily cases, profile-likelihood 95% CI for R₀ |
| `sensitivity.py` | Latin hypercube sampling + partial rank correlation coefficients (PRCC) |
| `stochastic.py` | Vectorised chain-binomial SEIR; extinction probability from branching-process theory |

## Model

```
dS/dt = -β(t) S I / N - ν ε S
dE/dt =  β(t) S I / N - σ E
dI/dt =  σ E - γ I
dR/dt = (1 - IFR) γ I + ν ε S
dD/dt =  IFR · γ I
dC/dt =  σ E                     (cumulative incidence, what surveillance sees)
```

β(t) = β₀ (1 − effect · sigmoid((t − t₀)/ramp)), ν = vaccination rate, ε = vaccine efficacy.

## Results (from `run_demo.py`)

* **Why SEIR matters.** With R₀ = 3.5, σ = 1/5.2, γ = 1/7 the true early growth rate is 0.143/day
  (doubling 4.8 days); plain SIR with the same β would claim 0.357/day. Ignoring the latent period
  overstates early growth by 2.5× and overstates the peak.
* **Validation.** Simulated final size 96.60% equals the solution of z = 1 − e^(−R₀z); the
  next-generation-matrix R₀ equals β/γ; simulated growth rate matches the analytic root (tests).
* **Scenarios** (N = 1M): no intervention → peak 19.8% infectious, 7,728 deaths; a 55% transmission
  cut on day 30 → peak 4.4%, 5,030 deaths; plus vaccination from day 90 → 1,123 deaths.
* **Calibration** on synthetic data with known truth: R₀ recovered as 3.38, 95% CI [3.06, 3.74]
  (truth 3.5).
* **Sensitivity.** Peak size and attack rate are driven by β and γ (|PRCC| ≈ 0.95); IFR only drives
  deaths (PRCC 0.96) and has ≈ 0 influence on transmission (a built-in sanity check).
* **Stochastic.** With one index case at R₀ = 2.5, ~35% of outbreaks fizzle (simulated 0.349;
  branching theory 0.338). The naive 1/R₀ = 0.408 is wrong for this model because a daily
  chain-binomial has geometric stage durations, giving R₀ = β/(1 − e^−γ) = 2.63. Comparisons are
  made against an ODE with matched rates.

## Assumptions and limitations (say these in any write-up)

* Homogeneous mixing: no age structure or contact matrices.
* Reporting fraction ρ is fixed, not estimated: with N fixed it is confounded with E₀.
* σ and γ are fixed from literature values, not fitted.
* The calibration demo uses synthetic data so the truth is known. To use real case counts,
  pass your daily series as `y` to `calibration.fit` and set `Setup.t_int` to the intervention date.
* Interventions are a smooth exogenous β(t), not behaviour that responds to case counts.

## Natural next steps

Age-structured SEIR with a contact matrix; time-varying R_t estimation from data; MCMC (e.g. PyMC/emcee)
in place of profile likelihood; behavioural feedback into β; fitting to hospitalisation and death series jointly.

> The old `sir_model.py` is superseded by `covid19_model/models.py` and can be deleted.
