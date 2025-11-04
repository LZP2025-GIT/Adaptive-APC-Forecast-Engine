# Adaptive APC Forecast Engine (AAFE)
Dynamic Bayesian prediction framework for real-time anomalous phenomena analysis, built on the Unified Archetypal Bayesian Theory (UABT).

**Author:** Zachariah Paul Laing  
**License:** CC-BY-SA 4.0  
**Status:** v1.0 (initial public release)

## Overview
AAFE converts UABT’s static posterior into a live, adaptive forecasting engine:

Posterior_Odds(t+Δt) = Posterior_Odds(t) × Π(LR_i,t) × Π(ω_j,t) × M_F5(t) × [1 + α(dΔ/dt)]
P_posterior = Posterior_Odds / (1 + Posterior_Odds)

- `LR_i,t`: domain likelihood ratios (S₁–S₉)
- `ω_j,t`: correction filters (F₁–F₅)
- `M_F5(t)`: verification multiplier (institutional throughput)
- `α(dΔ/dt)`: adaptive drift term

## Quickstart
```bash
# 1) Create a venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run a synthetic streaming demo (writes CSV + PNG)
python src/aafe_demo.py --steps 2000 --seed 42
