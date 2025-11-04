import time
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import yaml

from .utils import (
    odds_from_prob, prob_from_odds, product, clamp, verification_multiplier
)

@dataclass
class AAFEConfig:
    prior_probability: float
    likelihoods: Dict[str, float]
    filters: Dict[str, float]
    alpha: float
    drift_floor: float
    drift_ceiling: float
    flag_threshold: float

class AAFEEngine:
    """
    Minimal, transparent implementation of the Adaptive APC Forecast Engine.
    Implements the UABT × AAFE update:
      Posterior_Odds(t+Δt) =
        Posterior_Odds(t) × Π(LR_i,t) × Π(ω_j,t) × M_F5(t) × [1 + α * dΔ/dt]
    """

    def __init__(self, cfg: AAFEConfig):
        self.cfg = cfg
        self.posterior_odds = odds_from_prob(cfg.prior_probability)
        self.prev_posterior = cfg.prior_probability
        self.drift = 0.0
        self.t = 0

    @classmethod
    def from_yaml(cls, path: str) -> "AAFEEngine":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        cfg = AAFEConfig(
            prior_probability=float(raw["prior_probability"]),
            likelihoods={k: float(v) for k, v in raw["likelihoods"].items()},
            filters={k: float(v) for k, v in raw["filters"].items()},
            alpha=float(raw["alpha"]),
            drift_floor=float(raw["drift_floor"]),
            drift_ceiling=float(raw["drift_ceiling"]),
            flag_threshold=float(raw["flag_threshold"]),
        )
        return cls(cfg)

    def tick(self, domain_LRs: Dict[str, float], f5_verified_fraction: float) -> Dict[str, float]:
        # 1) Base multipliers
        LR_prod = product(domain_LRs.values()) if domain_LRs else 1.0
        filter_prod = product(self.cfg.filters.values())

        # 2) Institutional verification multiplier
        m_f5 = verification_multiplier(f5_verified_fraction)

        # 3) Adaptive drift update
        #   dΔ/dt = (P_t - P_{t-1}) / 1
        current_p = prob_from_odds(self.posterior_odds)
        d_delta_dt = (current_p - self.prev_posterior)
        self.drift = clamp(self.cfg.alpha * d_delta_dt, self.cfg.drift_floor, self.cfg.drift_ceiling)

        # 4) Update odds
        self.posterior_odds *= LR_prod * filter_prod * m_f5 * (1.0 + self.drift)
        new_p = prob_from_odds(self.posterior_odds)

        # 5) store & flag
        flagged = new_p >= self.cfg.flag_threshold
        self.prev_posterior = new_p
        self.t += 1

        return {
            "t": self.t,
            "LR_prod": LR_prod,
            "filter_prod": filter_prod,
            "M_F5": m_f5,
            "drift": self.drift,
            "posterior": new_p,
            "flagged": float(flagged),
        }
