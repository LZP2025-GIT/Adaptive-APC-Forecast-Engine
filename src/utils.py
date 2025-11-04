import math
from typing import Dict

def odds_from_prob(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return p / (1 - p)

def prob_from_odds(o: float) -> float:
    return o / (1 + o)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def product(values):
    out = 1.0
    for v in values:
        out *= float(v)
    return out

def verification_multiplier(f5_fraction: float) -> float:
    """
    Convert a verified fraction into a conservative multiplier.
    Example: baseline 0.10 -> 1.0; 0.18 -> ~1.6 uplift.
    """
    baseline = 0.10
    if f5_fraction <= 0:
        return 1.0
    return max(0.5, min(3.0, f5_fraction / baseline))
