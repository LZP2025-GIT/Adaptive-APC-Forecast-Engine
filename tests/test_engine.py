import os
from src.aafe_engine import AAFEEngine

def test_engine_loads():
    here = os.path.dirname(__file__)
    cfg = os.path.join(here, "..", "src", "aafe_config.yaml")
    eng = AAFEEngine.from_yaml(cfg)
    assert 0 < eng.prev_posterior < 1

def test_update_monotonicity():
    here = os.path.dirname(__file__)
    cfg = os.path.join(here, "..", "src", "aafe_config.yaml")
    eng = AAFEEngine.from_yaml(cfg)
    # strong LRs should push probability upward
    LRs = {k: 1.2 for k in eng.cfg.likelihoods.keys()}
    out = eng.tick(LRs, f5_verified_fraction=0.20)
    assert out["posterior"] > 0.02  # base prior is 0.02 in config
