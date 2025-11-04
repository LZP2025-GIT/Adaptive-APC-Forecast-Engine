import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .aafe_engine import AAFEEngine
from .utils import clamp

def synth_event(domain_names, base_rate, spike_rate, spike_prob, rng):
    """
    Simulate a tick's domain likelihood ratios (LR_i,t) and an F5 verified fraction.
    Occasionally generate spikes to emulate cross-domain convergence.
    """
    is_spike = rng.random() < spike_prob
    LRs = {}
    for d in domain_names:
        if is_spike:
            LRs[d] = rng.uniform(1.05, 1.20)  # elevated LR during spike
        else:
            LRs[d] = rng.uniform(0.98, 1.05)  # noise-ish variations

    # F5 verified fraction: higher during spikes, lower otherwise
    f5 = rng.uniform(0.16, 0.22) if is_spike else rng.uniform(0.08, 0.12)
    return LRs, f5

def main(steps, seed):
    from .aafe_engine import AAFEConfig
    import yaml
    # load config
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, "aafe_config.yaml")
    engine = AAFEEngine.from_yaml(cfg_path)

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    domain_names = list(raw["likelihoods"].keys())
    rspec = raw["random_stream"]
    rng = random.Random(seed)

    rows = []
    for _ in range(steps):
        LRs, f5 = synth_event(
            domain_names=domain_names,
            base_rate=rspec["base_rate"],
            spike_rate=rspec["spike_rate"],
            spike_prob=rspec["spike_prob"],
            rng=rng
        )
        # fold in config default LRs as prior-ish multipliers
        for k, v in raw["likelihoods"].items():
            LRs[k] *= float(v)

        out = engine.tick(LRs, f5_verified_fraction=f5)
        rows.append({**out, **{f"LR_{k}": v for k, v in LRs.items()}, "F5": f5})

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = "outputs/aafe_stream.csv"
    df.to_csv(csv_path, index=False)

    # quick plot
    plt.figure(figsize=(10, 4))
    plt.plot(df["t"], df["posterior"], label="Posterior P(APC)")
    plt.axhline(engine.cfg.flag_threshold, linestyle="--", label="Flag Threshold")
    plt.title("AAFE Rolling Posterior")
    plt.xlabel("t (ticks)")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    png_path = "outputs/posterior_plot.png"
    plt.savefig(png_path, dpi=160)
    print(f"Wrote {csv_path} and {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    main(args.steps, args.seed)
