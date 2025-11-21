#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="PPO training plots from CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV with metrics per iteration.")
    parser.add_argument("--out", default=".", help="Output folder for PNGs.")
    parser.add_argument("--show", action="store_true", help="Show plot windows.")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Read CSV ---
    df = pd.read_csv(args.csv)

    # Ensure numeric types
    numeric_cols = [
        "iteration", "timesteps_total", "episodes_total", "episodes_this_iter",
        "episode_reward_min", "episode_reward_max", "episode_reward_mean",
        "episode_len_mean",
        "policy_a_reward_mean", "policy_a_reward_min", "policy_a_reward_max",
        "policy_p_reward_mean", "policy_p_reward_min", "policy_p_reward_max",
        "kl_a", "entropy_a",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # X-axis: iteration
    if "iteration" in df.columns and df["iteration"].notna().any():
        x = df["iteration"]
    else:
        x = pd.Series(range(1, len(df) + 1), name="iteration")

    # ===== 1) Policy reward mean vs iteration (a and p) =====
    fig1 = plt.figure(figsize=(8, 4.5))
    # a
    if "policy_a_reward_mean" in df.columns:
        plt.plot(x, df["policy_a_reward_mean"], marker="o", label="policy_a_reward_mean")
    # p
    if "policy_p_reward_mean" in df.columns:
        plt.plot(x, df["policy_p_reward_mean"], marker="s", label="policy_p_reward_mean")

    plt.title("Policy reward mean vs iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Mean reward (per policy)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig1_path = outdir / "policy_rewards_mean_vs_iter.png"
    plt.savefig(fig1_path, dpi=200)

    # ===== 1.b) Policy *a* reward mean vs iteration (only a) =====
    if "policy_a_reward_mean" in df.columns:
        fig1b = plt.figure(figsize=(8, 4.5))
        plt.plot(x, df["policy_a_reward_mean"], marker="o", label="policy_a_reward_mean")
        # plt.title("Policy a - mean reward vs iteration")
        plt.title("Us Federal - Policy a - mean reward vs iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Mean reward (policy a)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        fig1b_path = outdir / "policy_a_reward_mean_vs_iter.png"
        plt.savefig(fig1b_path, dpi=200)

    # ===== 2) KL vs iteration (a) =====
    if "kl_a" in df.columns:
        fig2 = plt.figure(figsize=(8, 4.5))
        plt.plot(x, df["kl_a"], marker="o", label="KL (a)")
        plt.title("KL (policy a) vs iteration")
        plt.xlabel("Iteration")
        plt.ylabel("KL")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        fig2_path = outdir / "kl_a_vs_iter.png"
        plt.savefig(fig2_path, dpi=200)

    # ===== 3) Entropy of a vs iteration =====
    if "entropy_a" in df.columns:
        fig3 = plt.figure(figsize=(8, 4.5))
        plt.plot(x, df["entropy_a"], marker="o", label="Entropy (a)")
        plt.title("Entropy (policy a) vs iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        fig3_path = outdir / "entropy_a_vs_iter.png"
        plt.savefig(fig3_path, dpi=200)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
