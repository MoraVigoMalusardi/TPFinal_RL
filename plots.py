#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Gráficos de training PPO (desde CSV).")
    parser.add_argument("--csv", required=True, help="Ruta al CSV con métricas por iteración.")
    parser.add_argument("--out", default=".", help="Carpeta de salida para los PNG.")
    parser.add_argument("--show", action="store_true", help="Mostrar ventanas de los gráficos.")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

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

    if "iteration" in df.columns and df["iteration"].notna().any():
        x = df["iteration"]
    else:
        x = pd.Series(range(1, len(df) + 1), name="iteration")

    fig1 = plt.figure(figsize=(8, 4.5))
    # a
    if "policy_a_reward_mean" in df.columns:
        plt.plot(x, df["policy_a_reward_mean"], marker="o", label="policy_a_reward_mean")
    # p
    if "policy_p_reward_mean" in df.columns:
        plt.plot(x, df["policy_p_reward_mean"], marker="s", label="policy_p_reward_mean")

    plt.title("Policy reward mean vs iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Reward medio (por política)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig1_path = outdir / "policy_rewards_mean_vs_iter.png"
    plt.savefig(fig1_path, dpi=200)

    if "kl_a" in df.columns:
        fig2 = plt.figure(figsize=(8, 4.5))
        plt.plot(x, df["kl_a"], marker="o", label="KL (a)")
        plt.title("KL (política a) vs iteración")
        plt.xlabel("Iteración")
        plt.ylabel("KL")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        fig2_path = outdir / "kl_a_vs_iter.png"
        plt.savefig(fig2_path, dpi=200)

    if "entropy_a" in df.columns:
        fig3 = plt.figure(figsize=(8, 4.5))
        plt.plot(x, df["entropy_a"], marker="o", label="Entropía (a)")
        plt.title("Entropía (política a) vs iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Entropía")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        fig3_path = outdir / "entropy_a_vs_iter.png"
        plt.savefig(fig3_path, dpi=200)

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
