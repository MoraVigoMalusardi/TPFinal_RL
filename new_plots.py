#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Compare policy a mean reward across multiple runs."
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="List of CSV paths (one per policy).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=False,
        help="Labels for each CSV (same order as --csvs).",
    )
    parser.add_argument(
        "--out",
        default=".",
        help="Output folder for the PNG.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot window.",
    )
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_paths = args.csvs
    labels = args.labels or [Path(p).stem for p in csv_paths]

    if len(labels) != len(csv_paths):
        raise ValueError("Number of labels must match number of CSVs.")

    plt.figure(figsize=(8, 4.5))

    for csv_path, label in zip(csv_paths, labels):
        df = pd.read_csv(csv_path)

        # Asegurar numérico
        for col in ["timesteps_total", "policy_a_reward_mean"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Eje X
        if "timesteps_total" in df.columns and df["timesteps_total"].notna().any():
            x = df["timesteps_total"]
        else:
            x = pd.Series(range(1, len(df) + 1), name="timesteps_total")

        if "policy_a_reward_mean" not in df.columns:
            print(f"Warning: 'policy_a_reward_mean' not in {csv_path}, skipping.")
            continue

        y = df["policy_a_reward_mean"]
        plt.plot(x, y, linewidth=1.5, label=label)

    plt.title("Policy a - mean reward vs timesteps")
    plt.xlabel("timesteps")
    plt.ylabel("Mean reward (policy a)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = outdir / "policy_a_reward_mean_comparison.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved comparison plot to: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
