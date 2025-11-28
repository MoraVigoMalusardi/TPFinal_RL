#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_series(csv_path: Path, y_col: str):
    """Carga un CSV y devuelve (x, y) con iteration y la columna pedida."""
    df = pd.read_csv(csv_path)

    # Asegurar numérico
    if "timesteps_total" in df.columns:
        df["timesteps_total"] = pd.to_numeric(df["timesteps_total"], errors="coerce")
    else:
        df["timesteps_total"] = pd.Series(range(1, len(df) + 1), name="timesteps_total")

    if y_col not in df.columns:
        raise ValueError(f"Columna '{y_col}' no encontrada en {csv_path}")

    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    # Filtrar NaNs iniciales por las dudas
    mask = df[y_col].notna()
    x = df.loc[mask, "timesteps_total"]
    y = df.loc[mask, y_col]

    return x, y


def plot_comparison(csv_paths, labels, y_col, title, ylabel, out_path: Path, show=False):
    """Plotea comparación LSTM vs no LSTM para una columna dada."""
    plt.figure(figsize=(8, 4.5))

    for csv_path, label in zip(csv_paths, labels):
        x, y = load_series(csv_path, y_col)
        plt.plot(x, y, linewidth=1.5, label=label)

    plt.title(title)
    plt.xlabel("timesteps")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Guardado gráfico en: {out_path}")

    if show:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compara sin LSTM vs con LSTM para Free Market (agentes) y AI-Economist (planner)."
    )
    parser.add_argument(
        "--base-dir",
        default="resultado",
        help="Directorio base donde están con_lstm/ y sin_lstm (por defecto: resultado).",
    )
    parser.add_argument(
        "--out",
        default="plots",
        help="Carpeta de salida para los PNG (por defecto: plots).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar las figuras en pantalla.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Free Market: reward de agentes (policy_a_reward_mean)
    # ------------------------------------------------------------------
    fm_sin = base / "sin_lstm" / "free_market" / "ppo_results_agents.csv"
    fm_con = base / "con_lstm" / "free_market" / "ppo_results_agents.csv"

    plot_comparison(
        csv_paths=[fm_sin, fm_con],
        labels=["Sin LSTM", "Con LSTM"],
        y_col="policy_a_reward_mean",
        title="Free Market – reward medio de agentes vs timesteps",
        ylabel="Reward medio (policy a)",
        out_path=outdir / "free_market_agents_lstm_vs_no_lstm.png",
        show=args.show,
    )

    # ------------------------------------------------------------------
    # 2) AI-Economist: reward del planner (policy_p_reward_mean)
    # ------------------------------------------------------------------
    pl_sin = base / "sin_lstm" / "ai_economist" / "ppo_results_with_planner.csv"
    pl_con = base / "con_lstm" / "ai_economist" / "ppo_results_with_planner.csv"

    plot_comparison(
        csv_paths=[pl_sin, pl_con],
        labels=["Sin LSTM", "Con LSTM"],
        y_col="policy_p_reward_mean",
        title="AI-Economist – reward medio del planner vs timesteps",
        ylabel="Reward medio (policy p)",
        out_path=outdir / "ai_economist_planner_lstm_vs_no_lstm.png",
        show=args.show,
    )


if __name__ == "__main__":
    main()
