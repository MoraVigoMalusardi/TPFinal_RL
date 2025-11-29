import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import ray
import torch
import pandas as pd

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

# === IMPORTS DE SCRIPTS DE ENTRENAMIENTO ===

# MLP
from train_planner_ppo import (
    build_env_config as build_env_config_mlp,
    create_env_for_inspection as create_env_for_inspection_mlp,
    build_trainer_config as build_trainer_config_mlp,
)
from tutorials.rllib.env_wrapper import RLlibEnvWrapper

# LSTM + MLP
from train_planner_ppo_lstm_fc import (
    build_env_config as build_env_config_lstm,
    create_env_for_inspection as create_env_for_inspection_lstm,
    build_trainer_config as build_trainer_config_lstm,
    CustomLSTMPostFC,
)

# CNN + MLP
from train_planner_ppo_cnn import (
    build_env_config as build_env_config_cnn,
    create_env_for_inspection as create_env_for_inspection_cnn,
    build_trainer_config as build_trainer_config_cnn,
    SafeEnvWrapper,
    AI_Economist_CNN_PyTorch,
)

# -------------------------------------------------------------------
# Métrica de Gini
# -------------------------------------------------------------------

def gini(array):
    """Calcula el coeficiente de Gini."""
    array = np.array(array, dtype=float)
    if len(array) == 0 or array.sum() == 0:
        return 0.0
    sorted_array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_array)) / (n * np.sum(sorted_array)) - (n + 1) / n


# -------------------------------------------------------------------
# Helpers para construir trainers para cada arquitectura
# -------------------------------------------------------------------

def build_ai_trainer_mlp(config_path, agents_ckpt, planner_ckpt):
    """
    Construye trainer + env para la arquitectura MLP (train_planner_ppo.py)
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    env_config = build_env_config_mlp(cfg)
    env_obj = create_env_for_inspection_mlp(env_config)
    trainer_config = build_trainer_config_mlp(env_obj, cfg, env_config)

    trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)

    agent_weights = torch.load(agents_ckpt, map_location='cpu')
    trainer.get_policy("a").model.load_state_dict(agent_weights)

    planner_weights = torch.load(planner_ckpt, map_location='cpu')
    trainer.get_policy("p").model.load_state_dict(planner_weights)

    return trainer, env_obj


def build_ai_trainer_lstm(config_path, agents_ckpt, planner_ckpt):
    """
    Construye trainer + env para la arquitectura LSTM+MLP (train_planner_ppo_lstm_fc.py)
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    env_config = build_env_config_lstm(cfg)
    env_obj = create_env_for_inspection_lstm(env_config)
    trainer_config = build_trainer_config_lstm(env_obj, cfg, env_config)

    # Registrar modelo custom LSTM (igual que en el main de train_planner_ppo_lstm_fc)
    ModelCatalog.register_custom_model("lstm_post_fc_256", CustomLSTMPostFC)

    trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)

    agent_weights = torch.load(agents_ckpt, map_location='cpu')
    trainer.get_policy("a").model.load_state_dict(agent_weights)

    planner_weights = torch.load(planner_ckpt, map_location='cpu')
    trainer.get_policy("p").model.load_state_dict(planner_weights)

    return trainer, env_obj


def build_ai_trainer_cnn(config_path, agents_ckpt, planner_ckpt):
    """
    Construye trainer + env para la arquitectura CNN+MLP (train_planner_ppo_cnn.py)
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    env_config = build_env_config_cnn(cfg)
    env_obj = create_env_for_inspection_cnn(env_config)
    trainer_config = build_trainer_config_cnn(env_obj, cfg, env_config)

    # Registrar modelo custom CNN (igual que en train_planner_ppo_cnn)
    ModelCatalog.register_custom_model("paper_cnn_torch", AI_Economist_CNN_PyTorch)

    trainer = PPOTrainer(env=SafeEnvWrapper, config=trainer_config)

    agent_weights = torch.load(agents_ckpt, map_location='cpu')
    trainer.get_policy("a").model.load_state_dict(agent_weights)

    planner_weights = torch.load(planner_ckpt, map_location='cpu')
    trainer.get_policy("p").model.load_state_dict(planner_weights)

    return trainer, env_obj


# -------------------------------------------------------------------
# Evaluación genérica sobre un trainer + env existente
# -------------------------------------------------------------------

def get_base_env(env_obj):
    """
    Devuelve el entorno base que tiene .world.agents.
    - Para MLP/LSTM: env_obj es RLlibEnvWrapper -> usar env_obj.env
    - Para CNN: env_obj es SafeEnvWrapper -> usar env_obj.internal_env.env
    """
    if hasattr(env_obj, "env"):
        return env_obj.env
    if hasattr(env_obj, "internal_env") and hasattr(env_obj.internal_env, "env"):
        return env_obj.internal_env.env
    raise AttributeError("No pude encontrar atributo .env ni .internal_env.env en env_obj")

def evaluate_trainer_on_env(
    trainer,
    env_obj,
    policy_name,
    max_steps=1000,
    n_episodes=5,
    use_lstm=False,
    lstm_cell_size=128,
):
    """
    Evalúa un trainer dado en un env dado (ya construidos).
    Soporta tanto arquitecturas feed-forward como LSTM.

    Returns: dict con métricas agregadas.
    """
    print(f"\n{'='*70}")
    print(f"Evaluando: {policy_name}")
    print(f"{'='*70}")

    # --- NUEVO: obtenemos el env base (AI-Economist) ---
    base_env = get_base_env(env_obj)

    all_productivity = []
    all_equality = []
    all_gini = []

    for episode in range(n_episodes):
        obs = env_obj.reset()
        done = {"__all__": False}
        step = 0

        # Estados LSTM por agente+policy (solo si use_lstm=True)
        lstm_states = {}
        if use_lstm:
            for agent_id in obs.keys():
                policy_id = "a" if str(agent_id).isdigit() else "p"
                init_state = trainer.get_policy(policy_id).get_initial_state()
                lstm_states[(agent_id, policy_id)] = init_state

        # --- CAMBIO: usamos base_env en vez de env_obj.env ---
        initial_coins = [agent.total_endowment('Coin') for agent in base_env.world.agents]
        print(f"\n  Initial coins: {[f'{c:.1f}' for c in initial_coins]}")

        while not done["__all__"] and step < max_steps:
            actions = {}

            for agent_id, ob in obs.items():
                policy_id = "a" if str(agent_id).isdigit() else "p"

                if use_lstm:
                    key = (agent_id, policy_id)
                    state_in = lstm_states[key]
                    # full_fetch=True para obtener nuevo estado
                    action, state_out, _ = trainer.compute_action(
                        ob,
                        state=state_in,
                        policy_id=policy_id,
                        full_fetch=True,
                    )
                    lstm_states[key] = state_out
                else:
                    action = trainer.compute_action(ob, policy_id=policy_id)

                actions[agent_id] = action

                if step % 100 == 0:
                    print(f"    Agent {agent_id} took action {action}")

            obs, rew, done, info = env_obj.step(actions)

            if step % 100 == 0:
                print(f"    Step {step+1} done with rewards {rew}")

                # --- CAMBIO: también acá usamos base_env ---
                current_coins = [agent.total_endowment('Coin') for agent in base_env.world.agents]
                print(f"    Current coins: {[f'{c:.1f}' for c in current_coins]}")

                coin_changes = [curr - init for curr, init in zip(current_coins, initial_coins)]
                print(f"    Total change: {[f'{c:+.1f}' for c in coin_changes]}")

            step += 1

        # Métricas del episodio (otra vez base_env)
        final_coins = np.array([
            agent.total_endowment('Coin')
            for agent in base_env.world.agents
        ])

        productivity = final_coins.sum()
        gini_coeff = gini(final_coins)
        n = len(final_coins)
        equality = 1 - (n / (n - 1)) * gini_coeff if n > 1 else 1.0

        all_productivity.append(productivity)
        all_equality.append(equality)
        all_gini.append(gini_coeff)

        print(f"  Episode {episode+1}/{n_episodes}: "
              f"Prod={productivity:.1f}, Eq={equality:.3f}, Gini={gini_coeff:.3f}")

    results = {
        'policy_name': policy_name,
        'productivity': float(np.mean(all_productivity)),
        'productivity_std': float(np.std(all_productivity)),
        'equality': float(np.mean(all_equality)),
        'equality_std': float(np.std(all_equality)),
        'gini': float(np.mean(all_gini)),
        'gini_std': float(np.std(all_gini)),
        'eq_times_prod': float(np.mean(all_equality) * np.mean(all_productivity)),
    }

    print(f"\n  Resultados promediados para {policy_name}:")
    print(f"  Productivity: {results['productivity']:.1f} ± {results['productivity_std']:.1f}")
    print(f"  Equality: {results['equality']:.3f} ± {results['equality_std']:.3f}")
    print(f"  Gini: {results['gini']:.3f} ± {results['gini_std']:.3f}")
    print(f"  Eq × Prod: {results['eq_times_prod']:.1f}")

    return results


# -------------------------------------------------------------------
# Comparación de arquitecturas AI-Economist
# -------------------------------------------------------------------

def compare_ai_architectures():
    """
    Compara las 3 arquitecturas de AI Economist:
    - MLP
    - LSTM + MLP
    - CNN + MLP
    y genera un gráfico y un CSV.
    """
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # ===== MLP =====
    print("\n Cargando AI Economist - MLP...")
    mlp_config_path = "phase_aiecon/config.yaml"
    mlp_agents_ckpt = "checkpoints/nuevo_sin_lstm/policy_a_weights_w_planner.pt"
    mlp_planner_ckpt = "checkpoints/nuevo_sin_lstm/policy_p_weights_w_planner.pt"

    mlp_trainer, mlp_env = build_ai_trainer_mlp(
        mlp_config_path, mlp_agents_ckpt, mlp_planner_ckpt
    )

    results_mlp = evaluate_trainer_on_env(
        mlp_trainer,
        mlp_env,
        policy_name="MLP",
        n_episodes=10,
        use_lstm=False,
    )
    mlp_trainer.stop()

    # ===== LSTM + MLP =====
    print("\n Cargando AI Economist - LSTM+MLP...")
    lstm_config_path = "phase_aiecon_lstm/config.yaml"
    lstm_agents_ckpt = "checkpoints/nuevo_con_lstm_fc/policy_a_weights_w_planner.pt"
    lstm_planner_ckpt = "checkpoints/nuevo_con_lstm_fc/policy_p_weights_w_planner.pt"

    lstm_trainer, lstm_env = build_ai_trainer_lstm(
        lstm_config_path, lstm_agents_ckpt, lstm_planner_ckpt
    )

    results_lstm = evaluate_trainer_on_env(
        lstm_trainer,
        lstm_env,
        policy_name="LSTM+MLP",
        n_episodes=10,
        use_lstm=True,
        lstm_cell_size=128,  
    )
    lstm_trainer.stop()

    # ===== CNN + MLP =====
    print("\n Cargando AI Economist - CNN+MLP...")
    cnn_config_path = "phase_aiecon_cnn/config.yaml"
    cnn_agents_ckpt = "checkpoints/nuevo_cnn_planner/policy_a_cnn_weights_w_planner.pt"
    cnn_planner_ckpt = "checkpoints/nuevo_cnn_planner/policy_p_cnn_weights_w_planner.pt"

    cnn_trainer, cnn_env = build_ai_trainer_cnn(
        cnn_config_path, cnn_agents_ckpt, cnn_planner_ckpt
    )

    results_cnn = evaluate_trainer_on_env(
        cnn_trainer,
        cnn_env,
        policy_name="CNN+MLP",
        n_episodes=10,
        use_lstm=False,
    )
    cnn_trainer.stop()

    ray.shutdown()

    all_results = [results_mlp, results_lstm, results_cnn]
    plot_architectures_comparison(all_results)

    df = pd.DataFrame(all_results)
    df.to_csv('ai_architectures_comparison_results.csv', index=False)
    print("\n Resultados arquitecturas guardados en: ai_architectures_comparison_results.csv")

    return all_results



# -------------------------------------------------------------------
# Plot para arquitecturas
# -------------------------------------------------------------------

def plot_architectures_comparison(results_list, save_path='ai_econ_architectures.png'):
    """
    Gráfico de barras comparando arquitecturas de AI-Economist:
    MLP vs LSTM+MLP vs CNN+MLP.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    policies = [r['policy_name'] for r in results_list]
    colors = ['#4EA8DE', '#C77DFF', '#06A77D']  # Blue, Purple, Teal
    
    # 1) Productivity
    ax = axes[0]
    productivity_vals = [r['productivity'] for r in results_list]
    productivity_stds = [r['productivity_std'] for r in results_list]
    bars = ax.bar(range(len(policies)), productivity_vals, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(range(len(policies)), productivity_vals, yerr=productivity_stds,
               fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=11, fontweight='bold', rotation=15)
    ax.set_ylabel('Coin Produced', fontsize=13, fontweight='bold')
    ax.set_title('Economic Productivity', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, productivity_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(val)}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    # 2) Equality
    ax = axes[1]
    equality_vals = [r['equality'] * 100 for r in results_list]
    equality_stds = [r['equality_std'] * 100 for r in results_list]
    bars = ax.bar(range(len(policies)), equality_vals, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(range(len(policies)), equality_vals, yerr=equality_stds,
               fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=11, fontweight='bold', rotation=15)
    ax.set_ylabel('Coin Equality', fontsize=13, fontweight='bold')
    ax.set_title('Income Equality', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, equality_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(val)}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    # 3) Eq × Prod
    ax = axes[2]
    eq_prod_vals = [r['eq_times_prod'] for r in results_list]
    bars = ax.bar(range(len(policies)), eq_prod_vals, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=11, fontweight='bold', rotation=15)
    ax.set_ylabel('Eq. / Prod. Tradeoff', fontsize=13, fontweight='bold')
    ax.set_title('Equality x Productivity', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, eq_prod_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(val)}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Gráfico de arquitecturas guardado: {save_path}")
    plt.close()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

if __name__ == "__main__":
    results_arch = compare_ai_architectures()
