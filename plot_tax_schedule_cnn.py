import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import ray
from train_planner_ppo_cnn import (
    build_env_config as build_env_config_cnn,
    create_env_for_inspection as create_env_for_inspection_cnn,
    build_trainer_config as build_trainer_config_cnn,
    SafeEnvWrapper,
    AI_Economist_CNN_PyTorch,
)
from ray.rllib.models import ModelCatalog

import torch
import pandas as pd

from ray.rllib.agents.ppo import PPOTrainer

def gini(array):
    """Calcula el coeficiente de Gini."""
    array = np.array(array, dtype=float)
    if len(array) == 0 or array.sum() == 0:
        return 0.0
    sorted_array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_array)) / (n * np.sum(sorted_array)) - (n + 1) / n

def get_base_env(env_obj):
    """
    Devuelve el entorno base de AI-Economist que tiene .world.agents.
    - MLP/LSTM: env_obj es RLlibEnvWrapper  -> usar env_obj.env
    - CNN:      env_obj es SafeEnvWrapper   -> usar env_obj.internal_env.env
    """
    if hasattr(env_obj, "env"):
        return env_obj.env
    if hasattr(env_obj, "internal_env") and hasattr(env_obj.internal_env, "env"):
        return env_obj.internal_env.env
    raise AttributeError("No pude encontrar atributo .env ni .internal_env.env en env_obj")


def evaluate_policy2(config_path, policy_name, trainer=None, max_steps=1000, n_episodes=5):
    print(f"\n{'='*70}")
    print(f"Evaluando: {policy_name} (MODO HÍBRIDO: Brackets Auto + Acción Directa)")
    print(f"{'='*70}")
    
    with open(config_path, 'r') as f:
        run_configuration = yaml.safe_load(f)
    
    env_config = build_env_config_cnn(run_configuration)
    env_obj = create_env_for_inspection_cnn(env_config)

    # Desenrollar wrappers hasta llegar al entorno de AI-Economist
    env_core = get_core_env(env_obj)
    print(f"[DEBUG] Tipo de env_core: {type(env_core)}")
    # ============================================================

    last_episode_tax_history = [] 
    brackets = [] 

    all_productivity = []
    all_equality = []
    all_gini = []

    # -----------------------------------------------------------
    # 1. DETECCIÓN DE BRACKETS (usar env_core, no env_obj)
    # -----------------------------------------------------------
    tax_component = None
    for comp in env_core.components:
        if "Tax" in str(type(comp)):
            tax_component = comp
            if hasattr(comp, 'bracket_cutoffs'):
                brackets = comp.bracket_cutoffs
            elif hasattr(comp, 'brackets'):
                brackets = comp.brackets
            break
            
    if len(brackets) == 0:
        brackets = [10, 50, 100, 500, 1000, 2000, 5000]
    
    if hasattr(brackets, 'tolist'):
        brackets = brackets.tolist()
    
    print(f"Componente Fiscal: {type(tax_component).__name__}")
    print(f"Brackets detectados: {brackets}")
    # -----------------------------------------------------------

    for episode in range(n_episodes):
        obs = env_obj.reset()
        done = {"__all__": False}
        step = 0
        current_episode_taxes = []
        
        # para métricas de este episodio
        initial_coins = [agent.total_endowment('Coin') for agent in env_core.world.agents]

        last_planner_action_rates = np.zeros(len(brackets))

        while not done["__all__"] and step < max_steps:
            actions = {}
            for agent_id, ob in obs.items():
                policy_id = "a" if str(agent_id).isdigit() else "p"
                
                if trainer:
                    action = trainer.compute_action(ob, policy_id=policy_id, explore=False)
                else:
                    action = env_obj.action_space.sample()[agent_id]
                
                actions[agent_id] = action

                if policy_id == "p":
                    # Cada sub-acción es un índice en:
                    # {0, 0.05, 0.10, ..., 1.0}  (21 valores) + NO-OP (índice 21)
                    ACTION_LEVELS = np.linspace(0.0, 1.0, 21)  # [0.00, 0.05, ..., 1.00]

                    if hasattr(action, '__iter__'):
                        new_rates = []

                        for idx, prev_rate in zip(action, last_planner_action_rates):
                            idx = int(idx)

                            if idx == 21:
                                # NO-OP: mantiene la tasa que ya tenía ese bracket
                                new_rates.append(prev_rate)
                            else:
                                # 0..20 -> 0, 0.05, 0.10, ..., 1.0
                                new_rates.append(ACTION_LEVELS[idx])

                        rates = np.array(new_rates, dtype=float)

                        # Guardamos el vector de tasas actual como "última decisión"
                        if len(rates) == len(brackets):
                            last_planner_action_rates = rates
                        elif len(rates) > len(brackets):
                            last_planner_action_rates = rates[:len(brackets)]
                        else:
                            last_planner_action_rates[:len(rates)] = rates



            obs, rew, done, info = env_obj.step(actions)    
            current_episode_taxes.append(last_planner_action_rates)
            step += 1
        
        final_coins = np.array([
            agent.total_endowment('Coin')
            for agent in env_core.world.agents
        ])

        productivity = final_coins.sum()
        gini_coeff = gini(final_coins)
        n_agents = len(final_coins)
        equality = 1 - (n_agents / (n_agents - 1)) * gini_coeff if n_agents > 1 else 1.0

        all_productivity.append(productivity)
        all_gini.append(gini_coeff)
        all_equality.append(equality)

        print(f"  Episode {episode+1}/{n_episodes}: "
              f"Prod={productivity:.1f}, Eq={equality:.3f}, Gini={gini_coeff:.3f}")

        if episode == n_episodes - 1:
            last_episode_tax_history = np.array(current_episode_taxes)
        
        print(f"  Episode {episode+1} Done.")
    
    # --- NUEVO: promedios finales ---
    mean_prod = float(np.mean(all_productivity))
    mean_eq = float(np.mean(all_equality))
    mean_gini = float(np.mean(all_gini))

    print("\nResultados promediados en modo híbrido:")
    print(f"  Productivity: {mean_prod:.1f}")
    print(f"  Equality   : {mean_eq:.3f}")
    print(f"  Gini       : {mean_gini:.3f}")
    print(f"  Eq x Prod  : {mean_eq * mean_prod:.1f}")

    results = {
        'policy_name': policy_name,
        'tax_history': last_episode_tax_history,
        'brackets': brackets,
        'productivity': mean_prod,
        'equality': mean_eq,
        'gini': mean_gini,
        'eq_times_prod': mean_eq * mean_prod,
    }
    
    return results


def evaluate_policy(config_path, policy_name, trainer=None, max_steps=1000, n_episodes=5):
    print(f"\n{'='*70}")
    print(f"Evaluando: {policy_name} (MODO HÍBRIDO: Brackets Auto + Acción Directa)")
    print(f"{'='*70}")
    
    with open(config_path, 'r') as f:
        run_configuration = yaml.safe_load(f)
    
    # 1) Construyo el wrapper que usás para CNN
    env_config = build_env_config_cnn(run_configuration)
    env_obj = create_env_for_inspection_cnn(env_config)

    # 2) Obtengo el entorno base tal como hacías en evaluate_trainer_on_env
    base_env = get_base_env(env_obj)      # esto ya funciona para CNN y MLP/LSTM
    # En muchos casos base_env ya ES el entorno AI-Economist; si no, lo desempaquetamos un paso más:
    env_core = getattr(base_env, "env", base_env)

    print(f"[DEBUG] base_env: {type(base_env)}")
    print(f"[DEBUG] env_core: {type(env_core)}")

    last_episode_tax_history = [] 
    brackets = [] 

    all_productivity = []
    all_equality = []
    all_gini = []

    # -----------------------------------------------------------
    # 1. DETECCIÓN DE BRACKETS (usar env_core)
    # -----------------------------------------------------------
    tax_component = None
    if hasattr(env_core, "components"):
        for comp in env_core.components:
            if "Tax" in str(type(comp)):
                tax_component = comp
                if hasattr(comp, 'bracket_cutoffs'):
                    brackets = comp.bracket_cutoffs
                elif hasattr(comp, 'brackets'):
                    brackets = comp.brackets
                break
            
    if len(brackets) == 0:
        brackets = [10, 50, 100, 500, 1000, 2000, 5000]
    
    if hasattr(brackets, 'tolist'):
        brackets = brackets.tolist()
    
    print(f"Componente Fiscal: {type(tax_component).__name__ if tax_component is not None else 'None'}")
    print(f"Brackets detectados: {brackets}")

    # -----------------------------------------------------------
    for episode in range(n_episodes):
        obs = env_obj.reset()
        done = {"__all__": False}
        step = 0
        current_episode_taxes = []
        
        # para métricas de este episodio: usar env_core.world.agents
        initial_coins = [agent.total_endowment('Coin') for agent in env_core.world.agents]

        last_planner_action_rates = np.zeros(len(brackets))

        while not done["__all__"] and step < max_steps:
            actions = {}
            for agent_id, ob in obs.items():
                policy_id = "a" if str(agent_id).isdigit() else "p"
                
                if trainer:
                    action = trainer.compute_action(ob, policy_id=policy_id, explore=False)
                else:
                    action = env_obj.action_space.sample()[agent_id]
                
                actions[agent_id] = action

                if policy_id == "p":
                    ACTION_LEVELS = np.linspace(0.0, 1.0, 21)  # [0.00, 0.05, ..., 1.00]

                    if hasattr(action, '__iter__'):
                        new_rates = []

                        for idx, prev_rate in zip(action, last_planner_action_rates):
                            idx = int(idx)

                            if idx == 21:
                                new_rates.append(prev_rate)    # NO-OP
                            else:
                                new_rates.append(ACTION_LEVELS[idx])

                        rates = np.array(new_rates, dtype=float)

                        if len(rates) == len(brackets):
                            last_planner_action_rates = rates
                        elif len(rates) > len(brackets):
                            last_planner_action_rates = rates[:len(brackets)]
                        else:
                            last_planner_action_rates[:len(rates)] = rates

            obs, rew, done, info = env_obj.step(actions)    
            current_episode_taxes.append(last_planner_action_rates)
            step += 1
        
        # Métricas del episodio: también usando env_core.world.agents
        final_coins = np.array([
            agent.total_endowment('Coin')
            for agent in env_core.world.agents
        ])

        productivity = final_coins.sum()
        gini_coeff = gini(final_coins)
        n_agents = len(final_coins)
        equality = 1 - (n_agents / (n_agents - 1)) * gini_coeff if n_agents > 1 else 1.0

        all_productivity.append(productivity)
        all_gini.append(gini_coeff)
        all_equality.append(equality)

        print(f"  Episode {episode+1}/{n_episodes}: "
              f"Prod={productivity:.1f}, Eq={equality:.3f}, Gini={gini_coeff:.3f}")

        if episode == n_episodes - 1:
            last_episode_tax_history = np.array(current_episode_taxes)
        
        print(f"  Episode {episode+1} Done.")
    
    # --- promedios finales ---
    mean_prod = float(np.mean(all_productivity))
    mean_eq = float(np.mean(all_equality))
    mean_gini = float(np.mean(all_gini))

    print("\nResultados promediados en modo híbrido:")
    print(f"  Productivity: {mean_prod:.1f}")
    print(f"  Equality   : {mean_eq:.3f}")
    print(f"  Gini       : {mean_gini:.3f}")
    print(f"  Eq x Prod  : {mean_eq * mean_prod:.1f}")

    results = {
        'policy_name': policy_name,
        'tax_history': last_episode_tax_history,
        'brackets': brackets,
        'productivity': mean_prod,
        'equality': mean_eq,
        'gini': mean_gini,
        'eq_times_prod': mean_eq * mean_prod,
    }
    
    return results


def plot_planner_schedule(results, save_path="planner_tax_schedule.png"):
    tax_history = results.get('tax_history', [])
    brackets = results.get('brackets', [])

    if len(tax_history) == 0 or len(brackets) == 0:
        print("⚠️ No hay datos suficientes para graficar la policy del planner.")
        return

    tax_history = np.asarray(tax_history)

    # Opción 1: usar la última acción del planner
    # planner_rates = tax_history[-1]          # shape: [n_brackets]

    # Opción 2 (alternativa): promedio a lo largo del episodio
    planner_rates = tax_history.mean(axis=0)

    n_brackets = len(planner_rates)
    indices = np.arange(n_brackets)

    # Construir etiquetas de los intervalos, tipo "0-10", "10-50", ..., "> último"
    labels = []
    prev_b = 0
    for b in brackets:
        if b == 0:
            continue  # saltar el 0 inicial
        labels.append(f"{prev_b}-{b}")
        prev_b = b
    if len(labels) < len(planner_rates):
        labels.append(f"> {brackets[-1]}")

    if len(labels) != n_brackets:
        # fallback de seguridad
        labels = [f"B{i}" for i in range(n_brackets)]


    labels = []
    prev_b = 0
    for b in brackets:
        if b == 0:
            continue  # saltar el 0 inicial
        labels.append(f"{prev_b}-{b}")
        prev_b = b
    if len(labels) < len(planner_rates):
        labels.append(f"> {brackets[-1]}")


    plt.figure(figsize=(8, 5))
    plt.bar(indices, planner_rates, edgecolor='black', alpha=0.9)
    plt.xticks(indices, labels, rotation=30)
    plt.ylim(0, 1.05)
    plt.ylabel("Marginal Tax Rate")
    plt.xlabel("Income Brackets (Coins)")
    # plt.title("Planner Tax Schedule", fontweight="bold")
    plt.grid(axis='y', alpha=0.3)

    # Mostrar el valor numérico arriba de cada barra
    for x, y in zip(indices, planner_rates):
        if y > 0.01:
            plt.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[GRÁFICO] Planner tax schedule guardado en: {save_path}")
    plt.close()


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    config = 'phase_aiecon_cnn_eval/config.yaml'  #ACA VA LA CONFIG DE EVALUACION DE AI ECONOMIST
    agents = 'checkpoints/nuevo_cnn_planner/policy_a_cnn_weights_w_planner.pt' #ACA VA EL CHECKPOINT DE LOS AGENTES DE AI ECONOMIST (WITH PLANNER)
    planner = 'checkpoints/nuevo_cnn_planner/policy_p_cnn_weights_w_planner.pt' #ACA VA EL CHECKPOINT DEL PLANNER DE AI ECONOMIST (WITH PLANNER)

    with open(config, 'r') as f:
        ai_config = yaml.safe_load(f)
    
    env_config = build_env_config_cnn(ai_config)
    env_obj = create_env_for_inspection_cnn(env_config)
    trainer_config = build_trainer_config_cnn(env_obj, ai_config, env_config)

    # Registrar el modelo CNN, igual que en tu script de entrenamiento
    ModelCatalog.register_custom_model("paper_cnn_torch", AI_Economist_CNN_PyTorch)

    ai_trainer = PPOTrainer(env=SafeEnvWrapper, config=trainer_config)

 
    agent_weights = torch.load(agents, map_location='cpu')
    ai_trainer.get_policy("a").model.load_state_dict(agent_weights)
    
    planner_weights = torch.load(planner, map_location='cpu')
    ai_trainer.get_policy("p").model.load_state_dict(planner_weights)
    print(" Pesos cargados")
    
    results_ai_economist = evaluate_policy(
        config,
        'AI Economist',
        trainer=ai_trainer,
        n_episodes=1
    )
    # plot_tax_comparison(results_ai_economist, output_dir=".")
    plot_planner_schedule(results_ai_economist, save_path="planner_tax_schedule.png")
    
    ai_trainer.stop()
    del ai_trainer
    del env_obj
    ray.shutdown()


