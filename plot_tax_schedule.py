import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import ray
from train_agents_ppo import (
    build_env_config, 
    create_env_for_inspection,
    build_trainer_config,
    RLlibEnvWrapper
)
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


def evaluate_policy(config_path, policy_name, trainer=None, max_steps=1000, n_episodes=5):
    print(f"\n{'='*70}")
    print(f"Evaluando: {policy_name} (MODO HÍBRIDO: Brackets Auto + Acción Directa)")
    print(f"{'='*70}")
    
    with open(config_path, 'r') as f:
        run_configuration = yaml.safe_load(f)
    
    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)
    
    last_episode_tax_history = [] 
    brackets = [] 

    # --- NUEVO: listas para métricas ---
    all_productivity = []
    all_equality = []
    all_gini = []

    # -----------------------------------------------------------
    # 1. DETECCIÓN DE BRACKETS
    # -----------------------------------------------------------
    tax_component = None
    for comp in env_obj.env.components:
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
        initial_coins = [agent.total_endowment('Coin') for agent in env_obj.env.world.agents]

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
        
        # --- NUEVO: métricas del episodio ---
        final_coins = np.array([
            agent.total_endowment('Coin')
            for agent in env_obj.env.world.agents
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
    #results = compare_all_policies()

    #df = pd.DataFrame(results)
    #df.to_csv('policy_comparison_results.csv', index=False)
    #print("\n Resultados guardados en: policy_comparison_results.csv")
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    config = 'phase_aiecon_eval/config.yaml'  #ACA VA LA CONFIG DE EVALUACION DE AI ECONOMIST
    agents = 'checkpoints/nuevo_sin_lstm/policy_a_weights_w_planner.pt' #ACA VA EL CHECKPOINT DE LOS AGENTES DE AI ECONOMIST (WITH PLANNER)
    planner = 'checkpoints/nuevo_sin_lstm/policy_p_weights_w_planner.pt' #ACA VA EL CHECKPOINT DEL PLANNER DE AI ECONOMIST (WITH PLANNER)

    with open(config, 'r') as f:
        ai_config = yaml.safe_load(f)
    
    env_config = build_env_config(ai_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, ai_config, env_config)
    ai_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
 
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


