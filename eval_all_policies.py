import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import ray
from train_ppo_wconfig_planner import (
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
    """
    Evalúa una política de impuestos y retorna métricas.
    
    Args:
        config_path: Path al archivo de configuración YAML
        policy_name: Nombre de la política ("Free Market", "US Federal", etc.)
        trainer: Trainer con pesos cargados (solo para AI Economist)
        max_steps: Pasos máximos por episodio
        n_episodes: Número de episodios a promediar
    
    Returns:
        dict: Métricas promediadas (productivity, equality, etc.)
    """
    print(f"\n{'='*70}")
    print(f"Evaluando: {policy_name}")
    print(f"{'='*70}")
    
    # Cargar configuración
    with open(config_path, 'r') as f:
        run_configuration = yaml.safe_load(f)
    
    # Crear entorno
    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)
    

    # Ejecutar múltiples episodios
    all_productivity = []
    all_equality = []
    all_gini = []
    
    for episode in range(n_episodes):
        obs = env_obj.reset()
        done = {"__all__": False}
        step = 0
        
        while not done["__all__"] and step < max_steps:
            actions = {}
            for agent_id, ob in obs.items():
                policy_id = "a" if str(agent_id).isdigit() else "p"
                action = trainer.compute_action(ob, policy_id=policy_id)
                actions[agent_id] = action
            
            obs, rew, done, info = env_obj.step(actions)
            step += 1
        
        # Calcular métricas finales del episodio
        final_coins = np.array([
            agent.total_endowment('Coin') 
            for agent in env_obj.env.world.agents
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
    
    # Promediar resultados
    results = {
        'policy_name': policy_name,
        'productivity': np.mean(all_productivity),
        'productivity_std': np.std(all_productivity),
        'equality': np.mean(all_equality),
        'equality_std': np.std(all_equality),
        'gini': np.mean(all_gini),
        'gini_std': np.std(all_gini),
        'eq_times_prod': np.mean(all_equality) * np.mean(all_productivity),
    }
    
    print(f"\n  Resultados promediados:")
    print(f"  Productivity: {results['productivity']:.1f} ± {results['productivity_std']:.1f}")
    print(f"  Equality: {results['equality']:.3f} ± {results['equality_std']:.3f}")
    print(f"  Gini: {results['gini']:.3f} ± {results['gini_std']:.3f}")
    print(f"  Eq × Prod: {results['eq_times_prod']:.1f}")
    
    return results

def compare_all_policies():
    """
    Compara las 4 políticas y genera el gráfico tipo AI-Economist.
    """
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    # 1. Free Market (sin impuestos)
    print("\n Cargando pesos del Free Market...")
    with open('configs_eval/free_market.yaml', 'r') as f:
        free_config = yaml.safe_load(f)
    
    env_config = build_env_config(free_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, free_config, env_config)
    free_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
    
    
    agent_weights = torch.load('checkpoints/policy_a_weights.pt', map_location='cpu')
    free_trainer.get_policy("a").model.load_state_dict(agent_weights)
    

    print("Pesos cargados")
    
    results_free_market = evaluate_policy(
        'configs_eval/free_market.yaml',
        'Free Market',
        trainer=free_trainer,
        n_episodes=5
    )

    free_trainer.stop()
    
    # 2. US Federal

    print("\n🔄 Cargando pesos del US Federal...")
    with open('configs_eval/us_fed.yaml', 'r') as f:
        us_fed_config = yaml.safe_load(f)
    
    env_config = build_env_config(us_fed_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, us_fed_config, env_config)
    us_fed_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
    
    agent_weights = torch.load('checkpoints/us_federal/policy_a_weights.pt', map_location='cpu')
    us_fed_trainer.get_policy("a").model.load_state_dict(agent_weights)
    

    print("Pesos cargados")
    
    results_us_federal = evaluate_policy(
        'configs_eval/us_fed.yaml',
        'US Federal',
        trainer=us_fed_trainer,
        n_episodes=5
    )

    us_fed_trainer.stop()
    
    print("\n Cargando pesos de Saez Formula...")
    with open('configs_eval/saez.yaml', 'r') as f:
        saez_config = yaml.safe_load(f)
    
    env_config = build_env_config(saez_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, saez_config, env_config)
    saez_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
    
    agent_weights = torch.load('checkpoints/saez/policy_a_weights.pt', map_location='cpu')
    saez_trainer.get_policy("a").model.load_state_dict(agent_weights)
    

    print("Pesos cargados")
    
    results_saez = evaluate_policy(
        'configs_eval/saez.yaml',
        'Saez Formula',
        trainer=saez_trainer,
        n_episodes=5
    )
    saez_trainer.stop()

    # 4. AI Economist (cargar pesos entrenados)
    print("\n Cargando pesos del AI Economist...")
    with open('configs_eval/ai.yaml', 'r') as f:
        ai_config = yaml.safe_load(f)
    
    env_config = build_env_config(ai_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, ai_config, env_config)
    ai_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
 
    agent_weights = torch.load('checkpoints/policy_a_weights.pt', map_location='cpu')
    ai_trainer.get_policy("a").model.load_state_dict(agent_weights)
    
    planner_weights = torch.load('checkpoints/policy_p_weights_w_planner.pt', map_location='cpu')
    ai_trainer.get_policy("p").model.load_state_dict(planner_weights)
    print(" Pesos cargados")
    
    results_ai_economist = evaluate_policy(
        'configs_eval/ai.yaml',
        'AI Economist',
        trainer=ai_trainer,
        n_episodes=5
    )
    
    ai_trainer.stop()
    ray.shutdown()
    
    all_results = [
        results_free_market,
        results_us_federal,
        results_saez,
        results_ai_economist
    ]
    
    plot_comparison(all_results)
    
    return all_results

def plot_comparison(results_list, save_path='policy_comparison.png'):
    """
    Genera el gráfico de barras estilo AI-Economist paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    policies = [r['policy_name'] for r in results_list]
    colors = ['#FF6B6B', '#C77DFF', '#4EA8DE', '#06A77D']  # Coral, Purple, Blue, Teal
    
    # Subplot 1: Productivity
    ax = axes[0]
    productivity_vals = [r['productivity'] for r in results_list]
    productivity_stds = [r['productivity_std'] for r in results_list]
    bars = ax.bar(range(len(policies)), productivity_vals, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(range(len(policies)), productivity_vals, yerr=productivity_stds,
               fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=11, fontweight='bold')
    ax.set_ylabel('Coin Produced', fontsize=13, fontweight='bold')
    ax.set_title('Economic Productivity', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Valores sobre barras
    for i, (bar, val) in enumerate(zip(bars, productivity_vals)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{int(val)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Subplot 2: Equality
    ax = axes[1]
    equality_vals = [r['equality'] * 100 for r in results_list]  # Como porcentaje
    equality_stds = [r['equality_std'] * 100 for r in results_list]
    bars = ax.bar(range(len(policies)), equality_vals, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(range(len(policies)), equality_vals, yerr=equality_stds,
               fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=11, fontweight='bold')
    ax.set_ylabel('Coin Equality', fontsize=13, fontweight='bold')
    ax.set_title('Income Equality', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Valores sobre barras
    for i, (bar, val) in enumerate(zip(bars, equality_vals)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{int(val)}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Subplot 3: Equality × Productivity
    ax = axes[2]
    eq_prod_vals = [r['eq_times_prod'] for r in results_list]
    bars = ax.bar(range(len(policies)), eq_prod_vals, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=11, fontweight='bold')
    ax.set_ylabel('Eq. / Prod. Tradeoff', fontsize=13, fontweight='bold')
    ax.set_title('Equality x Productivity', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Valores sobre barras
    for i, (bar, val) in enumerate(zip(bars, eq_prod_vals)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{int(val)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Gráfico de comparación guardado: {save_path}")
    plt.close()

if __name__ == "__main__":
    results = compare_all_policies()

    df = pd.DataFrame(results)
    df.to_csv('policy_comparison_results.csv', index=False)
    print("\n Resultados guardados en: policy_comparison_results.csv")