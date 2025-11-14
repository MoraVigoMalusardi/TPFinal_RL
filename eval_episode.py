import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_ppo_wconfig import build_env_config, create_env_for_inspection, process_args
from train_ppo_wconfig import build_trainer_config

from ray.rllib.agents.ppo import PPOTrainer
from train_ppo_wconfig import RLlibEnvWrapper
import ray


import torch
import os



def gini(array):
    """Calcula el coeficiente de Gini de un vector 1D."""
    array = np.array(array, dtype=float)
    if np.amin(array) < 0:
        array -= np.amin(array)
    mean_x = np.mean(array)
    if mean_x == 0:
        return 0.0
    diff_sum = np.abs(array[:, None] - array[None, :]).sum()
    return diff_sum / (2.0 * len(array)**2 * mean_x)


def plot_agent_metrics(agent_idx, agent_data, time_steps, world_map=None, resource_sources=None):
    """
    Genera plots para un agente específico: inventario, trayectoria 2D, labor y skills.
    """
    fig = plt.figure(figsize=(14, 12))
    
    # Plot 1: Inventario
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time_steps, agent_data["Coin"], label="Coins", color="gold", linewidth=2)
    ax1.plot(time_steps, agent_data["Stone"], label="Stone", color="gray", linewidth=2)
    ax1.plot(time_steps, agent_data["Wood"], label="Wood", color="sienna", linewidth=2)
    ax1.set_title(f"Agent {agent_idx} - Inventory Over Time", fontweight='bold')
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Quantity")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Trayectoria 2D con recursos
    ax2 = plt.subplot(2, 2, 2)
    
    # Dibujar fuentes de recursos si están disponibles
    if resource_sources:
        for resource_type, positions in resource_sources.items():
            if resource_type == 'Wood':
                ax2.scatter([p[0] for p in positions], [p[1] for p in positions], 
                           c='green', marker='s', s=200, alpha=0.7, label='Wood Source', 
                           edgecolors='darkgreen', linewidths=2)
            elif resource_type == 'Stone':
                ax2.scatter([p[0] for p in positions], [p[1] for p in positions], 
                           c='gray', marker='^', s=200, alpha=0.7, label='Stone Source', 
                           edgecolors='black', linewidths=2)
    
    locs = np.array(agent_data["Location"])
    ax2.plot(locs[:, 0], locs[:, 1], color="blue", alpha=0.8, linewidth=2.5, label="Trajectory", zorder=3)
    ax2.scatter(locs[0, 0], locs[0, 1], color="lime", s=200, marker="o", label="Start", zorder=5, edgecolors='black', linewidths=2)
    ax2.scatter(locs[-1, 0], locs[-1, 1], color="red", s=200, marker="X", label="End", zorder=5, edgecolors='black', linewidths=2)
    ax2.set_title(f"Agent {agent_idx} - 2D Trajectory with Resources", fontweight='bold')
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.4, linestyle='--')
    ax2.set_aspect('equal', adjustable='box')

    # Plot 3: Labor
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time_steps, agent_data["Labor"], label="Labor", color="green", linewidth=2)
    ax3.set_title(f"Agent {agent_idx} - Labor Over Time", fontweight='bold')
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Labor")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Skills (valores estáticos)
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Obtener valores finales de skills
    build_skill = agent_data["BuildSkill"][-1] if agent_data["BuildSkill"] else 0
    build_payment = agent_data["BuildPayment"][-1] if agent_data["BuildPayment"] else 0
    bonus_gather = agent_data["BonusGatherProb"][-1] if agent_data["BonusGatherProb"] else 0
    
    # Crear texto con las métricas
    skills_text = f"""Agent {agent_idx} - Skills & Attributes
    
Build Skill: {build_skill:.4f}

Build Payment: {build_payment:.2f}

Bonus Gather Probability: {bonus_gather:.4f}
"""
    
    ax4.text(0.5, 0.5, skills_text, 
             transform=ax4.transAxes,
             fontsize=14,
             verticalalignment='center',
             horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')

    plt.tight_layout()
    plt.savefig(f"eval_plots_planner/agent_{agent_idx}_evaluation.png", dpi=150)
    plt.close()


def run_eval_episode_with_metrics(trainer, env_obj, max_steps=200, plot=True):
    """
    Ejecuta un episodio de evaluación con la política entrenada.
    Además calcula productividad total y Gini final de las monedas (coins) por agente.
    """
    obs = env_obj.reset()
    done = {"__all__": False}
    total_rewards = {agent_id: 0.0 for agent_id in obs.keys()}

    step = 0
    info = {}
    
    # Inicializar diccionarios para todos los agentes
    agents_data = [
        {"Coin": [], "Stone": [], "Wood": [], "Location": [], "Labor": [], 
         "BuildSkill": [], "BuildPayment": [], "BonusGatherProb": []}
        for _ in range(4)
    ]
    
    # Obtener posiciones de recursos desde los mapas
    resource_sources = {}
    
    if hasattr(env_obj.env, 'world') and hasattr(env_obj.env.world, 'maps'):
        world = env_obj.env.world
        
        print(f"\nExtrayendo recursos de los mapas...")
        for resource_name, resource_map in world.maps.items():
            if resource_name in ['Wood', 'Stone']:
                # Encontrar posiciones donde hay recursos (valores > 0)
                positions = np.argwhere(resource_map > 0)
                if len(positions) > 0:
                    resource_sources[resource_name] = positions.tolist()
                    print(f"  {resource_name}: {len(positions)} posiciones encontradas")
        
        print(f"\nRecursos detectados: {[(k, len(v)) for k, v in resource_sources.items()]}")
    
    # Debug: explorar estructura del primer agente
    debug_done = False

    while not done["__all__"] and step < max_steps:
        actions = {}
        for agent_id, ob in obs.items():
            policy_id = "a" if str(agent_id).isdigit() else "p"
            action = trainer.compute_action(ob, policy_id=policy_id)
            actions[agent_id] = action

        obs, rew, done, info = env_obj.step(actions)

        for agent in env_obj.env.world.agents:
            agent_data = agents_data[agent.idx]
            agent_data["Coin"].append(agent.total_endowment("Coin"))
            agent_data["Stone"].append(agent.total_endowment("Stone"))
            agent_data["Wood"].append(agent.total_endowment("Wood"))
            agent_data["Location"].append(agent.loc)
            agent_data["Labor"].append(agent.endogenous.get("Labor", 0))
            
            # Obtener skills desde agent.state
            if hasattr(agent, 'state') and isinstance(agent.state, dict):
                agent_data["BuildSkill"].append(agent.state.get('build_skill', 0))
                agent_data["BuildPayment"].append(agent.state.get('build_payment', 0))
                agent_data["BonusGatherProb"].append(agent.state.get('bonus_gather_prob', 0))
            else:
                agent_data["BuildSkill"].append(0)
                agent_data["BuildPayment"].append(0)
                agent_data["BonusGatherProb"].append(0)

        for agent_id, r in rew.items():
            total_rewards[agent_id] = total_rewards.get(agent_id, 0.0) + r

        step += 1

    print("\nEpisodio de evaluación finalizado:")
    print(f"  Pasos ejecutados: {step}")
    print("  Recompensa total por agente:")
    for agent_id, r in total_rewards.items():
        print(f"    {agent_id}: {r:.2f}")
    
    # Debug: Verificar si las skills cambian
    print("\nVerificando cambios en skills:")
    for i, agent_data in enumerate(agents_data):
        build_skill_unique = len(set(agent_data["BuildSkill"]))
        build_payment_unique = len(set(agent_data["BuildPayment"]))
        bonus_gather_unique = len(set(agent_data["BonusGatherProb"]))
        
        print(f"  Agent {i}:")
        print(f"    BuildSkill valores únicos: {build_skill_unique} (min: {min(agent_data['BuildSkill']):.4f}, max: {max(agent_data['BuildSkill']):.4f})")
        print(f"    BuildPayment valores únicos: {build_payment_unique} (min: {min(agent_data['BuildPayment']):.4f}, max: {max(agent_data['BuildPayment']):.4f})")
        print(f"    BonusGatherProb valores únicos: {bonus_gather_unique} (min: {min(agent_data['BonusGatherProb']):.4f}, max: {max(agent_data['BonusGatherProb']):.4f})")

    # === Mostrar evolución de inventarios y locaciones ===
    if plot:
        time_steps = range(step)
        for i, agent_data in enumerate(agents_data):
            plot_agent_metrics(i, agent_data, time_steps, None, resource_sources)

    # === Extraer coins finales ===
    final_coins = np.array([agent_data["Coin"][-1] for agent_data in agents_data], dtype=float)

    coins = final_coins
    productivity = coins.sum()
    gini_coeff = gini(coins)
    equality = 1 - (len(coins) / (len(coins) - 1)) * gini_coeff

    print("\nMétricas económicas finales:")
    print(f"  Productividad total: {productivity:.2f}")
    print(f"  Gini: {gini_coeff:.3f}")
    print(f"  Igualdad normalizada: {equality:.3f}")

    if plot:
        # Crear figura con dos subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Distribución de riqueza
        ax = axes[0]
        agent_labels = [f"Agent {i}" for i in range(4)]
        colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
        bars = ax.bar(agent_labels, coins, color=colors, alpha=0.75, edgecolor='black', linewidth=1.5)
        ax.set_xlabel("Agent", fontsize=12, fontweight='bold')
        ax.set_ylabel("Final Coins", fontsize=12, fontweight='bold')
        ax.set_title("Wealth Distribution (Final State)", fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Agregar valores sobre las barras
        for bar, value in zip(bars, coins):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Subplot 2: Métricas económicas
        ax = axes[1]
        metrics = ['Productivity', 'Equality\n(1-Gini)', 'Eq×Prod\n']
        values = [productivity, equality * 100, (equality * productivity) / 100]
        metric_colors = ['coral', 'mediumpurple', 'teal']
        bars = ax.bar(metrics, values, color=metric_colors, alpha=0.75, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Economic Metrics', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Valores sobre barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        filename = "eval_plots_planner/economic_summary.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Resumen económico guardado: {filename}")
        plt.close()

    return {"productivity": productivity, "gini": gini_coeff, "equality": equality}


def main():

    # Parsear argumentos y cargar configuración del YAML
    run_configuration, run_dir = process_args()
    # Extraer configuración del entorno
    env_config = build_env_config(run_configuration)

    # Crear un entorno local para consultar espacios y para eval
    env_obj = create_env_for_inspection(env_config)


    # Construir config de PPO + multi-agent desde YAML
    trainer_config = build_trainer_config(env_obj, run_configuration, env_config)


    ray.init(include_dashboard=False, log_to_driver=False)

    trainer = PPOTrainer(
        env=RLlibEnvWrapper,
        config=trainer_config
    )



    # cargar pesos
    a_path = "checkpoints/policy_a_weights.pt"
    if os.path.exists(a_path):
        trainer.get_policy("a").model.load_state_dict(torch.load(a_path, map_location="cpu"))
        print("Pesos de política 'a' cargados.")

    p_path = "checkpoints/policy_p_weights_w_planner.pt"
    if os.path.exists(p_path):
        trainer.get_policy("p").model.load_state_dict(torch.load(p_path, map_location="cpu"))
        print("Pesos de política 'p' cargados.")


    result = run_eval_episode_with_metrics(trainer, env_obj, max_steps=1000)
    print(result)


if __name__ == "__main__":
    main()
