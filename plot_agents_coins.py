import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_agents_ppo import (
    build_env_config,
    create_env_for_inspection,
    process_args,
    build_trainer_config,
    RLlibEnvWrapper,
)
from ray.rllib.agents.ppo import PPOTrainer
import ray
import torch
import os

np.random.seed(42)


def plot_agents_coins(trainer, env_obj, max_steps=200, plot=True):

    # --- Definir skills fijos (una sola vez) ---
    fixed_gather_skills = [1.0, 1.0, 1.0, 1.0]
    bonus_gather_prob = [0.0, 0.0, 0.0, 0.0]
    fixed_build_skills = []
    build_payment = []
    PMSM = 3

    for agent_idx in range(4):
        sampled_skill = np.random.pareto(4)
        fixed_build_skills.append(sampled_skill)
        pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
        bp = float(pay_rate * 10)
        build_payment.append(bp)

    print("\n=== Skills FIJOS elegidos para todos los episodios ===")
    for i in range(4):
        print(
            f"Agente {i}: build_skill={fixed_build_skills[i]:.4f}, "
            f"build_payment={build_payment[i]:.4f}, "
            f"gather_skill={fixed_gather_skills[i]}, "
            f"bonus_gather_prob={bonus_gather_prob[i]}"
        )

    # Guardo una lista de agents_data para luego sacar el promedio de cada cosa
    all_runs_agents_data = []

    for run_idx in range(100):
        print(f"\nIniciando episodio de evaluación {run_idx + 1}/100...")
        obs = env_obj.reset()

        # ---- Skills que pone el entorno por defecto ----
        print("\nSkills originales del entorno (antes de sobrescribir):")
        for agent_idx in range(4):
            agent = env_obj.env.world.agents[agent_idx]
            if hasattr(agent, "state") and isinstance(agent.state, dict):
                print(
                    f"  Agente {agent_idx}: "
                    f"build_skill={agent.state.get('build_skill')}, "
                    f"build_payment={agent.state.get('build_payment')}, "
                    f"gather_skill={agent.state.get('gather_skill')}, "
                    f"bonus_gather_prob={agent.state.get('bonus_gather_prob')}"
                )

        # ---- Sobrescribir con los skills fijos ----
        print("\nSobrescribiendo skills con valores FIJOS:")
        for agent_idx in range(4):
            agent = env_obj.env.world.agents[agent_idx]
            if hasattr(agent, 'state') and isinstance(agent.state, dict):

                agent.state["build_payment"] = build_payment[agent_idx]
                agent.state["build_skill"] = float(fixed_build_skills[agent_idx])

                agent.state['gather_skill'] = fixed_gather_skills[agent_idx]
                agent.state['bonus_gather_prob'] = bonus_gather_prob[agent_idx]

                print(
                    f"  Agente {agent_idx}: "
                    f"build_skill={agent.state['build_skill']:.4f}, "
                    f"build_payment={agent.state['build_payment']:.4f}, "
                    f"gather_skill={agent.state['gather_skill']}, "
                    f"bonus_gather_prob={agent.state['bonus_gather_prob']}"
                )

        done = {"__all__": False}
        total_rewards = {agent_id: 0.0 for agent_id in obs.keys()}
        step = 0

        # Inicializar diccionarios para todos los agentes
        agents_data = [
            {
                "Coin": [],
                "Stone": [],
                "Wood": [],
                "Location": [],
                "Labor": [],
                "BuildSkill": [],
                "BuildPayment": [],
                "BonusGatherProb": [],
            }
            for _ in range(4)
        ]

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

                # Guardar skills (para ver si se mantienen constantes)
                if hasattr(agent, 'state') and isinstance(agent.state, dict):
                    bs = agent.state.get('build_skill', 0)
                    bp = agent.state.get('build_payment', 0)
                    bgp = agent.state.get('bonus_gather_prob', 0)

                    agent_data["BuildSkill"].append(bs)
                    agent_data["BuildPayment"].append(bp)
                    agent_data["BonusGatherProb"].append(bgp)

                    # Chequeo fuerte: si algo se cambió, explotamos
                    idx = agent.idx
                    assert abs(bs - fixed_build_skills[idx]) < 1e-6, \
                        f"build_skill cambió en runtime para agente {idx}"
                    assert abs(bp - build_payment[idx]) < 1e-6, \
                        f"build_payment cambió en runtime para agente {idx}"
                    assert abs(agent.state['gather_skill'] - fixed_gather_skills[idx]) < 1e-6, \
                        f"gather_skill cambió en runtime para agente {idx}"
                else:
                    agent_data["BuildSkill"].append(0)
                    agent_data["BuildPayment"].append(0)
                    agent_data["BonusGatherProb"].append(0)

            for agent_id, r in rew.items():
                total_rewards[agent_id] = total_rewards.get(agent_id, 0.0) + r

            step += 1

        all_runs_agents_data.append(agents_data)

        print("\nEpisodio de evaluación finalizado:")
        print(f"  Pasos ejecutados: {step}")
        print("  Recompensa total por agente:")
        for agent_id, r in total_rewards.items():
            print(f"    {agent_id}: {r:.2f}")

    # --- Promediar datos de todos los runs ---
    agents_data = [
        {"Coin": [], "Stone": [], "Wood": [], "Location": [], "Labor": [],
         "BuildSkill": [], "BuildPayment": [], "BonusGatherProb": []}
        for _ in range(4)
    ]

    for agent_idx in range(4):
        for key in agents_data[agent_idx].keys():
            averaged_data = np.mean(
                [all_runs_agents_data[run_idx][agent_idx][key] for run_idx in range(100)],
                axis=0
            )
            agents_data[agent_idx][key] = averaged_data.tolist()

    # --- Gráficos ---
    if plot:
        plt.figure(figsize=(12, 8))
        for agent_idx, agent_data in enumerate(agents_data):
            skill = agent_data["BuildSkill"][-1] if agent_data["BuildSkill"] else 0.0
            plt.plot(agent_data["Coin"], label=f"Agente {agent_idx} - Skill {skill:.2f}")
        plt.xlabel("Pasos")
        plt.ylabel("Coins recolectadas")
        plt.grid()
        plt.legend()
        plt.savefig("coins_build_skill.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        for agent_idx, agent_data in enumerate(agents_data):
            skill = agent_data["BuildSkill"][-1] if agent_data["BuildSkill"] else 0.0
            plt.plot(agent_data["Labor"], label=f"Agente {agent_idx} - Skill {skill:.2f}")
        plt.xlabel("Pasos")
        plt.ylabel("Labor")
        plt.grid()
        plt.legend()
        plt.savefig("labor_over_time.png")
        plt.close()

    return agents_data


def main():
    run_configuration, run_dir = process_args()
    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)

    trainer_config = build_trainer_config(env_obj, run_configuration, env_config)

    ray.init(include_dashboard=False, log_to_driver=False)

    trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)

    a_path = "policy_a_weights.pt"
    if os.path.exists(a_path):
        trainer.get_policy("a").model.load_state_dict(torch.load(a_path, map_location="cpu"))
        print("Pesos de política 'a' cargados.")

    result = plot_agents_coins(trainer, env_obj, max_steps=1000)
    print("Listo, eval terminada.")
    ray.shutdown()


if __name__ == "__main__":
    main()
