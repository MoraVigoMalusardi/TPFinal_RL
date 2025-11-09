import numpy as np
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
    while not done["__all__"] and step < max_steps:
        actions = {}
        for agent_id, ob in obs.items():
            policy_id = "a" if str(agent_id).isdigit() else "p"
            action = trainer.compute_action(ob, policy_id=policy_id)
            actions[agent_id] = action

        obs, rew, done, info = env_obj.step(actions)

        for agent_id, r in rew.items():
            total_rewards[agent_id] = total_rewards.get(agent_id, 0.0) + r

        step += 1

    print("\n✅ Episodio de evaluación finalizado:")
    print(f"  Pasos ejecutados: {step}")
    print("  Recompensa total por agente:")
    for agent_id, r in total_rewards.items():
        print(f"    {agent_id}: {r:.2f}")

    # === Extraer coins finales (desde info del último paso) ===
    coins = []
    for agent_id, data in info.items():
        if agent_id.startswith("a"):  # solo agentes (no planner)
            if isinstance(data, dict) and "coin" in data:
                coins.append(data["coin"])

    if not coins:
        print("⚠ No se encontraron 'coin' en info. Revisá el dict del entorno.")
        return None

    coins = np.array(coins, dtype=float)
    productivity = coins.sum()
    gini_coeff = gini(coins)
    equality = 1 - (len(coins) / (len(coins) - 1)) * gini_coeff

    print("\n📊 Métricas económicas finales:")
    print(f"  Productividad total: {productivity:.2f}")
    print(f"  Gini: {gini_coeff:.3f}")
    print(f"  Igualdad normalizada: {equality:.3f}")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(coins)), coins, color="steelblue")
        plt.xlabel("Agente")
        plt.ylabel("Monedas finales (coins)")
        plt.title("Distribución final de riqueza por agente")
        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.show()

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
        print("✅ Pesos de política 'a' cargados.")

    p_path = "checkpoints/policy_p_weights.pt"
    if os.path.exists(p_path):
        trainer.get_policy("p").model.load_state_dict(torch.load(p_path, map_location="cpu"))
        print("✅ Pesos de política 'p' cargados.")


    result = run_eval_episode_with_metrics(trainer, env_obj, max_steps=1000)
    print(result)


if __name__ == "__main__":
    main()
