import os
import ray
from ray.rllib.agents.ppo import PPOTrainer

# Importa el wrapper de entorno tal como en el tutorial
from tutorials.rllib.env_wrapper import RLlibEnvWrapper

import os
import logging
import warnings

# Desactivar dashboard (fuente del error aiohttp.signals)
os.environ["RAY_DISABLE_DASHBOARD"] = "1"

# Silenciar TensorFlow y PyTorch (por si acaso)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Desactivar warnings de deprecación de Gym y otros
warnings.filterwarnings("ignore")

# Limitar el spam de Ray y RLlib
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.worker").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)


# -------------------------------------------------------------------
# 1) Configuración del entorno "gather-trade-build"
# -------------------------------------------------------------------
env_config_dict = {
    # Escenario que queremos usar
    "scenario_name": "layout_from_file/simple_wood_and_stone",

    # Componentes del entorno: construir casas, mercado, recolección y planner
    "components": [
        # (1) Construcción de casas
        (
            "Build",
            {
                "skill_dist": "pareto",
                "payment_max_skill_multiplier": 3,
                "build_labor": 10,
                "payment": 10,
            },
        ),
        # (2) Mercado de recursos (subasta doble continua)
        (
            "ContinuousDoubleAuction",
            {
                "max_bid_ask": 10,
                "order_labor": 0.25,
                "max_num_orders": 5,
                "order_duration": 50,
            },
        ),
        # (3) Movimiento y recolección de recursos
        (
            "Gather",
            {
                "move_labor": 1,
                "collect_labor": 1,
                "skill_dist": "pareto",
            },
        ),
        # (4) Planner con esquema de impuestos
        (
            "PeriodicBracketTax",
            {
                "period": 100,
                "bracket_spacing": "us-federal",
                "usd_scaling": 1000,
                "disable_taxes": False,
            },
        ),
    ],

    # Parámetros propios del escenario (no de la base)
    "env_layout_file": "quadrant_25x25_20each_30clump.txt",
    "starting_agent_coin": 10,
    "fixed_four_skill_and_loc": True,

    # Parámetros estándar (BaseEnvironment)
    "n_agents": 4,
    "world_size": [25, 25],
    "episode_length": 1000,

    # Multi-acción: planner sí, agentes no
    "multi_action_mode_agents": False,
    "multi_action_mode_planner": True,

    # Observaciones y máscaras aplanadas (como en el tutorial)
    "flatten_observations": True,
    "flatten_masks": True,

    # Frecuencia de dense logs
    "dense_log_frequency": 1,
}


def create_env_for_inspection():
    """
    Crea un entorno local (no de RLlib) para inspeccionar spaces
    y para hacer rollouts manuales de evaluación.
    """
    env_obj = RLlibEnvWrapper({"env_config_dict": env_config_dict}, verbose=True)
    return env_obj


# -------------------------------------------------------------------
# 2) Definición de políticas multi-agente (agentes + planner)
# -------------------------------------------------------------------
def build_multiagent_policies(env_obj):
    """
    Define las políticas multi-agente para RLlib:
    - "a" = política compartida entre todos los agentes (IDs numéricos)
    - "p" = política del planner (ID "p")
    """
    policies = {
        "a": (
            None,  # usa la política default de RLlib (PPO)
            env_obj.observation_space,
            env_obj.action_space,
            {},
        ),
        "p": (
            None,  # política default también
            env_obj.observation_space_pl,
            env_obj.action_space_pl,
            {},
        ),
    }

    # Mapea ID del agente -> ID de la policy
    #   - agentes: IDs enteros (0,1,2,3,...) -> policy "a"
    #   - planner: ID "p" -> policy "p"
    def policy_mapping_fn(agent_id):
        return "a" if str(agent_id).isdigit() else "p"

    policies_to_train = ["a", "p"]
    return policies, policy_mapping_fn, policies_to_train


# -------------------------------------------------------------------
# 3) Construcción del config de PPO + entorno
# -------------------------------------------------------------------
def build_trainer_config(env_obj):
    policies, policy_mapping_fn, policies_to_train = build_multiagent_policies(
        env_obj
    )

    trainer_config = {
        "multiagent": {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": policy_mapping_fn,
        }
    }

    # Parámetros básicos de entrenamiento (igual que el tutorial)
    trainer_config.update(
        {
            "num_workers": 2,
            "num_envs_per_worker": 2,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 4000,
            "num_sgd_iter": 1,
            "framework": "torch",
            "num_gpus": 0,
            "log_level": "ERROR",
        }
    )

    # Config específico del wrapper (lo usa para indexar envs)
    env_config = {
        "env_config_dict": env_config_dict,
        "num_envs_per_worker": trainer_config["num_envs_per_worker"],
    }

    trainer_config["env_config"] = env_config
    return trainer_config


# -------------------------------------------------------------------
# 4) Entrenamiento PPO
# -------------------------------------------------------------------
def train(trainer, num_iters=5):
    """
    Entrena el PPOTrainer por num_iters iteraciones e imprime reward medio.
    """
    for it in range(num_iters):
        print(f"\n********** Iteración: {it} **********")
        result = trainer.train()
        # print("RESULT: ", result)
        print(f"episode_reward_mean: {result.get('episode_reward_mean')}")


# -------------------------------------------------------------------
# 5) Rollout de evaluación con la política entrenada
# -------------------------------------------------------------------
def run_eval_episode(trainer, env_obj, max_steps=200):
    """
    Ejecuta un episodio en un entorno local usando las políticas entrenadas.
    No usa los workers de RLlib, sino el env_obj directo, para "probar" el entorno.
    """
    obs = env_obj.reset()
    done = {"__all__": False}

    # Acumulamos recompensas por agente
    total_rewards = {agent_id: 0.0 for agent_id in obs.keys()}

    step = 0
    while not done["__all__"] and step < max_steps:
        actions = {}
        for agent_id, ob in obs.items():
            # La misma lógica de mapeo que antes:
            policy_id = "a" if str(agent_id).isdigit() else "p"
            # Versión vieja de RLlib (0.8.4) usa compute_action
            action = trainer.compute_action(ob, policy_id=policy_id)
            actions[agent_id] = action

        obs, rew, done, info = env_obj.step(actions)

        for agent_id, r in rew.items():
            total_rewards[agent_id] = total_rewards.get(agent_id, 0.0) + r

        step += 1

    print("\nEpisodio de evaluación finalizado:")
    print(f"  Pasos ejecutados: {step}")
    print("  Recompensa total por agente:")
    for agent_id, r in total_rewards.items():
        print(f"    {agent_id}: {r}")


# -------------------------------------------------------------------
# 6) main()
# -------------------------------------------------------------------
def main():
    # Si ejecutás desde otro lugar, asegurate de estar en el root del repo:
    # os.chdir("RUTA/A/ai-economist")

    # Creamos un env local para consultar espacios y para eval
    env_obj = create_env_for_inspection()

    # Config de PPO + multi-agent
    trainer_config = build_trainer_config(env_obj)

    # Arrancamos Ray
    ray.init(include_dashboard=False, log_to_driver=False)

    # Creamos el trainer PPO con el wrapper como entorno
    trainer = PPOTrainer(
        env=RLlibEnvWrapper,
        config=trainer_config,
    )

    # Entrenamiento
    train(trainer, num_iters=5)

    # Episodio de evaluación para probar el entorno
    run_eval_episode(trainer, env_obj, max_steps=1000)

    # Apagamos Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
