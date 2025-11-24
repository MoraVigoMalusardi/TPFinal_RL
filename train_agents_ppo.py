import os
import sys
import csv
import ray
import yaml
import argparse
import logging
import warnings
from ray.tune.logger import UnifiedLogger

from ray.rllib.agents.ppo import PPOTrainer
from tutorials.rllib.env_wrapper import RLlibEnvWrapper

# -------------------------------------------------------------------
# Configuración global de entorno y logging
# -------------------------------------------------------------------

# Desactivar dashboard de Ray (evita errores de aiohttp.signals)
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
# Silenciar logs de TF (por si estuviera instalado)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Ignorar warnings de librerías (Gym, etc.)
warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("train_agents")
logger.setLevel(logging.INFO)

# Reducir verbosidad de Ray / RLlib / Gym
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.worker").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)


# -------------------------------------------------------------------
# 1) Carga de argumentos y configuración desde YAML
# -------------------------------------------------------------------

def process_args():
    """
    Parsea argumentos de línea de comandos y carga el config.yaml asociado.

    Returns
    -------
    run_configuration : dict
        Diccionario con toda la configuración cargada desde config.yaml.
    run_dir : str
        Directorio base donde se encuentra config.yaml.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default="phase1",
        help="Directorio que contiene config.yaml (por defecto: phase1).",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Número de iteraciones de entrenamiento (override de config.yaml si se especifica).",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró config.yaml en: {config_path}")

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    logger.info(f"Configuración cargada desde: {config_path}")

    # Permitir override desde CLI del número de iteraciones
    if args.num_iters is not None:
        run_configuration.setdefault("general", {})
        run_configuration["general"]["num_iterations"] = args.num_iters
        logger.info(f"Overriding general.num_iterations a {args.num_iters}")

    return run_configuration, args.run_dir


# -------------------------------------------------------------------
# 2) Configuración del entorno
# -------------------------------------------------------------------

def build_env_config(run_configuration):
    """
    Extrae la configuración del entorno desde el YAML.

    Parameters
    ----------
    run_configuration : dict
        Configuración completa cargada desde config.yaml.

    Returns
    -------
    env_config : dict
        Diccionario con la configuración del entorno para el wrapper.
    """
    env_config = run_configuration.get("env", {}).copy()

    # Valor por defecto del escenario si no se especifica
    env_config.setdefault("scenario_name", "layout_from_file/simple_wood_and_stone")

    logger.info(f"Configuración del entorno: scenario={env_config['scenario_name']}")
    return env_config


def create_env_for_inspection(env_config):
    """
    Crea un entorno local (no controlado por los workers de Ray) para:
    - Consultar observation_space / action_space.
    - Realizar rollouts de evaluación al final.

    Parameters
    ----------
    env_config : dict
        Configuración del entorno.

    Returns
    -------
    env_obj : RLlibEnvWrapper
        Instancia del wrapper del entorno.
    """
    env_obj = RLlibEnvWrapper({"env_config_dict": env_config}, verbose=True)
    return env_obj


# -------------------------------------------------------------------
# 3) Definición de políticas multi-agente (agentes + planner)
# -------------------------------------------------------------------

def build_multiagent_policies(env_obj, run_configuration):
    """
    Define las políticas multi-agente para RLlib usando la configuración del YAML.

    Convenciones:
    - Policy "a": política compartida por todos los agentes (IDs numéricos 0,1,2,...).
    - Policy "p": política del planner (ID 'p').
      En Fase 1 usualmente NO se entrena el planner.

    Parameters
    ----------
    env_obj : RLlibEnvWrapper
        Entorno ya inicializado, usado para obtener spaces.
    run_configuration : dict
        Configuración completa cargada desde YAML.

    Returns
    -------
    policies : dict
        Diccionario con la definición de las políticas para RLlib.
    policy_mapping_fn : callable
        Función que mapea agent_id -> policy_id.
    policies_to_train : list[str]
        Lista de IDs de políticas que se entrenarán.
    """
    general_config = run_configuration.get("general", {})
    agent_policy_config = run_configuration.get("agent_policy", {})
    planner_policy_config = run_configuration.get("planner_policy", {})

    # En fase 1 lo natural es train_planner = False, solo se entrena "a"
    train_planner = general_config.get("train_planner", False)

    # Modelo para agentes
    agent_model_cfg = agent_policy_config.get("model", {})
    agent_model = {
        "fcnet_hiddens": agent_model_cfg.get("fcnet_hiddens", [256, 256]),  # Usamos 2 capas fully-connected de 256 unidades con tanh
        "fcnet_activation": agent_model_cfg.get("fcnet_activation", "tanh"),
        "use_lstm": agent_model_cfg.get("use_lstm", False), 
        "vf_share_layers": agent_model_cfg.get("vf_share_layers", False),
    }

    # Modelo para planner (en fase 1 típicamente no se usa/entrena)
    planner_model_cfg = planner_policy_config.get("model", {})
    planner_model = {
        "fcnet_hiddens": planner_model_cfg.get("fcnet_hiddens", [256, 256]),  # Usamos 2 capas fully-connected de 256 unidades con tanh
        "fcnet_activation": planner_model_cfg.get("fcnet_activation", "tanh"),
        "use_lstm": planner_model_cfg.get("use_lstm", False), 
        "vf_share_layers": planner_model_cfg.get("vf_share_layers", False),
    }

    logger.info("Agent uses lstm: " + str(agent_model["use_lstm"]))
    logger.info("Planner uses lstm: " + str(planner_model["use_lstm"]))

    policies = {
        "a": (
            None,
            env_obj.observation_space,
            env_obj.action_space,
            {
                "model": agent_model,
                "gamma": agent_policy_config.get("gamma", 0.998),
                "lr": agent_policy_config.get("lr", 0.0003),
                "vf_loss_coeff": agent_policy_config.get("vf_loss_coeff", 0.05),
                "entropy_coeff": agent_policy_config.get("entropy_coeff", 0.025),
                "clip_param": agent_policy_config.get("clip_param", 0.3),
                "vf_clip_param": agent_policy_config.get("vf_clip_param", 50.0),
                "grad_clip": agent_policy_config.get("grad_clip", 10.0),
                "lambda": agent_policy_config.get("lambda", 0.98),
                "use_gae": agent_policy_config.get("use_gae", True),
            },
        ),
        "p": (
            None,
            env_obj.observation_space_pl,
            env_obj.action_space_pl,
            {
                "model": planner_model,
                "gamma": planner_policy_config.get("gamma", 0.998),
                "lr": planner_policy_config.get("lr", 0.0 if not train_planner else 0.0001),
                "vf_loss_coeff": planner_policy_config.get("vf_loss_coeff", 0.05),
                "entropy_coeff": planner_policy_config.get("entropy_coeff", 0.1),
                "clip_param": planner_policy_config.get("clip_param", 0.3),
                "vf_clip_param": planner_policy_config.get("vf_clip_param", 50.0),
                "grad_clip": planner_policy_config.get("grad_clip", 10.0),
                "lambda": planner_policy_config.get("lambda", 0.98),
                "use_gae": planner_policy_config.get("use_gae", True),
            },
        ),
    }

    def policy_mapping_fn(agent_id):
        """
        Mapea IDs de agentes del entorno a IDs de políticas:
        - IDs numéricos (0,1,2,3,...)  -> policy "a" (agentes).
        - ID "p"                        -> policy "p" (planner).
        """
        return "a" if str(agent_id).isdigit() else "p"

    # En fase 1 queremos entrenar solo "a" (salvo que explícitamente se diga lo contrario)
    policies_to_train = ["a"] if not train_planner else ["a", "p"]

    logger.info(f"Políticas configuradas - Train planner: {train_planner}")
    logger.info(f"  - LR agentes (a): {policies['a'][3]['lr']}")
    logger.info(f"  - LR planner (p): {policies['p'][3]['lr']}")

    return policies, policy_mapping_fn, policies_to_train


# -------------------------------------------------------------------
# 4) Construcción del config del PPOTrainer
# -------------------------------------------------------------------

def build_trainer_config(env_obj, run_configuration, env_config):
    """
    Construye el diccionario de configuración para PPOTrainer.

    Parameters
    ----------
    env_obj : RLlibEnvWrapper
        Entorno para leer spaces.
    run_configuration : dict
        Configuración completa cargada desde YAML.
    env_config : dict
        Configuración del entorno para pasar al wrapper.

    Returns
    -------
    trainer_config : dict
        Diccionario listo para pasar al constructor de PPOTrainer.
    """
    policies, policy_mapping_fn, policies_to_train = build_multiagent_policies(
        env_obj, run_configuration
    )

    trainer_yaml_config = run_configuration.get("trainer", {})

    trainer_config = {
        "multiagent": {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "num_workers": trainer_yaml_config.get("num_workers", 12),
        "num_envs_per_worker": trainer_yaml_config.get("num_envs_per_worker", 2),
        "framework": "torch",
        "num_gpus": trainer_yaml_config.get("num_gpus", 0),
        "log_level": "ERROR",
        "train_batch_size": trainer_yaml_config.get("train_batch_size", 4800),
        "sgd_minibatch_size": trainer_yaml_config.get("sgd_minibatch_size", 512),
        "num_sgd_iter": trainer_yaml_config.get("num_sgd_iter", 10),
        "rollout_fragment_length": trainer_yaml_config.get("rollout_fragment_length", 200),
        "batch_mode": trainer_yaml_config.get("batch_mode", "truncate_episodes"),
        "no_done_at_end": trainer_yaml_config.get("no_done_at_end", False),
    }

    # Config específico del wrapper de AI-Economist
    env_wrapper_config = {
        "env_config_dict": env_config,
        "num_envs_per_worker": trainer_config["num_envs_per_worker"],
    }
    trainer_config["env_config"] = env_wrapper_config

    logger.info("Configuración del trainer PPO:")
    logger.info(f"  - Num workers: {trainer_config['num_workers']}")
    logger.info(f"  - Train batch size: {trainer_config['train_batch_size']}")
    logger.info(f"  - SGD minibatch size: {trainer_config['sgd_minibatch_size']}")

    return trainer_config

def create_tb_logger_creator(run_dir):
    """
    Crea un logger_creator para que RLlib escriba logs de TensorBoard
    dentro de run_dir/tb_logs.
    """
    def logger_creator(config):
        logdir = os.path.join(run_dir, "tb_logs")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


# -------------------------------------------------------------------
# 5) Bucle de entrenamiento
# -------------------------------------------------------------------

def train(trainer, num_iters=5):
    """
    Ejecuta el lazo de entrenamiento de PPO.

    Parameters
    ----------
    trainer : PPOTrainer
        Instancia ya configurada de PPOTrainer.
    num_iters : int
        Número de iteraciones de entrenamiento (calls a trainer.train()).

    Returns
    -------
    history : list[dict]
        Lista con métricas resumidas por iteración de entrenamiento.
    """
    history = []

    for it in range(num_iters):
        print(f"\n********** Iteración: {it} **********")
        result = trainer.train()

        # Métricas globales por episodio
        episode_reward_mean = result.get("episode_reward_mean")
        episode_reward_min = result.get("episode_reward_min")
        episode_reward_max = result.get("episode_reward_max")
        episode_len_mean = result.get("episode_len_mean")
        episodes_this_iter = result.get("episodes_this_iter")
        episodes_total = result.get("episodes_total")
        timesteps_total = result.get("timesteps_total")
        training_iteration = result.get("training_iteration")

        # Métricas por política (multi-agente)
        policy_reward_mean = result.get("policy_reward_mean", {})
        policy_reward_min = result.get("policy_reward_min", {})
        policy_reward_max = result.get("policy_reward_max", {})

        a_mean = policy_reward_mean.get("a")
        p_mean = policy_reward_mean.get("p")
        a_min = policy_reward_min.get("a")
        p_min = policy_reward_min.get("p")
        a_max = policy_reward_max.get("a")
        p_max = policy_reward_max.get("p")

        print(f"episode_reward_mean: {episode_reward_mean}")
        print(f"policy_reward_mean: a={a_mean}, p={p_mean}")

        row = {
            "iteration": training_iteration if training_iteration is not None else it,
            "timesteps_total": timesteps_total,
            "episodes_total": episodes_total,
            "episodes_this_iter": episodes_this_iter,
            "episode_reward_min": episode_reward_min,
            "episode_reward_max": episode_reward_max,
            "episode_reward_mean": episode_reward_mean,
            "episode_len_mean": episode_len_mean,
            "policy_a_reward_mean": a_mean,
            "policy_a_reward_min": a_min,
            "policy_a_reward_max": a_max,
            "policy_p_reward_mean": p_mean,
            "policy_p_reward_min": p_min,
            "policy_p_reward_max": p_max,
        }
        history.append(row)

        if "policy_reward_mean" in result:
            print(f"  Policy rewards: {result['policy_reward_mean']}")

    # Guardar pesos de la policy 'a' al final de Fase 1
    import torch  # import local para no contaminar namespace global si no se usa

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        trainer.get_policy("a").model.state_dict(),
        "checkpoints/nuevo_sin_lstm/policy_a_weights.pt",
    )

    # Guardar también planner (aunque en Fase 1 típicamente no se entrena)
    if "p" in trainer.workers.local_worker().policy_map:
        torch.save(
            trainer.get_policy("p").model.state_dict(),
            "checkpoints/nuevo_sin_lstm/policy_p_weights.pt",
        )

    return history


# -------------------------------------------------------------------
# 6) Rollout de evaluación
# -------------------------------------------------------------------

def run_eval_episode(trainer, env_obj, max_steps=200):
    """
    Ejecuta un episodio de evaluación usando el entorno local (no Ray workers).

    Parameters
    ----------
    trainer : PPOTrainer
        Trainer ya entrenado, con políticas cargadas.
    env_obj : RLlibEnvWrapper
        Entorno local para ejecutar el rollout.
    max_steps : int
        Número máximo de pasos en el episodio.
    """
    obs = env_obj.reset()
    done = {"__all__": False}
    total_rewards = {agent_id: 0.0 for agent_id in obs.keys()}

    step = 0
    while not done["__all__"] and step < max_steps:
        actions = {}
        for agent_id, ob in obs.items():
            policy_id = "a" if str(agent_id).isdigit() else "p"
            action = trainer.compute_action(ob, policy_id=policy_id)
            actions[agent_id] = action

        obs, rew, done, _info = env_obj.step(actions)

        for agent_id, r in rew.items():
            total_rewards[agent_id] = total_rewards.get(agent_id, 0.0) + r

        step += 1

    print("\nEpisodio de evaluación finalizado:")
    print(f"  Pasos ejecutados: {step}")
    print("  Recompensa total por agente:")
    for agent_id, r in total_rewards.items():
        print(f"    {agent_id}: {r}")


# -------------------------------------------------------------------
# 7) Guardar historial a CSV
# -------------------------------------------------------------------

def save_history_to_csv(history, filepath):
    """
    Guarda el historial de entrenamiento en un archivo CSV.

    Parameters
    ----------
    history : list[dict]
        Lista de filas con métricas por iteración.
    filepath : str
        Ruta de salida para el CSV.
    """
    if not history:
        logger.warning("Historial vacío, no se guardará CSV.")
        return

    fieldnames = list(history[0].keys())
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print(f"\nHistorial de entrenamiento guardado en: {filepath}")


# -------------------------------------------------------------------
# 8) main()
# -------------------------------------------------------------------

def main():
    """
    Pipeline de Fase 1 (entrenamiento de agentes):
    1. Carga configuración desde YAML.
    2. Crea el entorno.
    3. Construye el PPOTrainer multi-agente.
    4. Entrena por N iteraciones.
    5. Guarda pesos de agentes y CSV de métricas.
    6. Ejecuta un episodio de evaluación.
    """
    run_configuration, run_dir = process_args()

    logger.info("=" * 70)
    logger.info("Iniciando entrenamiento FASE 1 (AGENTES) con AI-Economist")
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 70)

    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)

    trainer_config = build_trainer_config(env_obj, run_configuration, env_config)

    logger.info("Inicializando Ray...")
    ray.init(include_dashboard=False, log_to_driver=False)

    # Logger de TensorBoard
    logger_creator = create_tb_logger_creator(run_dir)

    logger.info("Creando PPOTrainer (con TensorBoard)...")
    trainer = PPOTrainer(
        env=RLlibEnvWrapper,
        config=trainer_config,
        logger_creator=logger_creator,
    )

    print(f"\nTensorBoard logs se están guardando en: {trainer.logdir}\n")


    policy_a = trainer.get_policy("a")
    print("\n=== Arquitectura completa de la policy 'a' ===")
    print(policy_a.model)
    print("=============================================\n")


    num_iterations = run_configuration.get("general", {}).get("num_iterations", 100)
    logger.info(f"Comenzando entrenamiento por {num_iterations} iteraciones...")

    history = train(trainer, num_iters=num_iterations)

    csv_path = os.path.join(run_dir, "nuevo_sin_lstm/ppo_results_agents.csv")
    save_history_to_csv(history, csv_path)

    logger.info("\nEjecutando episodio de evaluación...")
    episode_length = env_config.get("episode_length", 1000)
    run_eval_episode(trainer, env_obj, max_steps=episode_length)

    logger.info("Cerrando Ray...")
    ray.shutdown()

    logger.info("=" * 70)
    logger.info("Entrenamiento FASE 1 completado exitosamente!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
