import os
import sys
import ray
import yaml
import argparse
import logging
import warnings
import csv
import torch
from ray.rllib.agents.ppo import PPOTrainer

# Importa el wrapper de entorno tal como en el tutorial
from tutorials.rllib.env_wrapper import RLlibEnvWrapper

# Desactivar dashboard (fuente del error aiohttp.signals)
os.environ["RAY_DISABLE_DASHBOARD"] = "1"

# Silenciar TensorFlow y PyTorch (por si acaso)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Desactivar warnings de deprecación de Gym y otros
warnings.filterwarnings("ignore")

# Configurar logging
logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# Limitar el spam de Ray y RLlib
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.worker").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)


# -------------------------
# FUNCIONES DE CONFIGURACIÓN
# -------------------------

def process_args():
    """Parse los argumentos de línea de comandos y carga la configuración del YAML."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default="phase1",
        help="Path al directorio con config.yaml (default: phase1)"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Número de iteraciones de entrenamiento (override de config.yaml si se proporciona)"
    )
    args = parser.parse_args()
    
    # Cargar el archivo config.yaml
    config_path = os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró config.yaml en: {config_path}")
    
    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)
    
    logger.info(f"Configuración cargada desde: {config_path}")
    
    # Override num_iters si se especifica en línea de comandos
    if args.num_iters is not None:
        run_configuration["general"]["num_iterations"] = args.num_iters
        logger.info(f"Overriding num_iterations a {args.num_iters}")
    
    return run_configuration, args.run_dir


# -------------------------------------------------------------------
# CONFIGURACIÓN DEL ENTORNO (ahora se carga desde YAML)
# -------------------------------------------------------------------

def build_env_config(run_configuration):
    """
    Extrae la configuración del entorno desde el diccionario de YAML.
    """
    env_config = run_configuration.get("env", {})
    
    # Asegurarse de que scenario_name esté presente
    if "scenario_name" not in env_config:
        env_config["scenario_name"] = "layout_from_file/simple_wood_and_stone"
    
    logger.info(f"Configuración del entorno: scenario={env_config['scenario_name']}")
    
    return env_config


# -------------------------------------------------------------------


def create_env_for_inspection(env_config):
    """
    Crea un entorno local (no de RLlib) para inspeccionar spaces
    y para hacer rollouts manuales de evaluación.
    
    Args:
        env_config: Diccionario de configuración del entorno
    """
    env_obj = RLlibEnvWrapper({"env_config_dict": env_config}, verbose=True)
    return env_obj


# -------------------------------------------------------------------
# 2) Definición de políticas multi-agente (agentes + planner)
# -------------------------------------------------------------------
def build_multiagent_policies(env_obj, run_configuration):
    """
    Define las políticas multi-agente para RLlib usando configuración del YAML.
    - "a" = política compartida entre todos los agentes (IDs numéricos)
    - "p" = política del planner (ID "p")
    
    Args:
        env_obj: Entorno para obtener observation_space y action_space
        run_configuration: Diccionario del YAML con las configuraciones
    """
    
    # Extraer configs del YAML
    general_config = run_configuration.get("general", {})
    agent_policy_config = run_configuration.get("agent_policy", {})
    planner_policy_config = run_configuration.get("planner_policy", {})
    
    train_planner = general_config.get("train_planner", False)
    
    # Configuración del modelo para agentes
    agent_model_config = agent_policy_config.get("model", {})
    agent_model = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "use_lstm": agent_model_config.get("use_lstm", False),
        "lstm_cell_size": agent_model_config.get("lstm_cell_size", 128),
        "max_seq_len": agent_model_config.get("max_seq_len", 25),
        "vf_share_layers": False,
    }
    
    # Configuración del modelo para planner
    planner_model_config = planner_policy_config.get("model", {})
    planner_model = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "use_lstm": planner_model_config.get("use_lstm", False),
        "lstm_cell_size": planner_model_config.get("lstm_cell_size", 128),
        "max_seq_len": planner_model_config.get("max_seq_len", 25),
        "vf_share_layers": False,
    }
    
    policies = {
        "a": (
            None,  # usa la política default de RLlib (PPO)
            env_obj.observation_space,
            env_obj.action_space,
            {
                "model": agent_model,
                "gamma": agent_policy_config.get("gamma", 0.998),
                "lr": agent_policy_config.get("lr", 0.0003),
                "vf_loss_coeff": agent_policy_config.get("vf_loss_coeff", 0.05),
                "entropy_coeff": agent_policy_config.get("entropy_coeff", 0.025),
                "clip_param": agent_policy_config.get("clip_param", 0.3),
                "vf_clip_param": agent_policy_config.get("vf_clip_param", 10.0),
                "grad_clip": agent_policy_config.get("grad_clip", 10.0),
            },
        ),
        "p": (
            None,  # política default también
            env_obj.observation_space_pl,
            env_obj.action_space_pl,
            {
                "model": planner_model,
                "gamma": planner_policy_config.get("gamma", 0.998),
                "lr": planner_policy_config.get("lr", 0.0003) if train_planner else 0.0,
                "vf_loss_coeff": planner_policy_config.get("vf_loss_coeff", 0.05),
                "entropy_coeff": planner_policy_config.get("entropy_coeff", 0.025),
                "clip_param": planner_policy_config.get("clip_param", 0.3),
                "vf_clip_param": planner_policy_config.get("vf_clip_param", 10.0),
                "grad_clip": planner_policy_config.get("grad_clip", 10.0),
            },
        ),
    }

    # Mapea ID del agente -> ID de la policy
    #   - agentes: IDs enteros (0,1,2,3,...) -> policy "a"
    #   - planner: ID "p" -> policy "p"
    def policy_mapping_fn(agent_id):
        return "a" if str(agent_id).isdigit() else "p"

    policies_to_train = ["p"] if train_planner else ["a"]
    
    logger.info(f"Políticas configuradas - Train planner: {train_planner}")
    logger.info(f"  - Agent LR: {policies['a'][3]['lr']}")
    logger.info(f"  - Planner LR: {policies['p'][3]['lr']}")
    
    return policies, policy_mapping_fn, policies_to_train


# -------------------------------------------------------------------
# 3) Construcción del config de PPO + entorno
# -------------------------------------------------------------------
def build_trainer_config(env_obj, run_configuration, env_config):
    """
    Construye la configuración completa del PPOTrainer usando valores del YAML.
    
    Args:
        env_obj: Entorno para obtener observation_space y action_space
        run_configuration: Diccionario del YAML con las configuraciones
        env_config: Configuración del entorno
    """
    # Construir políticas multi-agente
    policies, policy_mapping_fn, policies_to_train = build_multiagent_policies(
        env_obj, run_configuration
    )

    # Extraer configuración del trainer del YAML
    trainer_yaml_config = run_configuration.get("trainer", {})
    
    trainer_config = {
        "multiagent": {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": policy_mapping_fn,
        }
    }

    # Parámetros de entrenamiento desde YAML
    # NOTA: Cambiamos batch_mode a "complete_episodes" para que se completen los episodios
    # y podamos ver rewards reales (no nan)
    trainer_config.update(
        {
            "num_workers": trainer_yaml_config.get("num_workers", 2),
            "num_envs_per_worker": trainer_yaml_config.get("num_envs_per_worker", 2),
            "framework": "torch",
            "num_gpus": trainer_yaml_config.get("num_gpus", 0),
            "log_level": "ERROR",
            "train_batch_size": trainer_yaml_config.get("train_batch_size", 2000),
            "sgd_minibatch_size": trainer_yaml_config.get("sgd_minibatch_size", 512),
            "num_sgd_iter": trainer_yaml_config.get("num_sgd_iter", 10),
            "rollout_fragment_length": trainer_yaml_config.get("rollout_fragment_length", 200),
            # Forzar complete_episodes para que se vean los rewards correctamente
            "batch_mode": "complete_episodes",
            "no_done_at_end": False,
        }
    )

    # Config específico del wrapper
    env_wrapper_config = {
        "env_config_dict": env_config,
        "num_envs_per_worker": trainer_config["num_envs_per_worker"],
    }

    trainer_config["env_config"] = env_wrapper_config
    
    logger.info("Configuración del trainer completa:")
    logger.info(f"  - Num workers: {trainer_config['num_workers']}")
    logger.info(f"  - Train batch size: {trainer_config['train_batch_size']}")
    logger.info(f"  - SGD minibatch size: {trainer_config['sgd_minibatch_size']}")
    
    return trainer_config


# -------------------------------------------------------------------
# 4) Entrenamiento PPO
# -------------------------------------------------------------------
def train(trainer, num_iters=5, planner=False):
    """
    Entrena el PPOTrainer por num_iters iteraciones e imprime reward medio.
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

        # Métricas por policy (multi-agent)
        policy_reward_mean = result.get("policy_reward_mean", {})
        policy_reward_min = result.get("policy_reward_min", {})
        policy_reward_max = result.get("policy_reward_max", {})

        a_mean = policy_reward_mean.get("a")
        p_mean = policy_reward_mean.get("p")
        a_min = policy_reward_min.get("a")
        p_min = policy_reward_min.get("p")
        a_max = policy_reward_max.get("a")
        p_max = policy_reward_max.get("p")

    
    #    kl_a = result["info"]['learner']['a']['kl']
    #    entropy_a = result["info"]['learner']['a']['entropy']

        # Descomentar cuando se entrene planner

        #kl_p = result["info"]['learner']['p']['kl']
        #entropy_p = result["info"]['learner']['p']['entropy']

        #print(f"KL Divergence:
        # a={kl_a}, p={kl_p}")
        #print(f"Entropy: a={entropy_a}, p={entropy_p}")

      #  print(f"KL Divergence (policy a): {kl_a}")
      #  print(f"Entropy (policy a): {entropy_a}")

        print(f"episode_reward_mean: {episode_reward_mean}")
        print(f"policy_reward_mean: a={a_mean}, p={p_mean}")

        # Guardamos una fila de info resumida para esta iteración
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
            "policy_p_reward_max": p_max
#            "kl_a": kl_a,
 #           "entropy_a": entropy_a 
            #,  DESCOMENTAR CUANDO SE ENTRENE PLANNER
            #"kl_p": kl_p, 
            #"entropy_p": entropy_p,
            
        }

       
        history.append(row)
        
        # Mostrar info de políticas individuales si está disponible
        if 'policy_reward_mean' in result:
            print(f"  Policy rewards: {result['policy_reward_mean']}")

    
    # Guardar checkpoint al final del entrenamiento en directorio: checkpoints/ como un pkl

    # checkpoint_dir = trainer.save(checkpoint_dir="checkpoints")
    # print(f"\nCheckpoint guardado en: {checkpoint_dir}")

    import torch, os

    os.makedirs("checkpoints", exist_ok=True)


    if planner:

        # guardar pesos de la política 'a'
        torch.save(
            trainer.get_policy("a").model.state_dict(),
            "checkpoints/policy_a_weights_w_planner.pt"
        )

        # si tenés planner 'p', guardalo también
        if "p" in trainer.workers.local_worker().policy_map:
            torch.save(
                trainer.get_policy("p").model.state_dict(),
                "checkpoints/policy_p_weights_w_planner.pt"
        )

    else:
        # guardar pesos de la política 'a'
        torch.save(
            trainer.get_policy("a").model.state_dict(),
            "checkpoints/policy_a_weights.pt"
        )

        # si tenés planner 'p', guardalo también
        if "p" in trainer.workers.local_worker().policy_map:
            torch.save(
                trainer.get_policy("p").model.state_dict(),
                "checkpoints/policy_p_weights.pt"
    )
    return history

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

def save_history_to_csv(history, filepath):
    """
    Guarda el historial de entrenamiento en un archivo CSV.
    
    Args:
        history: Lista de diccionarios con métricas por iteración
        filepath: Ruta del archivo CSV donde guardar
    """
    
    fieldnames = list(history[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
        
    print(f"\nHistorial de entrenamiento guardado en: {filepath}")

# -------------------------------------------------------------------
# 6) main()
# -------------------------------------------------------------------
def main(planner=False):
    """
    Función principal que:
    1. Carga configuración desde YAML
    2. Crea el entorno
    3. Construye el trainer PPO
    4. Entrena por N iteraciones
    5. Ejecuta evaluación
    """
    
    # Parsear argumentos y cargar configuración del YAML
    run_configuration, run_dir = process_args()
    
    logger.info("=" * 70)
    logger.info("Iniciando entrenamiento con AI-Economist")
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 70)
    
    # Extraer configuración del entorno
    env_config = build_env_config(run_configuration)
    
    # Crear un entorno local para consultar espacios y para eval
    env_obj = create_env_for_inspection(env_config)
    logger.info("Entorno creado exitosamente")
    
    # Construir config de PPO + multi-agent desde YAML
    trainer_config = build_trainer_config(env_obj, run_configuration, env_config)
    
    # Arrancamos Ray
    logger.info("Inicializando Ray...")
    ray.init(include_dashboard=False, log_to_driver=False)
    
    # Crear el trainer PPO con el wrapper como entorno
    logger.info("Creando PPOTrainer...")
    trainer = PPOTrainer(
        env=RLlibEnvWrapper,
        config=trainer_config,
    )

    train_planner = run_configuration["general"]["train_planner"]
    restore_agents_path = run_configuration["general"].get("restore_tf_weights_agents", "")
    restore_planner_path = run_configuration["general"].get("restore_tf_weights_planner", "")

    if restore_agents_path and os.path.exists(restore_agents_path):
        logger.info(f"Cargando pesos pre-entrenados de agentes desde: {restore_agents_path}")
        try:
            # Cargar pesos de PyTorch
            state_dict = torch.load(restore_agents_path, map_location='cpu')
            trainer.get_policy("a").model.load_state_dict(state_dict)
            logger.info("Pesos de política 'a' (agentes) cargados exitosamente.")
            
            # Congelar los pesos de los agentes si estamos en Phase 2
            if train_planner:
                for param in trainer.get_policy("a").model.parameters():
                    param.requires_grad = False
                logger.info("Pesos de agentes CONGELADOS (no se entrenarán).")
        except Exception as e:
            logger.error(f"Error al cargar pesos de agentes: {e}")
            logger.warning("Continuando con pesos aleatorios para agentes.")
    else:
        if train_planner:
            logger.warning("Phase 2 activado pero no se encontraron pesos de agentes.")
            logger.warning(f"   Buscando en: {restore_agents_path}")

    if restore_planner_path and os.path.exists(restore_planner_path):
        logger.info(f"Cargando pesos pre-entrenados de planner desde: {restore_planner_path}")
        try:

            state_dict = torch.load(restore_planner_path, map_location='cpu')
            trainer.get_policy("p").model.load_state_dict(state_dict)
            logger.info("Pesos de política 'p' (planner) cargados exitosamente.")
        except Exception as e:
            logger.error(f"Error al cargar pesos de planner: {e}")
            logger.warning("Continuando con pesos aleatorios para planner.")

    
    # Obtener número de iteraciones desde config
    num_iterations = run_configuration.get("general", {}).get("num_iterations", 100)
    logger.info(f"Comenzando entrenamiento por {num_iterations} iteraciones...")
    
    # Entrenamiento
    history = train(trainer, num_iters=num_iterations, planner=planner)

    # Guardar historial a CSV
    if planner:
        csv_path = "ppo_results_with_planner.csv"
    else:
        csv_path = "ppo_results.csv"
    save_history_to_csv(history, csv_path)
    
    # Episodio de evaluación para probar el entorno
    logger.info("\nEjecutando episodio de evaluación...")
    episode_length = env_config.get("episode_length", 1000)
    run_eval_episode(trainer, env_obj, max_steps=episode_length)
    
    # Apagar Ray
    logger.info("Cerrando Ray...")
    ray.shutdown()
    
    logger.info("=" * 70)
    logger.info("Entrenamiento completado exitosamente!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main(planner=True)
