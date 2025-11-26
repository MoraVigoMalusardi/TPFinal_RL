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



import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork 

import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class SafeEnvWrapper(MultiAgentEnv):
    """
    Un envoltorio de seguridad que elimina los datos problemáticos de la observación del Planner.
    Soluciona el error 'Observation outside given space' causado por tipos int64 vs float.
    CORREGIDO: Eliminada referencia a _agent_ids inexistente.
    """
    def __init__(self, config, **kwargs):
        # Extraemos 'verbose' si viene en los argumentos, si no, False
        verbose = kwargs.get("verbose", False)
        
        # Instanciamos el entorno original pasando el verbose
        self.internal_env = RLlibEnvWrapper(config, verbose=verbose)
        
        # 1. Copiamos los espacios del Agente tal cual
        self.observation_space = self.internal_env.observation_space
        self.action_space = self.internal_env.action_space
        
        # 2. Modificamos el espacio del PLANNER para quitar lo que da error
        orig_pl_space = self.internal_env.observation_space_pl
        if hasattr(orig_pl_space, "spaces"):
            new_spaces = {
                k: v for k, v in orig_pl_space.spaces.items() 
                if not (k.startswith('p') and k[1:].isdigit()) # Filtramos p0, p1...
            }
            self.observation_space_pl = gym.spaces.Dict(new_spaces)
        else:
            self.observation_space_pl = orig_pl_space

        self.action_space_pl = self.internal_env.action_space_pl

    def reset(self):
        obs = self.internal_env.reset()
        return self._clean_obs(obs)

    def step(self, actions):
        obs, rew, done, info = self.internal_env.step(actions)
        return self._clean_obs(obs), rew, done, info

    def _clean_obs(self, obs):
        """Borra las keys p0..p3 de la observación del planner 'p'"""
        if 'p' in obs and isinstance(obs['p'], dict):
            keys_to_remove = [k for k in obs['p'].keys() if k.startswith('p') and k[1:].isdigit()]
            for k in keys_to_remove:
                del obs['p'][k]
        return obs

class AI_Economist_CNN_PyTorch(RecurrentNetwork, nn.Module):
    """
    Implementación FINAL: CNN + MLP + LSTM (Todo en uno).
    Evita errores de dimensiones de RLlib manejando el LSTM internamente.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # Heredamos de RecurrentNetwork para decirle a RLlib que tenemos memoria
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. Configuración de tamaños
        self.cell_size = model_config.get("lstm_cell_size", 128)
        self.spatial_key = "world-map"
        
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        print(f"Observation space: {obs_space}")
        spatial_shape = None
        flat_dim = 0

        # 2. Análisis de Inputs
        if hasattr(original_space, "spaces"):
            for key, space in original_space.spaces.items():
                print(f"  - Key: {key}, Shape: {space.shape}")
                if key == self.spatial_key:
                    spatial_shape = space.shape
                    print(f"Detected spatial input shape: {spatial_shape}")
                else:
                    flat_dim += int(np.prod(space.shape))
        else:
            flat_dim = int(np.prod(obs_space.shape))

        # 3. Capas: CNN
        if spatial_shape:
            in_channels = spatial_shape[0] if spatial_shape[0] < spatial_shape[2] else spatial_shape[2]
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            cnn_out_dim = 32 * spatial_shape[1] * spatial_shape[2]
        else:
            self.cnn = None
            cnn_out_dim = 0

        # 4. Capas: Procesador Plano
        self.flat_processor = nn.Sequential(
            nn.Linear(flat_dim, 32),
            nn.ReLU()
        )

        # 5. Fusión (Antes del LSTM)
        concat_dim = cnn_out_dim + 32
        self.fc_pre_lstm = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU()
        )

        # 6. LSTM Interno
        # Input: 256 (Features combinadas) -> Output: cell_size (128)
        self.lstm = nn.LSTM(256, self.cell_size, batch_first=True)

        # 7. Cabezas de Salida (Action y Value)
        # Toman la salida del LSTM (128)
        self.action_branch = nn.Linear(self.cell_size, num_outputs)
        self.value_branch = nn.Linear(self.cell_size, 1)
        
        self._cur_value = None

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        RLlib llama a esto automáticamente, manejando la dimensión de tiempo.
        inputs: [Batch, Time, Features]
        """
        # Como inputs viene "aplanado" por RLlib en un tensor gigante, 
        # necesitamos "des-aplanar" para recuperar la estructura de diccionario si es posible.
        # PERO: RecurrentNetwork recibe un Tensor procesado. 
        # TRUCO: Sobrescribimos forward() completo para tener acceso al input_dict original.
        pass 

    @override(RecurrentNetwork)
    def forward(self, input_dict, state, seq_lens):
        """
        Sobrescribimos forward para procesar CNN frame por frame y luego pasar al LSTM.
        """
        obs = input_dict["obs"]
        
        # --- A. Procesamiento Espacial y Plano (Frame a Frame) ---
        # RLlib nos manda datos colapsados [Batch * Time, ...]
        # Procesamos todo como si fuera un batch gigante de imágenes estáticas.
        
        flat_inputs = []
        cnn_out = None

        # 1. CNN
        if self.cnn is not None and self.spatial_key in obs:
            map_input = obs[self.spatial_key].float()
            if map_input.shape[-1] < map_input.shape[1]: 
                 map_input = map_input.permute(0, 3, 1, 2)
            cnn_out = self.cnn(map_input)

        # 2. Flat
        if hasattr(obs, "items"): 
            for key, val in obs.items():
                if key != self.spatial_key:
                    flat_inputs.append(val.float().flatten(start_dim=1))
            flat_input = torch.cat(flat_inputs, dim=1)
        else:
            flat_input = obs.float()

        flat_out = self.flat_processor(flat_input)

        # 3. Combinar
        if cnn_out is not None:
            combined = torch.cat([cnn_out, flat_out], dim=1)
        else:
            combined = flat_out
            
        # 4. Reducir dimensión a 256
        features = self.fc_pre_lstm(combined) 

        # --- B. Lógica LSTM (Time Dimension) ---
        # Ahora tenemos 'features' de tamaño [Batch*Time, 256].
        # Necesitamos re-estructurarlo a [Batch, Time, 256] para el LSTM.
        
        B = len(seq_lens) # Batch size real (número de secuencias)
        T = features.shape[0] // B # Largo promedio (o máximo)
        
        # Vista [Batch, Time, Features]
        features_time = features.view(B, T, -1)
        
        # Paso por LSTM
        self.lstm.flatten_parameters()
        lstm_out, [h, c] = self.lstm(features_time, (state[0].view(1, B, -1), state[1].view(1, B, -1)))
        
        # Volvemos a aplanar para las capas densas de salida [Batch*Time, 128]
        lstm_out_flat = lstm_out.reshape(-1, self.cell_size)

        # --- C. Salidas Finales ---
        logits = self.action_branch(lstm_out_flat)
        self._cur_value = self.value_branch(lstm_out_flat).squeeze(1)

        # Retornamos logits y el nuevo estado [h, c] aplanado para RLlib
        return logits, [h.squeeze(0), c.squeeze(0)]

    @override(RecurrentNetwork)
    def get_initial_state(self):
        # Estado inicial en ceros [Hidden, Cell]
        return [
            torch.zeros(self.cell_size),
            torch.zeros(self.cell_size)
        ]

    @override(RecurrentNetwork)
    def value_function(self):
        return self._cur_value

# Registrar el modelo
ModelCatalog.register_custom_model("paper_cnn_torch", AI_Economist_CNN_PyTorch)

# -------------------------------------------------------------------
# Configuración global de entorno y logging
# -------------------------------------------------------------------

os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("train_agents")
logger.setLevel(logging.INFO)

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
    run_dir : str
    restore_checkpoint : str or None
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
    parser.add_argument(
        "--restore-checkpoint",
        type=str,
        default=None,
        help="Ruta a un checkpoint completo de RLlib para reanudar entrenamiento.",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró config.yaml en: {config_path}")
    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    logger.info(f"Configuración cargada desde: {config_path}")

    if args.num_iters is not None:
        run_configuration.setdefault("general", {})
        run_configuration["general"]["num_iterations"] = args.num_iters
        logger.info(f"Overriding general.num_iterations a {args.num_iters}")

    # Devolvemos también el path del checkpoint (puede ser None)
    return run_configuration, args.run_dir, args.restore_checkpoint


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
    env_obj = SafeEnvWrapper({"env_config_dict": env_config}, verbose=True)
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
    agent_model = {
        # Usamos nuestra clase nueva
        "custom_model": "paper_cnn_torch",
        
        "use_lstm": False,      # Ponemos False para que RLlib no agregue OTRO LSTM encima
        "lstm_cell_size": 128,  # Esto se pasará a nuestro modelo custom
        "max_seq_len": 50,
    }

    # Modelo para planner (en fase 1 típicamente no se usa/entrena)
    planner_model = {
        "custom_model": "paper_cnn_torch",

        "use_lstm": False,        # Usamos el LSTM manual interno
        "lstm_cell_size": 256,    
        "max_seq_len": 50,
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
    Ejecuta el lazo de entrenamiento de PPO y guarda:
    - CSV (se maneja afuera)
    - Pesos de policies (state_dict)
    - Checkpoint completo de RLlib para poder reanudar
    """
    history = []

    for it in range(num_iters):
        print(f"\n********** Iteración: {it} **********")
        result = trainer.train()

        episode_reward_mean = result.get("episode_reward_mean")
        episode_reward_min = result.get("episode_reward_min")
        episode_reward_max = result.get("episode_reward_max")
        episode_len_mean = result.get("episode_len_mean")
        episodes_this_iter = result.get("episodes_this_iter")
        episodes_total = result.get("episodes_total")
        timesteps_total = result.get("timesteps_total")
        training_iteration = result.get("training_iteration")

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

    # ==== Guardar pesos de policies (state_dict) ====
    import torch

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/nuevo_sin_lstm", exist_ok=True)
    torch.save(
        trainer.get_policy("a").model.state_dict(),
        "checkpoints/nuevo_sin_lstm/policy_a_weights.pt",
    )

    if "p" in trainer.workers.local_worker().policy_map:
        torch.save(
            trainer.get_policy("p").model.state_dict(),
            "checkpoints/nuevo_sin_lstm/policy_p_weights.pt",
        )

    # ==== Guardar checkpoint completo de RLlib ====
    checkpoint_root = os.path.join("checkpoints", "rllib_full")
    os.makedirs(checkpoint_root, exist_ok=True)
    checkpoint_path = trainer.save(checkpoint_root)
    logger.info(f"Checkpoint RLlib completo guardado en: {checkpoint_path}")

    return history, checkpoint_path


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
    Si NO se pasa --restore-checkpoint -> arranca desde cero.
    Si SÍ se pasa --restore-checkpoint -> restaura y sigue entrenando.
    """
    run_configuration, run_dir, restore_checkpoint = process_args()

    logger.info("=" * 70)
    logger.info("Iniciando entrenamiento FASE 1 (AGENTES) con AI-Economist")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Checkpoint a restaurar: {restore_checkpoint}")
    
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
        env=SafeEnvWrapper,
        config=trainer_config,
        logger_creator=logger_creator,
    )

    print(f"\nTensorBoard logs se están guardando en: {trainer.logdir}\n")

    # === AGREGAR ESTO PARA VER LA ARQUITECTURA ===
    print("\n" + "="*50)
    print("ARQUITECTURA DEL AGENTE (Policy 'a')")
    print("="*50)
    # Obtenemos el modelo de la política 'a' (agentes)
    model_a = trainer.get_policy("a").model
    print(model_a)
    print("="*50 + "\n")
    # =============================================

    # ==== RESTAURAR CHECKPOINT COMPLETO (OPCIONAL) ====
    if restore_checkpoint is not None:
        if os.path.exists(restore_checkpoint):
            logger.info(f"Restaurando trainer desde checkpoint: {restore_checkpoint}")
            trainer.restore(restore_checkpoint)
        else:
            logger.warning(f"No se encontró el checkpoint: {restore_checkpoint}. Se entrenará desde cero.")

    num_iterations = run_configuration.get("general", {}).get("num_iterations", 100)
    logger.info(f"Comenzando entrenamiento por {num_iterations} iteraciones...")

    history, last_checkpoint = train(trainer, num_iters=num_iterations)
    logger.info(f"Último checkpoint RLlib: {last_checkpoint}")

    # CSV
    os.makedirs("nuevo_sin_lstm", exist_ok=True)
    csv_path = "nuevo_sin_lstm/ppo_results_agents.csv"
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