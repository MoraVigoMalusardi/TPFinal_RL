import os
import sys
import csv
import ray
import yaml
import torch
import argparse
import logging
import warnings
import gym
import numpy as np
import torch.nn as nn

from ray.tune.logger import UnifiedLogger
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from tutorials.rllib.env_wrapper import RLlibEnvWrapper

# -------------------------------------------------------------------
# Configuración global
# -------------------------------------------------------------------

os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("train_planner_cnn")
logger.setLevel(logging.INFO)

logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.worker").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)


# -------------------------------------------------------------------
# 1) Wrapper de Seguridad (SafeEnvWrapper)
# -------------------------------------------------------------------
class SafeEnvWrapper(MultiAgentEnv):
    """
    Envuelve el entorno para asegurar que las observaciones sean compatibles
    con el modelo CNN y elimina claves conflictivas del Planner.
    """
    def __init__(self, config, **kwargs):
        verbose = kwargs.get("verbose", False)
        self.internal_env = RLlibEnvWrapper(config, verbose=verbose)
        
        self.observation_space = self.internal_env.observation_space
        self.action_space = self.internal_env.action_space
        
        # Limpieza del espacio del Planner (eliminar p0, p1...)
        orig_pl_space = self.internal_env.observation_space_pl
        if hasattr(orig_pl_space, "spaces"):
            new_spaces = {
                k: v for k, v in orig_pl_space.spaces.items() 
                if not (k.startswith('p') and k[1:].isdigit())
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
        # Limpiar observaciones del planner
        if 'p' in obs and isinstance(obs['p'], dict):
            keys_to_remove = [k for k in obs['p'].keys() if k.startswith('p') and k[1:].isdigit()]
            for k in keys_to_remove:
                del obs['p'][k]
        return obs


# -------------------------------------------------------------------
# 2) Modelo Híbrido: CNN (Mapa) + MLP (Stats)
# -------------------------------------------------------------------
class AI_Economist_CNN_PyTorch(TorchModelV2, nn.Module):
    """
    Arquitectura CNN+MLP para agentes y planner.
    - Entrada: mapa 2D (world-map) + features planos (inventario, etc.)
    - Salida: logits de acción + valor (con ramas separadas actor/crítico)
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.spatial_key = "world-map"
        
        # --- Detección de Inputs ---
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        spatial_shape = None
        flat_dim = 0

        if hasattr(original_space, "spaces"):
            for key, space in original_space.spaces.items():
                if key == self.spatial_key:
                    spatial_shape = space.shape
                elif key == "action_mask":
                    # NO la metemos en el vector plano, se usa solo para enmascarar logits
                    continue
                else:
                    flat_dim += int(np.prod(space.shape))
        else:
            flat_dim = int(np.prod(obs_space.shape))

        # --- 1. Rama CNN (Visión) ---
        if spatial_shape:
            in_channels = spatial_shape[0] if spatial_shape[0] < spatial_shape[2] else spatial_shape[2]
            
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            dummy_shape = (1, in_channels, spatial_shape[1], spatial_shape[2])
            dummy_input = torch.zeros(dummy_shape)
            with torch.no_grad():
                cnn_out_dim = self.cnn(dummy_input).shape[1]
                
            logger.info(f"CNN inicializada. Stride=2. Salida plana: {cnn_out_dim}")
        else:
            self.cnn = None
            cnn_out_dim = 0

        # --- 2. Rama Flat ---
        self.flat_processor = nn.Sequential(
            nn.Linear(flat_dim, flat_dim),
            nn.Tanh()
        )

        concat_dim = cnn_out_dim + flat_dim
        
        # --- 3. RAMAS GEMELAS ---
        
        # RAMA ACTOR
        self.actor_layers = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.action_head = nn.Linear(256, num_outputs)

        # RAMA CRÍTICO
        self.critic_layers = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.value_head = nn.Linear(256, 1)
        
        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        flat_parts = []
        cnn_out = None

        # A. Procesar Mapa
        if self.cnn is not None:
            if isinstance(obs, dict) and self.spatial_key in obs:
                map_input = obs[self.spatial_key].float()
                # [B, H, W, C] -> [B, C, H, W] si hace falta
                if map_input.shape[-1] < map_input.shape[1]:
                    map_input = map_input.permute(0, 3, 1, 2)
                cnn_out = self.cnn(map_input)

        # B. Procesar Vector Plano
        if isinstance(obs, dict):
            for key, val in obs.items():
                if key != self.spatial_key and key != "action_mask":
                    flat_parts.append(val.float().reshape(val.shape[0], -1))
            
            if flat_parts:
                flat_input = torch.cat(flat_parts, dim=1)
            else:
                if cnn_out is not None:
                    dev = cnn_out.device
                else:
                    dev = torch.device("cpu")
                flat_input = torch.zeros(obs[list(obs.keys())[0]].shape[0], 0).to(dev)
        else:
            flat_input = obs.float()

        flat_out = self.flat_processor(flat_input)

        # C. Fusionar
        if cnn_out is not None:
            features = torch.cat([cnn_out, flat_out], dim=1)
        else:
            features = flat_out
            
        # D. Caminos Separados
        
        # Actor
        actor_out = self.actor_layers(features)
        logits = self.action_head(actor_out)
        
        # Crítico
        critic_out = self.critic_layers(features)
        self._cur_value = self.value_head(critic_out).squeeze(1)

        # Aplicar Action Mask si existe
        if isinstance(obs, dict) and "action_mask" in obs:
            inf_mask = torch.clamp(torch.log(obs["action_mask"]), min=-1e10)
            logits = logits + inf_mask

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._cur_value


# -------------------------------------------------------------------
# 3) Argumentos y configuración
# -------------------------------------------------------------------

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default="phase2",
        help="Directorio que contiene config.yaml para Fase 2 (por defecto: phase2).",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Override de general.num_iterations en config.yaml.",
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

    return run_configuration, args.run_dir, args.restore_checkpoint


# -------------------------------------------------------------------
# 4) Config del entorno
# -------------------------------------------------------------------

def build_env_config(run_configuration):
    env_config = run_configuration.get("env", {}).copy()
    env_config.setdefault("scenario_name", "layout_from_file/simple_wood_and_stone")

    # Importante para CNN:
    env_config["flatten_observations"] = False
    env_config["flatten_masks"] = True

    logger.info(f"Env Config - Scenario: {env_config['scenario_name']}")
    logger.info("Flatten Observations set to FALSE for CNN compatibility.")
    return env_config


def create_env_for_inspection(env_config):
    # Usamos SafeEnvWrapper también para inspección
    return SafeEnvWrapper({"env_config_dict": env_config}, verbose=True)


# -------------------------------------------------------------------
# 5) Políticas multi-agente (agentes + planner)
# -------------------------------------------------------------------

def build_multiagent_policies(env_obj, run_configuration):
    general_config = run_configuration.get("general", {})
    agent_policy_config = run_configuration.get("agent_policy", {})
    planner_policy_config = run_configuration.get("planner_policy", {})

    # En fase 2 queremos entrenar planner y agentes => True
    train_planner = general_config.get("train_planner", True)

    # Modelo CNN para agentes
    agent_model = {
        "custom_model": "paper_cnn_torch",
        "custom_model_config": {},
        "use_lstm": False,
        "vf_share_layers": False,
    }

    # Modelo CNN para planner
    planner_model = {
        "custom_model": "paper_cnn_torch",
        "custom_model_config": {},
        "use_lstm": False,
        "vf_share_layers": False,
    }

    logger.info("Agent CNN uses lstm: " + str(agent_model["use_lstm"]))
    logger.info("Planner CNN uses lstm: " + str(planner_model["use_lstm"]))

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
        # IDs numéricos -> "a", planner -> "p"
        return "a" if str(agent_id).isdigit() else "p"

    # Si train_planner=True, entrenamos ambos ("p" y "a")
    policies_to_train = ["p", "a"] if train_planner else ["a"]

    logger.info(f"Políticas configuradas - Train planner: {train_planner}")
    logger.info(f"  - LR agentes (a): {policies['a'][3]['lr']}")
    logger.info(f"  - LR planner (p): {policies['p'][3]['lr']}")

    return policies, policy_mapping_fn, policies_to_train


# -------------------------------------------------------------------
# 6) Config del PPOTrainer
# -------------------------------------------------------------------

def build_trainer_config(env_obj, run_configuration, env_config):
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

    env_wrapper_config = {
        "env_config_dict": env_config,
        "num_envs_per_worker": trainer_config["num_envs_per_worker"],
    }
    trainer_config["env_config"] = env_wrapper_config

    logger.info("Configuración del trainer PPO (Fase 2 CNN):")
    logger.info(f"  - Num workers: {trainer_config['num_workers']}")
    logger.info(f"  - Train batch size: {trainer_config['train_batch_size']}")
    logger.info(f"  - SGD minibatch size: {trainer_config['sgd_minibatch_size']}")

    return trainer_config


def create_tb_logger_creator(run_dir):
    def logger_creator(config):
        logdir = os.path.join(run_dir, "tb_logs_planner_cnn")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=None)
    return logger_creator


# -------------------------------------------------------------------
# 7) Bucle de entrenamiento
# -------------------------------------------------------------------

def train(trainer, num_iters=5, planner=True):
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

    # ---- Guardar pesos al final (state_dicts) ----
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/nuevo_cnn_planner", exist_ok=True)

    if planner:
        torch.save(
            trainer.get_policy("a").model.state_dict(),
            "checkpoints/nuevo_cnn_planner/policy_a_cnn_weights_w_planner.pt",
        )
        if "p" in trainer.workers.local_worker().policy_map:
            torch.save(
                trainer.get_policy("p").model.state_dict(),
                "checkpoints/nuevo_cnn_planner/policy_p_cnn_weights_w_planner.pt",
            )
    else:
        torch.save(
            trainer.get_policy("a").model.state_dict(),
            "checkpoints/nuevo_cnn_planner/policy_a_cnn_weights.pt",
        )
        if "p" in trainer.workers.local_worker().policy_map:
            torch.save(
                trainer.get_policy("p").model.state_dict(),
                "checkpoints/nuevo_cnn_planner/policy_p_cnn_weights.pt",
            )

    # ---- Guardar checkpoint completo de RLlib ----
    checkpoint_root = os.path.join("checkpoints", "rllib_cnn_planner_full")
    os.makedirs(checkpoint_root, exist_ok=True)
    checkpoint_path = trainer.save(checkpoint_root)
    logger.info(f"Checkpoint RLlib completo (planner+agentes CNN) guardado en: {checkpoint_path}")

    return history, checkpoint_path


# -------------------------------------------------------------------
# 8) Rollout de evaluación
# -------------------------------------------------------------------

def run_eval_episode(trainer, env_obj, max_steps=200):
    obs = env_obj.reset()
    done = {"__all__": False}
    total_rewards = {agent_id: 0.0 for agent_id in obs.keys()}

    step = 0
    while not done["__all__"] and step < max_steps:
        actions = {}
        for agent_id, ob in obs.items():
            policy_id = "a" if str(agent_id).isdigit() else "p"
            # Sin LSTM, state vacío
            action, _, _ = trainer.compute_action(
                ob,
                state=[],
                policy_id=policy_id,
                full_fetch=True,
            )
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
# 9) Guardar historial a CSV
# -------------------------------------------------------------------

def save_history_to_csv(history, filepath):
    if not history:
        logger.warning("Historial vacío, no se guardará CSV.")
        return

    fieldnames = list(history[0].keys())
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print(f"\nHistorial de entrenamiento guardado en: {filepath}")


# -------------------------------------------------------------------
# 10) main() - Fase 2 con CNN
# -------------------------------------------------------------------

def main(planner=True):
    run_configuration, run_dir, restore_checkpoint = process_args()

    logger.info("=" * 70)
    logger.info("Iniciando entrenamiento FASE 2 (PLANNER + AGENTES) con CNN")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Checkpoint a restaurar: {restore_checkpoint}")
    logger.info("=" * 70)

    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, run_configuration, env_config)

    logger.info("Inicializando Ray...")
    ray.init(include_dashboard=False, log_to_driver=False)
    logger_creator = create_tb_logger_creator(run_dir)

    # Registrar modelo CNN
    ModelCatalog.register_custom_model("paper_cnn_torch", AI_Economist_CNN_PyTorch)
    logger.info("Modelo Custom 'paper_cnn_torch' registrado.")

    logger.info("Creando PPOTrainer (CNN)...")
    trainer = PPOTrainer(
        env=SafeEnvWrapper,
        config=trainer_config,
        logger_creator=logger_creator,
    )
    print(f"\nTensorBoard logs se están guardando en: {trainer.logdir}\n")

    general_cfg = run_configuration.get("general", {})
    train_planner_flag = general_cfg.get("train_planner", True)

    # A) Si hay checkpoint completo, reanudar desde ahí
    if restore_checkpoint is not None and os.path.exists(restore_checkpoint):
        logger.info(f"Restaurando trainer desde checkpoint: {restore_checkpoint}")
        trainer.restore(restore_checkpoint)

    else:
        # B) Si NO hay checkpoint, arrancar Fase 2 cargando pesos previos
        restore_agents_path = general_cfg.get("restore_tf_weights_agents", "")
        restore_planner_path = general_cfg.get("restore_tf_weights_planner", "")

        # Cargar pesos de agentes entrenados en Fase 1 (CNN)
        if restore_agents_path and os.path.exists(restore_agents_path):
            logger.info(f"Cargando pesos pre-entrenados de agentes desde: {restore_agents_path}")
            try:
                state_dict = torch.load(restore_agents_path, map_location="cpu")
                trainer.get_policy("a").model.load_state_dict(state_dict)
                logger.info("Pesos de política 'a' (agentes) cargados exitosamente.")
                if train_planner_flag:
                    logger.info("Pesos de agentes utilizados para entrenar en Fase 2.")
            except Exception as e:
                logger.error(f"Error al cargar pesos de agentes: {e}")
                logger.warning("Continuando con pesos aleatorios para agentes.")
        else:
            if train_planner_flag:
                logger.warning("Fase 2 activada (train_planner=True) pero no se encontraron pesos de agentes.")
                logger.warning(f"  Path de agentes especificado: {restore_agents_path}")

        # Cargar pesos previos del planner (opcional)
        if restore_planner_path and os.path.exists(restore_planner_path):
            logger.info(f"Cargando pesos pre-entrenados de planner desde: {restore_planner_path}")
            try:
                state_dict = torch.load(restore_planner_path, map_location="cpu")
                trainer.get_policy("p").model.load_state_dict(state_dict)
                logger.info("Pesos de política 'p' (planner) cargados exitosamente.")
            except Exception as e:
                logger.error(f"Error al cargar pesos de planner: {e}")
                logger.warning("Continuando con pesos aleatorios para planner.")

    # ---- Entrenamiento ----
    num_iterations = general_cfg.get("num_iterations", 100)
    logger.info(f"Comenzando entrenamiento por {num_iterations} iteraciones...")

    history, last_checkpoint = train(trainer, num_iters=num_iterations, planner=planner)
    logger.info(f"Último checkpoint RLlib (planner+agentes CNN): {last_checkpoint}")

    # Guardar historial
    os.makedirs("nuevo_cnn_planner", exist_ok=True)
    csv_path = "nuevo_cnn_planner/ppo_results_with_planner_cnn.csv"
    save_history_to_csv(history, csv_path)

    # Evaluación
    logger.info("\nEjecutando episodio de evaluación...")
    episode_length = env_config.get("episode_length", 1000)
    run_eval_episode(trainer, env_obj, max_steps=episode_length)

    logger.info("Cerrando Ray...")
    ray.shutdown()

    logger.info("=" * 70)
    logger.info("Entrenamiento FASE 2 CNN completado exitosamente!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main(planner=True)
