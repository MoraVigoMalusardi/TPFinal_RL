import os
import sys
import csv
import ray
import yaml
import argparse
import logging
import warnings
import gym
import numpy as np
import torch
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
logger = logging.getLogger("train_agents")
logger.setLevel(logging.INFO)

# Silenciar logs ruidosos
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
    Arquitectura CIENTÍFICA para comparación justa (A/B Test):
    1. Input: CNN (Optimizada con Stride=2) + Inventario Flat.
    2. Fusión.
    3. RAMAS SEPARADAS (Igual que el baseline 'vf_share_layers: false'):
       - Rama Actor: 2 capas de 256 (Tanh) -> Acción.
       - Rama Crítico: 2 capas de 256 (Tanh) -> Valor.
    """
    def _init_(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2._init_(self, obs_space, action_space, num_outputs, model_config, name)
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
                else:
                    flat_dim += int(np.prod(space.shape))
        else:
            flat_dim = int(np.prod(obs_space.shape))

        # --- 1. Rama CNN (Visión) - Stride 2 para velocidad ---
        if spatial_shape:
            # Detectar canales (usualmente 7)
            in_channels = spatial_shape[0] if spatial_shape[0] < spatial_shape[2] else spatial_shape[2]
            
            self.cnn = nn.Sequential(
                # Capa 1: Stride 1 (Mantiene resolución)
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # Capa 2: Stride 2 (Reduce a la mitad) -> Reemplaza al MaxPool para ser más preciso
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Cálculo dinámico del tamaño de salida
            # Creamos un tensor dummy con el formato correcto [1, C, H, W]
            dummy_shape = (1, in_channels, spatial_shape[1], spatial_shape[2])
            dummy_input = torch.zeros(dummy_shape)
            with torch.no_grad():
                cnn_out_dim = self.cnn(dummy_input).shape[1]
                
            logger.info(f"CNN inicializada. Stride=2. Salida plana: {cnn_out_dim}")
        else:
            self.cnn = None
            cnn_out_dim = 0

        # --- 2. Rama Flat (Inventario) ---
        # Usamos Tanh para ser consistentes con el baseline
        self.flat_processor = nn.Sequential(
            nn.Linear(flat_dim, 32),
            nn.Tanh()
        )

        concat_dim = cnn_out_dim + 32
        
        # --- 3. RAMAS GEMELAS (Igualando tu Baseline) ---
        # En lugar de una sola hidden_layer, creamos dos caminos separados.
        
        # RAMA ACTOR (Decide qué hacer)
        # Replica: Linear -> Tanh -> Linear -> Tanh (256 neuronas)
        self.actor_layers = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.action_head = nn.Linear(256, num_outputs)

        # RAMA CRÍTICO (Estima el valor)
        # Replica: _value_branch_separate de tu baseline
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
                # Ajuste de canales: [B, H, W, C] -> [B, C, H, W]
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
                # Caso borde si solo hubiera mapa
                device = input_dict["obs_flat"].device if "obs_flat" in input_dict else map_input.device
                flat_input = torch.zeros(map_input.shape[0], 0).to(device)
        else:
            flat_input = obs.float()

        flat_out = self.flat_processor(flat_input)

        # C. Fusionar
        if cnn_out is not None:
            features = torch.cat([cnn_out, flat_out], dim=1)
        else:
            features = flat_out
            
        # D. Caminos Separados (Igual al Baseline)
        
        # Camino del Actor
        actor_out = self.actor_layers(features)
        logits = self.action_head(actor_out)
        
        # Camino del Crítico (Value)
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
# 3) Funciones de Configuración
# -------------------------------------------------------------------

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="phase1")
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--restore-checkpoint", type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.yaml found at: {config_path}")

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    if args.num_iters is not None:
        run_configuration.setdefault("general", {})
        run_configuration["general"]["num_iterations"] = args.num_iters

    return run_configuration, args.run_dir, args.restore_checkpoint


def build_env_config(run_configuration):
    env_config = run_configuration.get("env", {}).copy()
    env_config.setdefault("scenario_name", "layout_from_file/simple_wood_and_stone")

    # === CAMBIO CRÍTICO PARA CNN ===
    # Desactivamos el aplanado para conservar la estructura 2D del mapa
    env_config["flatten_observations"] = False 
    env_config["flatten_masks"] = True # Mantenemos máscaras planas para facilitar logits
    # ===============================

    logger.info(f"Env Config - Scenario: {env_config['scenario_name']}")
    logger.info("Flatten Observations set to FALSE for CNN compatibility.")
    return env_config


def create_env_for_inspection(env_config):
    # Usamos SafeEnvWrapper para inspección también
    return SafeEnvWrapper({"env_config_dict": env_config}, verbose=True)


# -------------------------------------------------------------------
# 4) Políticas Multi-Agente
# -------------------------------------------------------------------

def build_multiagent_policies(env_obj, run_configuration):
    general_config = run_configuration.get("general", {})
    agent_policy_config = run_configuration.get("agent_policy", {})
    planner_policy_config = run_configuration.get("planner_policy", {})

    train_planner = general_config.get("train_planner", False)

    # Configuración del Modelo para Agentes
    agent_model = {
        "custom_model": "paper_cnn_torch", 
        "custom_model_config": {},
        # Desactivamos LSTM nativa de RLlib para usar solo nuestra CNN+MLP
        "use_lstm": False, 
        "vf_share_layers": False,
    }

    # Configuración del Modelo para Planner
    planner_model = {
        "custom_model": "paper_cnn_torch", 
        "custom_model_config": {},
        "use_lstm": False, 
        "vf_share_layers": False,
    }

    # Crear políticas
    policies = {
        "a": (
            None,
            env_obj.observation_space, # Ahora es Dict Space (no plano)
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
        return "a" if str(agent_id).isdigit() else "p"

    policies_to_train = ["a"] if not train_planner else ["a", "p"]

    return policies, policy_mapping_fn, policies_to_train


# -------------------------------------------------------------------
# 5) Trainer Config
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
        "num_workers": trainer_yaml_config.get("num_workers", 12), # Ajustado a 4 por seguridad
        "num_envs_per_worker": trainer_yaml_config.get("num_envs_per_worker", 2),
        "framework": "torch",
        "num_gpus": trainer_yaml_config.get("num_gpus", 0),
        "log_level": "ERROR",
        "train_batch_size": trainer_yaml_config.get("train_batch_size", 4800),
        "sgd_minibatch_size": trainer_yaml_config.get("sgd_minibatch_size", 512),
        "num_sgd_iter": trainer_yaml_config.get("num_sgd_iter", 10),
        "rollout_fragment_length": trainer_yaml_config.get("rollout_fragment_length", 200),
        "batch_mode": "truncate_episodes",
        "no_done_at_end": False,
        "env_config": {
            "env_config_dict": env_config,
            "num_envs_per_worker": trainer_yaml_config.get("num_envs_per_worker", 2),
        }
    }
    return trainer_config


def create_tb_logger_creator(run_dir):
    def logger_creator(config):
        logdir = os.path.join(run_dir, "tb_logs_cnn")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=None)
    return logger_creator


# -------------------------------------------------------------------
# 6) Bucle de Entrenamiento
# -------------------------------------------------------------------

def train(trainer, num_iters=5):
    """
    Ejecuta el lazo de entrenamiento de PPO y guarda:
    - CSV
    - Pesos de policies (state_dict) -> RESTAURADO
    - Checkpoint completo de RLlib
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
        
        # Extraemos métricas para el CSV
        a_mean = policy_reward_mean.get("a")
        p_mean = policy_reward_mean.get("p")
        
        print(f"episode_reward_mean: {episode_reward_mean}")
        print(f"policy_reward_mean: a={a_mean}, p={p_mean}")

        row = {
            "iteration": training_iteration if training_iteration is not None else it,
            "timesteps_total": timesteps_total,
            "episodes_total": episodes_total,
            "episode_reward_mean": episode_reward_mean,
            "episode_len_mean": episode_len_mean,
            "policy_a_reward_mean": a_mean,
            "policy_p_reward_mean": p_mean,
        }
        history.append(row)

    # ==== (RESTAURADO) Guardar pesos de policies (state_dict) ====
    # Esto guarda solo los pesos de la red neuronal (CNN+MLP), sin el estado del optimizador
    import torch

    # Usamos una carpeta distinta para no mezclar con lo anterior
    os.makedirs("checkpoints/nuevo_cnn", exist_ok=True)
    
    # Guardar Agente
    torch.save(
        trainer.get_policy("a").model.state_dict(),
        "checkpoints/nuevo_cnn/policy_a_cnn_weights.pt",
    )
    print("Pesos del Agente (CNN) guardados en checkpoints/nuevo_cnn/policy_a_cnn_weights.pt")

    # Guardar Planner (si existe)
    if "p" in trainer.workers.local_worker().policy_map:
        torch.save(
            trainer.get_policy("p").model.state_dict(),
            "checkpoints/nuevo_cnn/policy_p_cnn_weights.pt",
        )

    # ==== Guardar checkpoint completo de RLlib ====
    checkpoint_root = os.path.join("checkpoints", "rllib_cnn_full")
    os.makedirs(checkpoint_root, exist_ok=True)
    checkpoint_path = trainer.save(checkpoint_root)
    logger.info(f"Checkpoint RLlib completo guardado en: {checkpoint_path}")

    return history, checkpoint_path


# -------------------------------------------------------------------
# 7) Rollout de Evaluación
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
            # En CNN sin LSTM, state es una lista vacía y devuelve lista vacía
            action, _, _ = trainer.compute_action(
                ob, 
                state=[], 
                policy_id=policy_id,
                full_fetch=True
            )
            actions[agent_id] = action

        obs, rew, done, _info = env_obj.step(actions)

        for agent_id, r in rew.items():
            total_rewards[agent_id] = total_rewards.get(agent_id, 0.0) + r
        step += 1

    print("\nEpisodio de evaluación finalizado.")
    print("Recompensa total por agente:", total_rewards)


# -------------------------------------------------------------------
# 8) Main
# -------------------------------------------------------------------

def main():
    run_configuration, run_dir, restore_checkpoint = process_args()
    
    # 1. Registrar el modelo CNN
    logger.info("Inicializando Ray...")
    ray.init(include_dashboard=False, log_to_driver=False)
    ModelCatalog.register_custom_model("paper_cnn_torch", AI_Economist_CNN_PyTorch)
    logger.info("Modelo Custom 'paper_cnn_torch' registrado.")

    # 2. Configurar Entorno (FORZANDO FLATTEN=FALSE)
    env_config = build_env_config(run_configuration)
    
    # 3. Crear entorno de inspección con SafeWrapper
    env_obj = create_env_for_inspection(env_config)

    # 4. Crear Config del Trainer
    trainer_config = build_trainer_config(env_obj, run_configuration, env_config)
    logger_creator = create_tb_logger_creator(run_dir)

    # 5. Instanciar Trainer
    # OJO: Pasamos la clase SafeEnvWrapper, no RLlibEnvWrapper directo
    trainer = PPOTrainer(
        env=SafeEnvWrapper,
        config=trainer_config,
        logger_creator=logger_creator,
    )

    print(f"\nTensorBoard logs se están guardando en: {trainer.logdir}\n")


    # =============================================
    if restore_checkpoint and os.path.exists(restore_checkpoint):
        trainer.restore(restore_checkpoint)
    
    # Imprimir arquitectura
    print("\n" + "="*30)
    print("Arquitectura del Modelo Agente:")
    print(trainer.get_policy("a").model)
    print("="*30 + "\n")

    num_iterations = run_configuration.get("general", {}).get("num_iterations", 100)
    history, last_ckpt = train(trainer, num_iters=num_iterations)

    # Guardar CSV
    os.makedirs("nuevo_cnn", exist_ok=True)
    keys = history[0].keys()
    with open("nuevo_cnn/ppo_results_agents.csv", "w", newline="") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(history)

    run_eval_episode(trainer, env_obj, max_steps=500)
    ray.shutdown()

if __name__ == "__main__":
    main()