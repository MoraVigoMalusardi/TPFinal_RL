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
logger = logging.getLogger("train_planner")
logger.setLevel(logging.INFO)

logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)


# -------------------------------------------------------------------
# 1) Wrapper de Seguridad
# -------------------------------------------------------------------
class SafeEnvWrapper(MultiAgentEnv):
    def __init__(self, config, **kwargs):
        verbose = kwargs.get("verbose", False)
        self.internal_env = RLlibEnvWrapper(config, verbose=verbose)
        
        self.observation_space = self.internal_env.observation_space
        self.action_space = self.internal_env.action_space
        
        # Limpieza del espacio del Planner
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
        if 'p' in obs and isinstance(obs['p'], dict):
            keys_to_remove = [k for k in obs['p'].keys() if k.startswith('p') and k[1:].isdigit()]
            for k in keys_to_remove:
                del obs['p'][k]
        return obs


# -------------------------------------------------------------------
# 2) Modelo CNN (El mismo para Agentes y Planner)
# -------------------------------------------------------------------
class AI_Economist_CNN_PyTorch(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.spatial_key = "world-map"
        
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        spatial_shape = None
        flat_dim = 0

        if hasattr(original_space, "spaces"):
            for key, space in original_space.spaces.items():
                if key == self.spatial_key:
                    spatial_shape = space.shape 
                elif key == "action_mask":
                    pass 
                else:
                    flat_dim += int(np.prod(space.shape))
        else:
            flat_dim = int(np.prod(obs_space.shape))

        # --- Rama CNN ---
        if spatial_shape:
            in_channels = spatial_shape[2] 
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            with torch.no_grad():
                dummy_input = torch.zeros(1, spatial_shape[0], spatial_shape[1], in_channels)
                dummy_input = dummy_input.permute(0, 3, 1, 2)
                cnn_out_dim = self.cnn(dummy_input).shape[1]
        else:
            self.cnn = None
            cnn_out_dim = 0

        # --- Rama MLP ---
        self.flat_processor = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        concat_dim = cnn_out_dim + 64
        self.hidden_layer = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU()
        )

        self.action_branch = nn.Linear(256, num_outputs)
        self.value_branch = nn.Linear(256, 1)
        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        cnn_out = None
        flat_parts = []

        if self.cnn is not None:
            if isinstance(obs, dict) and self.spatial_key in obs:
                map_input = obs[self.spatial_key].float()
                map_input = map_input.permute(0, 3, 1, 2)
                cnn_out = self.cnn(map_input)

        if isinstance(obs, dict):
            for key, val in obs.items():
                if key != self.spatial_key and key != "action_mask":
                    flat_parts.append(val.float().reshape(val.shape[0], -1))
            if flat_parts:
                flat_input = torch.cat(flat_parts, dim=1)
            else:
                # Manejo de caso borde si no hay parte plana
                if hasattr(input_dict["obs_flat"], "device"):
                    dev = input_dict["obs_flat"].device
                else:
                    dev = torch.device("cpu")
                flat_input = torch.zeros(obs[list(obs.keys())[0]].shape[0], 0).to(dev)
        else:
            flat_input = obs.float()

        flat_out = self.flat_processor(flat_input)

        if cnn_out is not None:
            combined = torch.cat([cnn_out, flat_out], dim=1)
        else:
            combined = flat_out
            
        x = self.hidden_layer(combined)
        logits = self.action_branch(x)
        self._cur_value = self.value_branch(x).squeeze(1)

        if isinstance(obs, dict) and "action_mask" in obs:
            inf_mask = torch.clamp(torch.log(obs["action_mask"]), min=-1e10)
            logits = logits + inf_mask

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._cur_value


# -------------------------------------------------------------------
# 3) Configuración y Argumentos
# -------------------------------------------------------------------

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="phase2", help="Directorio config")
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

    # === CONFIGURACIÓN CRÍTICA PARA CNN ===
    env_config["flatten_observations"] = False 
    env_config["flatten_masks"] = True
    env_config["planner_gets_spatial_info"] = True
    # ======================================

    return env_config


def create_env_for_inspection(env_config):
    return SafeEnvWrapper({"env_config_dict": env_config}, verbose=True)


# -------------------------------------------------------------------
# 4) Políticas Multi-Agente (agentes + planner)
# -------------------------------------------------------------------

def build_multiagent_policies(env_obj, run_configuration):
    general_config = run_configuration.get("general", {})
    agent_policy_config = run_configuration.get("agent_policy", {})
    planner_policy_config = run_configuration.get("planner_policy", {})

    # Definimos el modelo CNN para todos
    common_model_config = {
        "custom_model": "cotraining_cnn_torch",
        "custom_model_config": {},
        "use_lstm": False,
        "vf_share_layers": False,
    }

    policies = {
        # AGENTES (Policy "a") 
        "a": (
            None,
            env_obj.observation_space,
            env_obj.action_space,
            {
                "model": common_model_config,
                "gamma": agent_policy_config.get("gamma", 0.998),
                "lr": agent_policy_config.get("lr", 0.0003), 
                "vf_loss_coeff": agent_policy_config.get("vf_loss_coeff", 0.05),
                "entropy_coeff": agent_policy_config.get("entropy_coeff", 0.025),
                "clip_param": agent_policy_config.get("clip_param", 0.3),
            },
        ),
        # PLANNER (Policy "p")
        "p": (
            None,
            env_obj.observation_space_pl,
            env_obj.action_space_pl,
            {
                "model": common_model_config,
                "gamma": planner_policy_config.get("gamma", 0.998),
                "lr": planner_policy_config.get("lr", 0.0001),
                "vf_loss_coeff": planner_policy_config.get("vf_loss_coeff", 0.05),
                "entropy_coeff": planner_policy_config.get("entropy_coeff", 0.1),
                "clip_param": planner_policy_config.get("clip_param", 0.3),
            },
        ),
    }

    def policy_mapping_fn(agent_id):
        return "a" if str(agent_id).isdigit() else "p"

    policies_to_train = ["a", "p"] 

    logger.info(f"Entrenando políticas: {policies_to_train}")
    logger.info(f"LR Agentes: {policies['a'][3]['lr']}")
    logger.info(f"LR Planner: {policies['p'][3]['lr']}")

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
        "num_workers": trainer_yaml_config.get("num_workers", 12),
        "num_envs_per_worker": trainer_yaml_config.get("num_envs_per_worker", 2),
        "framework": "torch",
        "num_gpus": 0,
        "log_level": "ERROR",
        "train_batch_size": trainer_yaml_config.get("train_batch_size", 4800),
        "sgd_minibatch_size": trainer_yaml_config.get("sgd_minibatch_size", 512),
        "num_sgd_iter": trainer_yaml_config.get("num_sgd_iter", 10),
        "rollout_fragment_length": trainer_yaml_config.get("rollout_fragment_length", 200),
        "batch_mode": "truncate_episodes",
        "env_config": {
            "env_config_dict": env_config,
            "num_envs_per_worker": trainer_yaml_config.get("num_envs_per_worker", 2),
        }
    }
    return trainer_config


def create_tb_logger_creator(run_dir):
    def logger_creator(config):
        logdir = os.path.join(run_dir, "tb_logs_planner_cnn")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=None)
    return logger_creator


# -------------------------------------------------------------------
# 6) Entrenamiento y Guardado
# -------------------------------------------------------------------

def train(trainer, num_iters=5):
    history = []
    for it in range(num_iters):
        print(f"\n********** Co-Training Iteración: {it} **********")
        result = trainer.train()

        ep_reward = result.get("episode_reward_mean")
        pol_reward = result.get("policy_reward_mean", {})
        
        a_mean = pol_reward.get("a")
        p_mean = pol_reward.get("p")
        
        print(f"episode_reward_mean: {ep_reward}")
        print(f"policy_reward_mean: a={a_mean} (Train), p={p_mean} (Train)")

        history.append({
            "iteration": it,
            "timesteps_total": result.get("timesteps_total"),
            "episode_reward_mean": ep_reward,
            "policy_a_reward_mean": a_mean,
            "policy_p_reward_mean": p_mean,
        })

    # Guardar pesos de ambos
    os.makedirs("checkpoints/nuevo_cnn", exist_ok=True)
    
    torch.save(
        trainer.get_policy("a").model.state_dict(),
        "checkpoints/nuevo_cnn/policy_a_weights_w_planner.pt",
    )
    if "p" in trainer.workers.local_worker().policy_map:
        torch.save(
            trainer.get_policy("p").model.state_dict(),
            "checkpoints/nuevo_cnn/policy_p_weights_w_planner.pt",
        )
    print("Pesos guardados en checkpoints/nuevo_cnn/")

    # Guardar checkpoint completo
    checkpoint_path = trainer.save("checkpoints/rllib_nuevo_cnn_planner_full")
    return history, checkpoint_path


# -------------------------------------------------------------------
# 7) Main
# -------------------------------------------------------------------

def main():
    run_configuration, run_dir, restore_checkpoint = process_args()
    
    logger.info("Iniciando CO-TRAINING: Agentes + Planner con CNN")
    ray.init(include_dashboard=False, log_to_driver=False)
    
    # 1. Registrar Modelo
    ModelCatalog.register_custom_model("cotraining_cnn_torch", AI_Economist_CNN_PyTorch)
    
    # 2. Configs
    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, run_configuration, env_config)
    logger_creator = create_tb_logger_creator(run_dir)

    # 3. Trainer
    trainer = PPOTrainer(
        env=SafeEnvWrapper,
        config=trainer_config,
        logger_creator=logger_creator,
    )

    # 4. CARGAR PESOS INICIALES (Si existen)
    # Si queremos empezar desde los agentes ya entrenados en Fase 1
    if restore_checkpoint and os.path.exists(restore_checkpoint):
        trainer.restore(restore_checkpoint)
    else:
        # Intentar cargar pesos de agentes previos
        agents_weights_path = run_configuration.get("general", {}).get(
            "restore_tf_weights_agents", 
            "checkpoints/nuevo_cnn/policy_a_cnn_weights.pt"
        )
        
        if os.path.exists(agents_weights_path):
            logger.info(f"Cargando pesos de Agentes Fase 1 para Co-Training: {agents_weights_path}")
            try:
                state_dict = torch.load(agents_weights_path)
                trainer.get_policy("a").model.load_state_dict(state_dict)
                logger.info("Pesos de Agentes cargados. Se continuará su entrenamiento.")
            except Exception as e:
                logger.error(f"Error cargando pesos: {e}")
        else:
            logger.warning("No se encontraron pesos previos. Se entrenará todo desde cero.")

    # 5. Entrenar
    num_iterations = run_configuration.get("general", {}).get("num_iterations", 100)
    history, _ = train(trainer, num_iters=num_iterations)

    # 6. Guardar CSV
    os.makedirs("nuevo_cnn", exist_ok=True)
    keys = history[0].keys()
    with open("nuevo_cnn/ppo_results_with_planner.csv", "w", newline="") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(history)

    ray.shutdown()
    logger.info("Co-Training Finalizado.")

if __name__ == "__main__":
    main()