import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_agents import build_env_config, create_env_for_inspection, process_args, build_trainer_config, RLlibEnvWrapper
from ray.rllib.agents.ppo import PPOTrainer
import ray

import torch
import os

np.random.seed(10)  


def plot_agents_coins(trainer, env_obj, max_steps=200, plot=True):


    #Correr 100 veces y promediar resultados con el numero fijo de build skill por agente
    # fixed_build_skills = np.random.pareto(a=2.0, size=4) + 1 # Hacer fijo el build skill para cada agente en esta corrida
    # fixed_build_skills = fixed_build_skills / fixed_build_skills.max()
    # print(f"\nBuild skills fijos para esta corrida: {fixed_build_skills}")
    
    #fixed_build_skills = [0.5, 1.0, 1.5, 2.0]  
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





    # Guardo una lista de agents_data para luego sacar el promedio de cada cosa
    all_runs_agents_data = []


    for run_idx in range(100):
        print(f"\nIniciando episodio de evaluación {run_idx + 1}/100...")
        obs = env_obj.reset()
        for agent_idx in range(4):
            agent = env_obj.env.world.agents[agent_idx]
            if hasattr(agent, 'state') and isinstance(agent.state, dict):

                agent.state["build_payment"] = build_payment[agent_idx]
                agent.state["build_skill"] = float(fixed_build_skills[agent_idx])
                # agent.state["build_payment"] = [1, 2, 3, 4][agent_idx] * 10
                # agent.state["build_skill"] = [1, 2, 3, 4][agent_idx] * 10

                agent.state['gather_skill'] = fixed_gather_skills[agent_idx]
                agent.state['bonus_gather_prob'] = bonus_gather_prob[agent_idx]
                
                print(f"Seteado build_skill del Agente {agent_idx} a {fixed_build_skills[agent_idx]:.3f}")
                print(f"Seteado gather_skill del Agente {agent_idx} a {fixed_gather_skills[agent_idx]}")
                print(f"Seteado bonus_gather_prob del Agente {agent_idx} a {bonus_gather_prob[agent_idx]}")
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

        all_runs_agents_data.append(agents_data)

        print("\nEpisodio de evaluación finalizado:")
        print(f"  Pasos ejecutados: {step}")
        print("  Recompensa total por agente:")
        for agent_id, r in total_rewards.items():
            print(f"    {agent_id}: {r:.2f}")

    # Promediar datos de todos los runs
    agents_data = [
        {"Coin": [], "Stone": [], "Wood": [], "Location": [], "Labor": [], "BuildSkill": [], "BuildPayment": [], "BonusGatherProb": []}
        for _ in range(4)
    ]
    for agent_idx in range(4):
        for key in agents_data[agent_idx].keys():
            # Promediar sobre todos los runs
            averaged_data = np.mean(
                [all_runs_agents_data[run_idx][agent_idx][key] for run_idx in range(100)],
                axis=0
            )
            agents_data[agent_idx][key] = averaged_data.tolist()


    # Generar gráficos si se solicita
    if plot:
        plt.figure(figsize=(12, 8))
        # Graficar monedas recolectadas por cada agente a lo largo del tiempo dependiendo de su skill todo en el mismo grafico
        for agent_idx, agent_data in enumerate(agents_data):
           
            plt.plot(agent_data["Coin"], label=f"Agente {agent_idx} - Skill {agent_data['BuildSkill'][-1]:.3f}")
            plt.xlabel("Pasos")
            plt.ylabel("Coins recolectadas")
        plt.grid()
        plt.legend()
        plt.savefig(f"coins_build_skill.png")
        plt.close()

        # Graficar labor a lo largo del tiempo para cada agente
        plt.figure(figsize=(12, 8))
        for agent_idx, agent_data in enumerate(agents_data):
            plt.plot(agent_data["Labor"], label=f"Agente {agent_idx} - Skill {agent_data['BuildSkill'][-1]:.3f}")
            plt.xlabel("Pasos")
            plt.ylabel("Labor")
        plt.grid()
        plt.legend()
        plt.savefig(f"labor_over_time.png")
        plt.close()

      


def main():

    # Parsear argumentos y cargar configuración del YAML
    run_configuration, run_dir, restore_checkpoint = process_args()
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
    a_path = "policy_a_weights.pt"
    if os.path.exists(a_path):
        trainer.get_policy("a").model.load_state_dict(torch.load(a_path, map_location="cpu"))
        print("Pesos de política 'a' cargados.")



    result = plot_agents_coins(trainer, env_obj, max_steps=1000)
    print(result)


if __name__ == "__main__":
    main()

    

