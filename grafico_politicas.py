import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import ray
from train_agents_ppo import (
    build_env_config, 
    create_env_for_inspection,
    build_trainer_config,
    RLlibEnvWrapper
)
import torch
import pandas as pd

from ray.rllib.agents.ppo import PPOTrainer

def gini(array):
    """Calcula el coeficiente de Gini."""
    array = np.array(array, dtype=float)
    if len(array) == 0 or array.sum() == 0:
        return 0.0
    sorted_array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_array)) / (n * np.sum(sorted_array)) - (n + 1) / n

def evaluate_policy(config_path, policy_name, trainer=None, max_steps=1000, n_episodes=5):
    """
    Evalúa una política de impuestos y retorna métricas.
    
    Args:
        config_path: Path al archivo de configuración YAML
        policy_name: Nombre de la política ("Free Market", "US Federal", etc.)
        trainer: Trainer con pesos cargados (solo para AI Economist)
        max_steps: Pasos máximos por episodio
        n_episodes: Número de episodios a promediar
    
    Returns:
        dict: Métricas promediadas (productivity, equality, etc.)
    """
    print(f"\n{'='*70}")
    print(f"Evaluando: {policy_name}")
    print(f"{'='*70}")
    
    # Cargar configuración
    with open(config_path, 'r') as f:
        run_configuration = yaml.safe_load(f)
    
    # Crear entorno
    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)
    

    # Ejecutar múltiples episodios
    all_productivity = []
    all_equality = []
    all_gini = []
    
    for episode in range(n_episodes):
        obs = env_obj.reset()
        done = {"__all__": False}
        step = 0

        initial_coins = [agent.total_endowment('Coin') for agent in env_obj.env.world.agents]
        print(f"\n  Initial coins: {[f'{c:.1f}' for c in initial_coins]}")
        
        while not done["__all__"] and step < max_steps:
            actions = {}
            for agent_id, ob in obs.items():
                policy_id = "a" if str(agent_id).isdigit() else "p"
                action = trainer.compute_action(ob, policy_id=policy_id, explore=False)
                actions[agent_id] = action

                #cada 100 pasos imprimo la accion tomada por cada agente
                if step % 100 == 0:
                    print(f"    Agent {agent_id} took action {action}")
            
            obs, rew, done, info = env_obj.step(actions)    
            if step % 100 == 0:
                print(f"    Step {step+1} done with rewards {rew}")

                current_coins = [agent.total_endowment('Coin') for agent in env_obj.env.world.agents]
                print(f"    Current coins: {[f'{c:.1f}' for c in current_coins]}")

                coin_changes = [curr - init for curr, init in zip(current_coins, initial_coins)]
                print(f"    Total change: {[f'{c:+.1f}' for c in coin_changes]}")
            step += 1
        
        # Calcular métricas finales del episodio
        final_coins = np.array([
            agent.total_endowment('Coin') 
            for agent in env_obj.env.world.agents
        ])
        
        productivity = final_coins.sum()
        gini_coeff = gini(final_coins)
        n = len(final_coins)
        equality = 1 - (n / (n - 1)) * gini_coeff if n > 1 else 1.0
        
        all_productivity.append(productivity)
        all_equality.append(equality)
        all_gini.append(gini_coeff)
        
        print(f"  Episode {episode+1}/{n_episodes}: "
              f"Prod={productivity:.1f}, Eq={equality:.3f}, Gini={gini_coeff:.3f}")
    
    # Promediar resultados
    results = {
        'policy_name': policy_name,
        'productivity': np.mean(all_productivity),
        'productivity_std': np.std(all_productivity),
        'equality': np.mean(all_equality),
        'equality_std': np.std(all_equality),
        'gini': np.mean(all_gini),
        'gini_std': np.std(all_gini),
        'eq_times_prod': np.mean(all_equality) * np.mean(all_productivity),
    }
    
    print(f"\n  Resultados promediados:")
    print(f"  Productivity: {results['productivity']:.1f} ± {results['productivity_std']:.1f}")
    print(f"  Equality: {results['equality']:.3f} ± {results['equality_std']:.3f}")
    print(f"  Gini: {results['gini']:.3f} ± {results['gini_std']:.3f}")
    print(f"  Eq × Prod: {results['eq_times_prod']:.1f}")
    
    return results

def compare_all_policies():
    """
    Compara las 4 políticas y genera el gráfico tipo AI-Economist.
    """
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    # 1. Free Market (sin impuestos)

    config = 'config_free_market.yaml' #ACA VA LA CONFIG DE EVALUACION DE FREE MARKET
    print("\n Cargando pesos del Free Market...")
    with open(config, 'r') as f:   
        free_config = yaml.safe_load(f)
    
    env_config = build_env_config(free_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, free_config, env_config)
    free_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
    
    
    agent_weights = torch.load('a_fm.pt', map_location='cpu') #ACA VA EL CHECKPOINT DE FREE MARKET (LOS PRIMEROS AGENTES QUE ENTRENAMOS)
    free_trainer.get_policy("a").model.load_state_dict(agent_weights)
    

    print("Pesos cargados")
    
    results_free_market = evaluate_policy(
        config,
        'Free Market',
        trainer=free_trainer,
        n_episodes=10
    )

    free_trainer.stop()
    
    # 2. US Federal

    config = 'config_us_fed.yaml'  #ACA VA LA CONFIG DE EVALUACION DE US FEDERAL

    print("\n Cargando pesos del US Federal...")
    with open(config, 'r') as f:
        us_fed_config = yaml.safe_load(f)
    
    env_config = build_env_config(us_fed_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, us_fed_config, env_config)
    us_fed_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
    
    agent_weights = torch.load('a_usfed.pt', map_location='cpu')
    us_fed_trainer.get_policy("a").model.load_state_dict(agent_weights)
    

    print("Pesos cargados")
    
    results_us_federal = evaluate_policy(
        config,
        'US Federal',
        trainer=us_fed_trainer,
        n_episodes=10
    )

    us_fed_trainer.stop()
    
    # 3. Saez Formula 

    config = 'config_saez_nueva.yaml'  #ACA VA LA CONFIG DE EVALUACION DE SAEZ FORMULA

    print("\n Cargando pesos de Saez Formula...")
    with open(config, 'r') as f:
        saez_config = yaml.safe_load(f)
    
    env_config = build_env_config(saez_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, saez_config, env_config)
    saez_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
    
    agent_weights = torch.load('a_saez.pt', map_location='cpu')
    saez_trainer.get_policy("a").model.load_state_dict(agent_weights)
    

    print("Pesos cargados")
    
    results_saez = evaluate_policy(
        config,
        'Saez Formula',
        trainer=saez_trainer,
        n_episodes=10
    )
    saez_trainer.stop()

    # 4. AI Economist (cargar pesos entrenados)
    print("\n Cargando pesos del AI Economist...")
    config = 'config_ai_econ.yaml'  #ACA VA LA CONFIG DE EVALUACION DE AI ECONOMIST
    agents = 'a_aiecon.pt' #ACA VA EL CHECKPOINT DE LOS AGENTES DE AI ECONOMIST (WITH PLANNER)
    planner = 'p_aiecon.pt' #ACA VA EL CHECKPOINT DEL PLANNER DE AI ECONOMIST (WITH PLANNER)

    with open(config, 'r') as f:
        ai_config = yaml.safe_load(f)
    
    env_config = build_env_config(ai_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, ai_config, env_config)
    ai_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
 
    agent_weights = torch.load(agents, map_location='cpu')
    ai_trainer.get_policy("a").model.load_state_dict(agent_weights)
    
    planner_weights = torch.load(planner, map_location='cpu')
    ai_trainer.get_policy("p").model.load_state_dict(planner_weights)
    print(" Pesos cargados")
    
    results_ai_economist = evaluate_policy(
        config,
        'AI Economist',
        trainer=ai_trainer,
        n_episodes=10
    )
    
    ai_trainer.stop()
    ray.shutdown()
    
    all_results = [
        results_free_market,
        results_us_federal,
        results_saez,
        results_ai_economist
    ]
    
    plot_comparison(all_results)
    
    return all_results

def plot_comparison(results_list, save_path='policy_comparison.png'):
    """
    Genera el gráfico de barras estilo AI-Economist paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    policies = [r['policy_name'] for r in results_list]
    colors = ['#FF6B6B', '#C77DFF', '#4EA8DE', '#06A77D']  # Coral, Purple, Blue, Teal
    
    # Subplot 1: Productivity
    ax = axes[0]
    productivity_vals = [r['productivity'] for r in results_list]
    productivity_stds = [r['productivity_std'] for r in results_list]
    bars = ax.bar(range(len(policies)), productivity_vals, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(range(len(policies)), productivity_vals, yerr=productivity_stds,
               fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=11, fontweight='bold')
    ax.set_ylabel('Coin Produced', fontsize=13, fontweight='bold')
    ax.set_title('Economic Productivity', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Valores sobre barras
    for i, (bar, val) in enumerate(zip(bars, productivity_vals)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{int(val)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Subplot 2: Equality
    ax = axes[1]
    equality_vals = [r['equality'] * 100 for r in results_list]  # Como porcentaje
    equality_stds = [r['equality_std'] * 100 for r in results_list]
    bars = ax.bar(range(len(policies)), equality_vals, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(range(len(policies)), equality_vals, yerr=equality_stds,
               fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=11, fontweight='bold')
    ax.set_ylabel('Coin Equality', fontsize=13, fontweight='bold')
    ax.set_title('Income Equality', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Valores sobre barras
    for i, (bar, val) in enumerate(zip(bars, equality_vals)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{int(val)}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Subplot 3: Equality × Productivity
    ax = axes[2]
    eq_prod_vals = [r['eq_times_prod'] for r in results_list]
    bars = ax.bar(range(len(policies)), eq_prod_vals, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=11, fontweight='bold')
    ax.set_ylabel('Eq. / Prod. Tradeoff', fontsize=13, fontweight='bold')
    ax.set_title('Equality x Productivity', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Valores sobre barras
    for i, (bar, val) in enumerate(zip(bars, eq_prod_vals)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{int(val)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Gráfico de comparación guardado: {save_path}")
    plt.close()

def plot_tax_comparison(ai_results, output_dir="."):
    """
    Genera un gráfico de barras comparando las tasas de la IA vs Free Market.
    Toma el promedio de impuestos del último episodio de la IA.
    """
    tax_history = ai_results.get('tax_history', [])
    brackets = ai_results.get('brackets', [])
    
    if len(tax_history) == 0 or len(brackets) == 0:
        print("⚠️ No hay datos suficientes para graficar la comparación de impuestos.")
        return

    # 1. Calcular el promedio de tasas del último episodio (para suavizar ruido)
    # tax_history shape: [Steps, NumBrackets] -> Promedio por columna
    ai_avg_rates = np.mean(tax_history, axis=0)
    
    # 2. Free Market (Todo 0)
    free_market_rates = np.zeros_like(ai_avg_rates)
    
    # 3. Configuración del Plot
    n_brackets = len(brackets)
    indices = np.arange(n_brackets)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Barra IA (Violeta)
    rects1 = ax.bar(indices - width/2, ai_avg_rates, width, 
                    label='AI Economist', color='mediumpurple', edgecolor='black', alpha=0.9)

    # Barra Free Market (Gris)
    rects2 = ax.bar(indices + width/2, free_market_rates, width, 
                    label='Free Market', color='lightgray', edgecolor='black', hatch='//')

    # Etiquetas y Títulos
    ax.set_ylabel('Marginal Tax Rate')
    ax.set_xlabel('Income Brackets (Coins)')
    ax.set_title('Tax Policy: AI Economist vs Free Market', fontweight='bold')
    ax.set_ylim(0, 1.05) # De 0% a 100%

    # Crear etiquetas legibles para el eje X (ej: "0-10", "10-50")
    labels = []
    prev_b = 0
    for b in brackets:
        labels.append(f"{prev_b}-{b}")
        prev_b = b
    # Etiqueta para el último bracket (infinito)
    # Si hay más tasas que brackets definidos, el último es "> X"
    if len(labels) < len(ai_avg_rates):
         labels.append(f"> {brackets[-1]}")
    
    # Ajustar si las longitudes no coinciden exactamente (seguridad)
    if len(labels) != len(indices):
        labels = [f"B{i}" for i in indices]

    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Poner el valor numérico encima de las barras de la IA
    for rect in rects1:
        height = rect.get_height()
        if height > 0.01:
            ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = f"{output_dir}/tax_comparison_bar_chart.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n[GRAFICO] Comparación de barras guardada en: {save_path}")
    plt.close()

def evaluate_policy2(config_path, policy_name, trainer=None, max_steps=1000, n_episodes=5):
    print(f"\n{'='*70}")
    print(f"Evaluando: {policy_name} (MODO DEBUG VALORES)")
    print(f"{'='*70}")
    
    with open(config_path, 'r') as f:
        run_configuration = yaml.safe_load(f)
    
    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)
    
    last_episode_tax_history = [] 
    brackets = [] 

    # 1. ENCONTRAR EL COMPONENTE DE IMPUESTOS
    tax_component = None
    for comp in env_obj.env.components:
        if "Tax" in str(type(comp)):
            tax_component = comp
            # Intentar sacar brackets
            if hasattr(comp, 'bracket_cutoffs'):
                brackets = comp.bracket_cutoffs
            elif hasattr(comp, 'brackets'):
                brackets = comp.brackets
            break
            
    if len(brackets) == 0: brackets = [10, 50, 100, 500, 1000, 2000, 5000]
    print(f"✅ Componente Fiscal: {type(tax_component).__name__}")
    print(f"✅ Brackets: {brackets}")
    
    # Validar si encontramos running_avg
    if tax_component and hasattr(tax_component, 'running_avg_tax_rates'):
        print("✅ Atributo 'running_avg_tax_rates' detectado.")
    else:
        print("⚠️ 'running_avg_tax_rates' NO detectado, usaremos 'curr_bracket_tax_rates'.")

    for episode in range(n_episodes):
        obs = env_obj.reset()
        done = {"__all__": False}
        step = 0
        current_episode_taxes = []

        while not done["__all__"] and step < max_steps:
            actions = {}
            for agent_id, ob in obs.items():
                policy_id = "a" if str(agent_id).isdigit() else "p"
                if trainer:
                    action = trainer.compute_action(ob, policy_id=policy_id, explore=False)
                else:
                    action = env_obj.action_space.sample()[agent_id]
                actions[agent_id] = action

            obs, rew, done, info = env_obj.step(actions)    
            
            # --- CAPTURAR TASAS ---
            current_step_rates = np.zeros(len(brackets))
            
            if tax_component:
                # ESTRATEGIA: Intentar leer el promedio acumulado primero (más estable)
                if hasattr(tax_component, 'running_avg_tax_rates'):
                     vals = tax_component.running_avg_tax_rates
                     if isinstance(vals, (list, np.ndarray)) and np.sum(vals) > 0:
                         current_step_rates = np.array(vals)
                
                # Si sigue siendo 0, intentar la tasa actual instantánea
                if np.sum(current_step_rates) == 0:
                    if hasattr(tax_component, 'curr_bracket_tax_rates'):
                        vals = tax_component.curr_bracket_tax_rates
                        if isinstance(vals, (list, np.ndarray)) and len(vals) > 0:
                            current_step_rates = np.array(vals)
            
            # Asegurar longitud correcta
            if len(current_step_rates) != len(brackets):
                current_step_rates = np.resize(current_step_rates, len(brackets))

            # --- DEBUGGING VISUAL ---
            # Imprimir los valores reales en el paso 100 para ver qué está pasando
            if episode == 0 and step == 100:
                print(f"\n[PASO 100] Tasas detectadas: {current_step_rates}")
                if np.sum(current_step_rates) == 0:
                    print("⚠️ ALERTA: Las tasas son todo CEROS. El Planner está eligiendo 0% o no está actuando.")
            # ------------------------

            current_episode_taxes.append(current_step_rates)
            step += 1
        
        if episode == n_episodes - 1:
            last_episode_tax_history = np.array(current_episode_taxes)
        
        print(f"  Episode {episode+1} Done.")
    
    results = {
        'policy_name': policy_name,
        'tax_history': last_episode_tax_history,
        'brackets': brackets,
    }
    
    return results


def evaluate_policy3(config_path, policy_name, trainer=None, max_steps=1000, n_episodes=5):
    print(f"\n{'='*70}")
    print(f"Evaluando: {policy_name} (MODO ACCIÓN DIRECTA)")
    print(f"{'='*70}")
    
    with open(config_path, 'r') as f:
        run_configuration = yaml.safe_load(f)
    
    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)
    
    last_episode_tax_history = [] 
    brackets = [10, 50, 100, 500, 1000, 2000, 5000] # Default seguro

    # Buscar brackets reales si existen
    for comp in env_obj.env.components:
        if hasattr(comp, 'brackets'):
            brackets = comp.brackets
            break
      

    for episode in range(n_episodes):
        obs = env_obj.reset()
        done = {"__all__": False}
        step = 0
        current_episode_taxes = []
        
        # Variable para recordar la última acción del planner
        last_planner_action = np.zeros(len(brackets))

        while not done["__all__"] and step < max_steps:
            actions = {}
            for agent_id, ob in obs.items():
                policy_id = "a" if str(agent_id).isdigit() else "p"
                
                # --- AQUÍ CAPTURAMOS LA ACCIÓN ---
                if trainer:
                    # Usa explore=False para ver la política real aprendida
                    action = trainer.compute_action(ob, policy_id=policy_id, explore=False)
                else:
                    action = env_obj.action_space.sample()[agent_id]
                
                actions[agent_id] = action

                # Si es el planner, guardamos su decisión
                if policy_id == "p":
                    # La acción es un array de índices (ej: [0, 5, 21...])
                    # Lo convertimos a Tasa % aproximada (Indice / Max_Indice)
                    # El espacio es MultiDiscrete(22), así que dividimos por 21.0
                    if hasattr(action, '__iter__'): 
                        rates_from_action = np.array(action) / 21.0
                        
                        # Asegurar longitud
                        if len(rates_from_action) == len(brackets):
                            last_planner_action = rates_from_action
            
            obs, rew, done, info = env_obj.step(actions)    
            
            # --- C. GUARDAMOS LA ACCIÓN COMO TASA ---
            # En lugar de leer el estado del env, usamos lo que decidió el cerebro
            current_episode_taxes.append(last_planner_action)
            # ----------------------------------------

            step += 1
        
        if episode == n_episodes - 1:
            last_episode_tax_history = np.array(current_episode_taxes)
        
        print(f"  Episode {episode+1} Done.")
    
    results = {
        'policy_name': policy_name,
        'tax_history': last_episode_tax_history,
        'brackets': brackets,
    }
    
    return results

def evaluate_policy4(config_path, policy_name, trainer=None, max_steps=1000, n_episodes=5):
    print(f"\n{'='*70}")
    print(f"Evaluando: {policy_name} (MODO HÍBRIDO: Brackets Auto + Acción Directa)")
    print(f"{'='*70}")
    
    with open(config_path, 'r') as f:
        run_configuration = yaml.safe_load(f)
    
    env_config = build_env_config(run_configuration)
    env_obj = create_env_for_inspection(env_config)
    
    last_episode_tax_history = [] 
    brackets = [] 

    # -----------------------------------------------------------
    # 1. TU CÓDIGO: DETECCIÓN ROBUSTA DE BRACKETS
    # -----------------------------------------------------------
    tax_component = None
    for comp in env_obj.env.components:
        if "Tax" in str(type(comp)):
            tax_component = comp
            # Intentar sacar brackets
            if hasattr(comp, 'bracket_cutoffs'):
                brackets = comp.bracket_cutoffs
            elif hasattr(comp, 'brackets'):
                brackets = comp.brackets
            break
            
    if len(brackets) == 0: brackets = [10, 50, 100, 500, 1000, 2000, 5000]
    
    # Convertir a lista de python normal si es numpy
    if hasattr(brackets, 'tolist'): brackets = brackets.tolist()
    
    print(f"✅ Componente Fiscal: {type(tax_component).__name__}")
    print(f"✅ Brackets detectados: {brackets}")
    # -----------------------------------------------------------

    for episode in range(n_episodes):
        obs = env_obj.reset()
        done = {"__all__": False}
        step = 0
        current_episode_taxes = []
        
        # Buffer para guardar la última decisión del planner
        # Inicializamos en 0 con el tamaño de los brackets detectados
        last_planner_action_rates = np.zeros(len(brackets))

        while not done["__all__"] and step < max_steps:
            actions = {}
            for agent_id, ob in obs.items():
                policy_id = "a" if str(agent_id).isdigit() else "p"
                
                # Usamos explore=False para ver la política real
                if trainer:
                    action = trainer.compute_action(ob, policy_id=policy_id, explore=False)
                else:
                    action = env_obj.action_space.sample()[agent_id]
                
                actions[agent_id] = action

                # -----------------------------------------------------------
                # 2. MI LÓGICA: CAPTURAR LA ACCIÓN (para que no salga en 0)
                # -----------------------------------------------------------
                if policy_id == "p":
                    # El espacio es MultiDiscrete(22) -> Indices 0 a 21
                    MAX_INDICE = 21.0
                    
                    if hasattr(action, '__iter__'):
                        # Convertimos índice a porcentaje (ej: Indice 21 -> 100%)
                        rates = np.array(action) / MAX_INDICE
                        
                        # Guardamos si coincide el tamaño
                        # (A veces la acción tiene un elemento más o menos que los brackets, ajustamos)
                        if len(rates) == len(brackets):
                            last_planner_action_rates = rates
                        elif len(rates) > len(brackets):
                            last_planner_action_rates = rates[:len(brackets)]
                        else:
                            # Relleno si falta
                            last_planner_action_rates[:len(rates)] = rates
                # -----------------------------------------------------------

            obs, rew, done, info = env_obj.step(actions)    
            
            # Guardamos la tasa calculada desde la acción
            current_episode_taxes.append(last_planner_action_rates)
            step += 1
        
        if episode == n_episodes - 1:
            last_episode_tax_history = np.array(current_episode_taxes)
        
        print(f"  Episode {episode+1} Done.")
    
    results = {
        'policy_name': policy_name,
        'tax_history': last_episode_tax_history,
        'brackets': brackets,
    }
    
    return results

if __name__ == "__main__":
    #results = compare_all_policies()

    #df = pd.DataFrame(results)
    #df.to_csv('policy_comparison_results.csv', index=False)
    #print("\n Resultados guardados en: policy_comparison_results.csv")
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    config = 'phase_aiecon_eval/config.yaml'  #ACA VA LA CONFIG DE EVALUACION DE AI ECONOMIST
    agents = 'checkpoints/nuevo_sin_lstm/policy_a_weights_w_planner.pt' #ACA VA EL CHECKPOINT DE LOS AGENTES DE AI ECONOMIST (WITH PLANNER)
    planner = 'checkpoints/nuevo_sin_lstm/policy_p_weights_w_planner.pt' #ACA VA EL CHECKPOINT DEL PLANNER DE AI ECONOMIST (WITH PLANNER)

    with open(config, 'r') as f:
        ai_config = yaml.safe_load(f)
    
    env_config = build_env_config(ai_config)
    env_obj = create_env_for_inspection(env_config)
    trainer_config = build_trainer_config(env_obj, ai_config, env_config)
    ai_trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)
 
    agent_weights = torch.load(agents, map_location='cpu')
    ai_trainer.get_policy("a").model.load_state_dict(agent_weights)
    
    planner_weights = torch.load(planner, map_location='cpu')
    ai_trainer.get_policy("p").model.load_state_dict(planner_weights)
    print(" Pesos cargados")
    
    results_ai_economist = evaluate_policy4(
        config,
        'AI Economist',
        trainer=ai_trainer,
        n_episodes=1
    )
    plot_tax_comparison(results_ai_economist, output_dir=".")
    
    ai_trainer.stop()
    ray.shutdown()

