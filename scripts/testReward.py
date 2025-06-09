from src.environments.lunar import LunarLanderEnv
from src.agents.DQN import DQNAgent
import numpy as np
import argparse

def test_reward_statistics_with_steps(agent=None, episodes=100, max_steps_per_episode=1500, steps_to_run_before_pause=0):
    lunar = LunarLanderEnv(render_mode=None)

    if agent is not None:
        agent.lunar = lunar

    rewards = []
    successes = 0

    for episode in range(episodes):
        counter, score = 0, 0

        while True:
            if agent is not None:
                observation, reward, done, action = agent.act()
            else:
                action = lunar.env.action_space.sample()
                observation, reward, done = lunar.take_action(action)
            
            score += reward
            counter += 1

            if done or counter >= max_steps_per_episode:
                left_leg = observation[6] > 0.5
                right_leg = observation[7] > 0.5
                x, y = observation[0], observation[1]          
                vx, vy = observation[2], observation[3]       
                angular_vel = observation[5] 
                landed_center = abs(x) < 0.2                  
                soft_landing = abs(vy) < 0.6 and abs(vx) < 0.6  
                stable = abs(angular_vel) < 0.2 
            
                if done and left_leg and right_leg and landed_center and soft_landing  and stable:
                    successes += 1

                print(f"🎯 Episode {episode + 1}/{episodes} - Score: {score:.2f} - Steps: {counter} - Total Successes: {successes}")
                break

        rewards.append(score)
        if agent is not None:
            agent.lunar.reset()
        else:
            lunar.reset()

    lunar.close()

    # Estadísticas finales
    rewards = np.array(rewards)
    avg_reward = np.mean(rewards)
    above_100 = np.sum(rewards > 100)
    above_200 = np.sum(rewards > 200)

    print("\n📊 Estadísticas finales:")
    print(f"Media de rewards: {avg_reward:.2f}")
    print(f"Episodios con reward > 100: {above_100}/{episodes}")
    print(f"Episodios con reward > 200: {above_200}/{episodes}")
    print(f"Total Successes: {successes}")

    return avg_reward, above_100, above_200

def test_visual(agent=None, episodes=1, max_steps_per_episode=1500):
    lunar = LunarLanderEnv(render_mode="human")
    
    if agent is not None:
        agent.lunar = lunar

    for episode in range(episodes):
        counter, score = 0, 0
        observation = lunar.reset()

        while True:
            if agent is not None:
                observation, reward, done, action = agent.act()
            else:
                action = lunar.env.action_space.sample()
                observation, reward, done = lunar.take_action(action)

            score += reward
            counter += 1

            if done or counter >= max_steps_per_episode:
                print(f"🎯 Episodio {episode + 1}/{episodes} terminado - Score: {score:.2f} - Steps: {counter}")
                break

        if agent is not None:
            agent.lunar.reset()
        else:
            lunar.reset()

    lunar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Lunar Lander')
    parser.add_argument("--visual", action="store_true", help="Ejecutar en modo visual")
    parser.add_argument("--episodes", type=int, default=100, help="Número de episodios")
    args = parser.parse_args()

    lunar_env = LunarLanderEnv(render_mode=None)
    agent = DQNAgent(lunar_env, epsilon=0.0)
    
    try:
        agent.load_model("saved_models.modelo_DQN_final.weights.h5")
        print("✅ Modelo cargado correctamente")
        
        if args.visual:
            print("=== Modo Visual ===")
            test_visual(agent=agent, episodes=args.episodes)
        else:
            print("=== Modo Estadísticas ===")
            test_reward_statistics_with_steps(agent=agent, episodes=args.episodes)
            
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
