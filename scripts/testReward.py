from src.environments.lunar import LunarLanderEnv
from src.agents.DQN import DQNAgent
import numpy as np

def test_reward_statistics_with_steps(agent=None, episodes=100, max_steps_per_episode=1000, steps_to_run_before_pause=0):
    lunar = LunarLanderEnv(render_mode=None)

    if agent is not None:
        agent.lunar = lunar

    rewards = []
    successes = 0


    for episode in range(episodes):
        counter, score = 0, 0

        while True:
            # Puedes descomentar esta parte si quieres pausar cada X pasos
            # if steps_to_run_before_pause != 0 and counter % steps_to_run_before_pause == 0:
            #     input("Presiona Enter para continuar...")

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
            
                if done and left_leg and right_leg:
                    successes += 1

                print(f"🎯 Episodio {episode + 1}/{episodes} terminado - Score: {score:.2f} - Steps: {counter} - Total Successes :{successes}")
                break

        rewards.append(score)

        # Reinicio del entorno
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
    print(f"Total Successes {successes}")

    return avg_reward, above_100, above_200


if __name__ == "__main__":
    lunar_env = LunarLanderEnv(render_mode=None)
    agent = DQNAgent(lunar_env, epsilon=0.0)

    try:
        agent.load_model("./saved_models/modelo_DQN_50avrg.weights.h5")
        print("✅ Modelo cargado correctamente")
        test_reward_statistics_with_steps(agent=agent, episodes=100, max_steps_per_episode=1000)
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
