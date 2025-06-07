import argparse
from src.agents.DQN import DQNAgent
from src.environments.lunar import LunarLanderEnv
import matplotlib.pyplot as plt

# Gráfica: Recompensa obtenida por cada episodio y pérdida promedio por episodio.
def plot_results(rewards, losses):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward por episodio")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.title("Recompensa")
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Pérdida promedio por episodio", color='orange')
    plt.xlabel("Episodios")
    plt.ylabel("Pérdida")
    plt.title("Loss")
    
    plt.tight_layout()
    plt.show()

def main(args):
    # Crea el entorno de lunar.
    env = LunarLanderEnv(render_mode="human" if args.render else None)

    # Inicializa el agente con los parámetros indicados por consola (o por defecto).
    agent = DQNAgent(lunar=env, episodes=args.episodes)

    # Se inicia el entrenamiento y de devuelve la recompensa y la pérdida del entrenamiento.
    rewards, losses = agent.train()

    # Grafica los resultados si se solicita
    if args.plot:
        plot_results(rewards, losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento del agente DQN para Lunar Lander")

    parser.add_argument("--episodes", type=int, default=2500, help="Número de episodios de entrenamiento")
    parser.add_argument("--render", action="store_true", help="Renderizar el entorno durante el entrenamiento")
    parser.add_argument("--plot", action="store_true", help="Mostrar gráficas de entrenamiento")

    args = parser.parse_args()
    main(args)
