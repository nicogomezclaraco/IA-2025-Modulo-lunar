import argparse
from DQN import DQNAgent
from lunar import LunarLanderEnv
import matplotlib.pyplot as plt
import numpy as np

# Gráfica: Recompensa obtenida por cada episodio, pérdida promedio por episodio,
# evolución de epsilon y steps por episodio.
def plot_results(rewards, losses, epsilon_history, steps_per_episode):
    window_size = 100
    avg_rewards = [np.mean(rewards[max(0, i-window_size):i+1]) for i in range(len(rewards))]
    avg_losses = [np.mean(losses[max(0, i-window_size):i+1]) for i in range(len(losses))]
    avg_steps = [np.mean(steps_per_episode[max(0, i-window_size):i+1]) for i in range(len(steps_per_episode))]

    plt.figure(figsize=(15, 10))

    # Recompensa por episodio
    plt.subplot(2, 2, 1)
    plt.plot(rewards, label="Recompensa por episodio")
    plt.plot(avg_rewards, label=f"Recompensa promedio ({window_size})", linestyle='--')
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.title("Recompensa")
    plt.legend()

    # Loss
    plt.subplot(2, 2, 2)
    plt.plot(losses, label="Loss por episodio", color='orange', alpha=0.4)
    plt.plot(avg_losses, label=f"Loss promedio ({window_size})", color='red', linestyle='--')
    plt.xlabel("Episodios")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Epsilon
    plt.subplot(2, 2, 3)
    plt.plot(epsilon_history, label="ε (epsilon)")
    plt.xlabel("Episodios")
    plt.ylabel("Epsilon")
    plt.title("Epsilon decay")
    plt.legend()

    # Steps por episodio
    plt.subplot(2, 2, 4)
    plt.plot(steps_per_episode, label="Steps por episodio", color='green', alpha=0.4)
    plt.plot(avg_steps, label=f"Steps promedio ({window_size})", color='blue', linestyle='--')
    plt.xlabel("Episodios")
    plt.ylabel("Número de steps")
    plt.title("Duración de episodios")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main(args):
    # Crea el entorno de lunar.
    env = LunarLanderEnv(render_mode="human" if args.render else None)

    # Inicializa el agente con los parámetros indicados por consola (o por defecto).
    agent = DQNAgent(
        lunar=env,
        episodes=args.episodes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        target_network_update_freq=args.target_network_update_freq
    )

    # Se inicia el entrenamiento y devuelve varias métricas.
    rewards, losses, epsilon_history, steps_per_episode = agent.train()

    # Grafica los resultados si se solicita
    if args.plot:
        plot_results(rewards, losses, epsilon_history, steps_per_episode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento del agente DQN para Lunar Lander")

    # Parámetros de entrenamiento.
    parser.add_argument("--episodes", type=int, default=1500, help="Número de episodios de entrenamiento")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Factor de descuento (gamma)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Valor inicial de epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="Valor mínimo de epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Factor de decaimiento de epsilon")
    parser.add_argument("--target_network_update_freq", type=int, default=10, help="Frecuencia de actualización de la red target")

    # Parámetros de adicionales.
    parser.add_argument("--render", action="store_true", help="Renderizar el entorno durante el entrenamiento")
    parser.add_argument("--plot", action="store_true", help="Mostrar gráficas de entrenamiento")

    args = parser.parse_args()
    main(args)
