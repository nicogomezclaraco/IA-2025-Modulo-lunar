# Otras librerias necesarias.
import numpy as np
import random
import time

# Permite crear y entrenar redes neuronales.
import tensorflow as tf

# Se importa ´deque´ que es una cola de doble extremo (se pueden sacar elementos de ambos extremos) para el buffer.
from collections import deque

# Se usa el módulo lunar ya definido.
from src.environments.lunar import LunarLanderEnv

# Lecturas interesantes: 
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (Playing atari with DQN)
# https://www.nature.com/articles/nature14236 (Human level control through RL)
# https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained

# Clase donde se define la red neuronal a partir de un modelo de TensorFlow.
class DQN(tf.keras.Model):
    # Se define la red neuronal.
    def __init__(self, state_size, action_size, hidden_size = 128):
        super(DQN, self).__init__()

        # Capas ocultas 1 y 2: con tamaño (hidden_size), función de activación de tipo relu (recomendada para redes neuronales, f(x) = max(0, x)) 
        # y 'he_normal' para tomar pesos aleatorios al inicio.
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation = 'relu', kernel_initializer = 'he_normal')
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation = 'relu', kernel_initializer = 'he_normal')

        # Capa output que nos devolverá los Q values: con tamaño (action_size), función de activación de tipo lineal (la salida es el valor de la red) 
        # y 'glorot_uniform' para tomar pesos ni muy grandes ni muy pequeños.
        self.output_layer = tf.keras.layers.Dense(action_size, activation = 'linear', kernel_initializer = 'glorot_uniform')

    # Se aplica el vector de 8 estados a la red neuronal (capa a capa) y se obtiene un vector de Q values para cada acción.
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Clase donde el agente guarda sus experiencias, una tupla (state, action, reward, next_state, done).
class ReplayBuffer():
    # Se fine el buffer de tipo ´deque´ que es una cola que guarda hasta buffer_size elementos.
    def __init__(self, buffer_size = 10000):
        self.buffer = deque(maxlen = buffer_size)

    # Añade al buffer una tupla (state, action, reward, next_state, done).
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Elige un conjunto de experiencias con tamaño batch_size aleatorio para entrenar (pop).
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([x[0] for x in batch], dtype = np.float32)
        actions = np.array([x[1] for x in batch], dtype = np.int32)
        rewards = np.array([x[2] for x in batch], dtype = np.float32)
        next_states = np.array([x[3] for x in batch], dtype = np.float32)
        dones = np.array([x[4] for x in batch], dtype = np.float32)
        return states, actions, rewards, next_states, dones
        
    # Número de experiencias guardadas en ese momento.
    def __len__(self):
        return len(self.buffer)

# Clase donde se define el agente DQN, los hiperparámetros que toma y sus métodos.
class DQNAgent():
     # Se inicializan las 2 redes neuronales (Q-Network y Target Network) y se inicializa el buffer.
    def __init__(self, lunar: LunarLanderEnv, gamma = 0.99, 
                epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.01,
                learning_rate = 0.0005, batch_size = 128, 
                memory_size = 100000, episodes = 2000, 
                target_network_update_freq = 400,
                warmup_steps = 10000):
        """
        Initialize the DQN agent with the given parameters.
        
        Parameters:
        lunar (LunarLanderEnv): The Lunar Lander environment instance.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for exploration rate.
        epsilon_min (float): Minimum exploration rate.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Size of the batch for experience replay.
        memory_size (int): Number of experiences stored on the replay memory.
        episodes (int): Number of episodes to train the agent.
        target_network_update_freq (int): Frequency of updating the target network.
        """
        
        # Se inicializan los hiperparámetros.
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        self.warmup_steps = warmup_steps
        self.target_updt_freq = target_network_update_freq
        self.step_count = 0
        
        # Se inicializa el buffer donde se guardan las experiencias.
        self.memory = ReplayBuffer(memory_size)
        
        # Se inicializa el módulo lunar.
        self.lunar = lunar

        # 8 variables del entorno que irán variando en cada estado:
        # Posición X, Posición Y,
        # Velocidad lineal X, Velocidad lineal Y,
        # Ángulo, Velocidad angular,
        # Contacto con la pierna izquierda y Contacto con la pierna derecha.
        observation_space = lunar.env.observation_space
                    
        # 4 acciones son las permitidas por el módulo lunar:
        # No hacer nada,
        # Encender el motor de orientación izquierdo, Encender el motor de orientación derecho,
        # Encender el motor principal.
        action_space = lunar.env.action_space
        
        # Se construyen la Q-Network principal y la network objetivo: con misma estructura de red neuronal.
        self.q_network = DQN(state_size = observation_space.shape[0], action_size = action_space.n, hidden_size = 64)
        self.target_network = DQN(state_size = observation_space.shape[0], action_size = action_space.n, hidden_size = 64)

        # Se fuerza la construcción de la red en TensorFlow.
        dummy_state = tf.zeros((1, observation_space.shape[0]))
        _ = self.q_network(dummy_state)
        _ = self.target_network(dummy_state)

        # Se copian los pesos de la Q-Network principal q_network a la network objetivo.
        self.update_target_network()

        # Aquí se define el optimizador que vamos a usar para entrenar la red. 
        # Adam ajusta dinámicamente el learning rate por parámetro.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
        print(f"QNetwork:\n {self.q_network}")

    # Copiar los pesos de la Q-Network principal a la network objetivo.
    def update_target_network(self):
        for target_param, main_param in zip(self.target_network.trainable_variables, self.q_network.trainable_variables):
            target_param.assign(main_param)
       
    # Algoritmo para realizar el entrenamiento del agente DQN.
    def train(self):

        """
        Train the DQN agent on the given environment for a specified number of episodes.
        The agent will interact with the environment, store experiences in memory, and learn from them.
        The target network will be updated periodically based on the update freq parameter.
        The agent will also decay the exploration rate (epsilon) over time.
        The training process MUST be logged to the console.    
        Returns:
        None
        """
        # Variables iniciales del entrenamiento.
        rewards_history = [] # Lista para guardar la recompensa total de cada episodio.
        losses = [] # Lista para guardar la pérdida promedio de cada episodio.
        best_avg_reward = -float('inf') # Guarda el mejor promedio de recompensa.
        start_time = time.time() # Para medir cuánto tiempo tarda el entrenamiento.

        for episode in range(self.episodes):
            # Se reinicia el entorno.
            state = self.lunar.reset()
            self.lunar.state = state
            total_reward = 0
            episode_loss = 0
            steps_in_episode = 0
            done = False

            while not done:
                # Elige una acción. Aplicando la política ε-greedy (exploración / explotación) y va acumulando la recompensa.
                next_state, reward, done, action = self.act()
                total_reward += reward

                # Guarda la experiencia de haber realizado esa acción en el replay buffer.
                self.memory.push(state, action, reward, next_state, done)

                # Se modifican las variables iniciales.
                self.lunar.state = next_state
                state = next_state
                steps_in_episode += 1
                self.step_count += 1

                # Al principio no entrena (esperas a tener experiencias suficientes, warmup).
                # Tras tener esa experiencia actualizas el modelo con un conjunto de experiencias.
                # Se acumulan las perdidas trás la actualización del modelo.
                if self.step_count > self.warmup_steps and self.step_count % 4 == 0:
                    loss = self.update_model()
                    episode_loss += loss
        
            # Se realiza una actualización de la network objetivo (evita entrenamiento inestable).
            if episode % self.target_updt_freq == 0:
                self.update_target_network()
            
            # A medida que entrena, el agente va dejando de explorar tanto y usa lo aprendido.
            if self.epsilon > self.epsilon_min:
                # Decay exponencial más suave
                self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-episode / 500)

            # Cuando termina el episodio guarda la puntuación obtenida hasta ese momento.
            rewards_history.append(total_reward)
            # Cuando termina el episodio guarda el promedio de pérdida por paso en ese episodio.
            losses.append(episode_loss / steps_in_episode if steps_in_episode > 0 else 0)

            # Se calcula la recompensa promedio obtenida hasta ahora.
            if len(rewards_history) >= 100:
                avg_reward = np.mean(rewards_history[-100:])
            else:
                avg_reward = np.mean(rewards_history)

            # Se guarda la mejor recompensa que se ha obtenido.
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

            # Log cada 50 episodios para reducir overhead
            if episode % 50 == 0 :
                elapsed_time = time.time() - start_time
                print(f"Ep: {episode+1}/{self.episodes} | Reward: {total_reward:.1f} | "
                      f"Avg: {avg_reward:.1f} | ε: {self.epsilon:.3f} | "
                      f"Loss: {episode_loss/max(steps_in_episode,1):.4f} | "
                      f"Time: {elapsed_time/60:.1f}min")
                
        # Se finaliza el entrenamiento
        total_time = time.time() - start_time
        print(f"\n✅ Entrenamiento completado en {total_time/60:.1f} minutos!")
        print(f"Mejor promedio alcanzado: {best_avg_reward:.2f}")
        print("\nTraining completed!")
        print("Guardando modelo")
        self.save_model("modelo_DQN.weights.h5")    
        return rewards_history, losses

    # Elige una acción siguiendo la política ε-greedy (exploración / explotación).
    def act(self):
        """
        This function takes an action based on the current state of the environment.
        it can be randomly sampled from the action space (based on epsilon) or
        it can be the action with the highest Q-value from the model.
        """
        # Obtiene el estado actual del entorno y lo convierte en un array de NumPy.
        state = np.array(self.lunar.state, dtype = np.float32).reshape(1, -1)

        # Se aplica una política ε-greedy (exploración / explotación)
        if np.random.rand() <= self.epsilon:
            # Si el número aleatorio [0,1] <= ε. Elige una acción aleatoria (explora).
            action =  self.lunar.env.action_space.sample()  
        else:
            # En otro caso. Obtiene los Q-values para cada acción posible y toma el máximo (explota).
            q_values = self.q_network(state, training = False) # [Q(s,a₀), Q(s,a₁), Q(s,a₂), Q(s,a₃)]
            action = np.argmax(q_values[0]) 

        # Toma la acción elegida y devuelve el siguiente estado, la recompensa y si se ha cumplido el objetivo.
        next_state, reward, done = self.lunar.take_action(action, verbose = False)
        return next_state, reward, done, action

    # Hace que la red Q-Network aprenda, usa el replay buffer a través de un minibatch de experiencias pasadas,
    # calcula los Q-values objetivos y los actuales, y calcula el error entre ambos.
    def update_model(self):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
    
        Returns:
        loss: The computed loss value
        """

        # No entrena si no existen suficientes datos guardados
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return 0

        # Toma del replay buffer un conjunto de experiencias de ejemplos
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
    
        # Convierte a tensores de TensorFlow esas experiencias de ejemplos.
        states = tf.convert_to_tensor(states, dtype = tf.float32)
        actions = tf.convert_to_tensor(actions, dtype = tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
        dones = tf.convert_to_tensor(dones, dtype = tf.float32)
    
        # Para cada estado futuro se calculan los Q-values, y se obtiene la ación con el Q-value más alto para cada estado.
        # Next actions es un array con las mejores acciones sugeridas por la red principal.
        next_actions = tf.argmax(self.q_network(next_states, training = False), axis = 1)
        # Se guardan además esos Q-values de cada estado, a través de la network objetivo ya que es más estable.
        next_q_values = self.target_network(next_states, training = False)
    
        # Clasifica para cada experiencia del batch realiza un conjunto (índice, mejor acción).
        batch_indices = tf.range(self.batch_size)
        next_action_indices = tf.stack([batch_indices, tf.cast(next_actions, tf.int32)], axis = 1)
        max_next_q = tf.gather_nd(next_q_values, next_action_indices)

        # Ecuación de Bellman (valor deseado que la red debería aprender).
        # Si el lander choca o aterriza bien no hay futuro, si el episodio no termina si consideramos el futuro.
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Get current Q values from main network
        with tf.GradientTape() as tape:
            # La red devuelve los Q-values para las acciones posibles en cada estado.
            current_q_values = self.q_network(states, training = True)
        
            # Crear índices para actualizar solo las acciones tomadas
            batch_indices = tf.range(self.batch_size)
            action_indices = tf.stack([batch_indices, actions], axis = 1)
        
            # Obtener Q-values para las acciones tomadas
            current_q_action = tf.gather_nd(current_q_values, action_indices)
        
            # Computar pérdidas solo para las acciones tomadas usando Huber loss, parecida a MSE.
            loss = tf.keras.losses.Huber(delta = 1.0)(targets, current_q_action)
        
        # Se actualizan los pesos usando el gradiente calculado.
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        # Realiza un clip sobre los gradientes para evitar inestabilidad.
        clipped_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else None for grad in grads]
        # Actualiza los pesos usando de nuevo un optimizador Adam.
        self.optimizer.apply_gradients(zip(clipped_grads, self.q_network.trainable_variables))
    
        return loss.numpy()
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        # guardar el modelo en el path indicado
        # esto solo guarda los pesos por eso lo he cambiado         
        self.q_network.save_weights(path)
    
    def load_model(self, path):
        """
        Load the model weights from a file.
        Parameters:
        path (str): The path to load the model weights from.
        Returns:
        None
        """
        # cargar el modelo desde el path indicado
        self.q_network.load_weights(path)
        self.update_target_network()
