import numpy as np
import random
import time

# import torch

import tensorflow as tf

from collections import deque

from lunar import LunarLanderEnv

# Lecturas interesantes: 
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (Playing atari with DQN)
# https://www.nature.com/articles/nature14236 (Human level control through RL)
# https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained

class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        # la capaa de entrada es la input_shape
        # le he preguntado al chat y me ha recomendado la funcion relu mejor que la sigmoide 
        # que es la que queria usar y me ha dicho que dos capas internas deberia estar bien
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(state_size,))
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size, activation='linear')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
    
class ReplayBuffer():
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size) # deque es una doble cola que permite añadir y quitar elementos de ambos extremos

    def push(self, state, action, reward, next_state, done):
        # insert into buffer
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent():
    def __init__(self, lunar: LunarLanderEnv, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                learning_rate=0.001, batch_size=64, 
                memory_size=10000, episodes=1500, 
                target_network_update_freq=10,
                replays_per_episode=1000):
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
        
        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        
        self.target_updt_freq = target_network_update_freq
        self.replays_per_episode = replays_per_episode
        
        # Initialize replay memory
        # a deque is a double sided queue that allows us to append and pop elements from both ends
        self.memory = ReplayBuffer(memory_size)
        
        # Initialize the environment
        self.lunar = lunar
        
        observation_space = lunar.env.observation_space
        action_space = lunar.env.action_space
        
        # La red neuronal debe tener un numero de parametros
        # de entrada igual al espacio de observaciones
        # y un numero de salida igual al espacio de acciones.
        # Asi como un numero de capas intermedias adecuadas.
        self.q_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=64  #elegir un tamaño de capa oculta // lo he buscado y este parece un buen standar
        )
        
        self.target_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=64  #elegir un tamaño de capa oculta // lo he buscado y este parece un buen standar
        )
        
        # Set weights of target network to be the same as those of the q network
        self.target_network.set_weights(self.q_network.get_weights())
      
      # lo normal aqui seria usar .SGD  pero el chat dice que converge mejor con .Adam tbn he estado mirando y podemos usar .RMSProp
        self.optimizer =tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        print(f"QNetwork:\n {self.q_network}")
          
    def act(self):
        """
        This function takes an action based on the current state of the environment.
        it can be randomly sampled from the action space (based on epsilon) or
        it can be the action with the highest Q-value from the model.
        """
        # usamos una e-greedy 

        if np.random.rand() <= self.epsilon:
            return self.lunar.env.action_space.sample()  
        else:
            # Asegúrate de que el estado tiene la forma correcta para la red neuronal
            state = np.array(state).reshape(1, -1)  # Convierte a array y da la forma (1, n)
            q_values = self.q_network.predict(verbose=0)
            return np.argmax(q_values[0]) 
    
    def update_model(self):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
        """
        
        if len(self.memory) < self.batch_size:
            return 0  # Not enough samples
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Calculate target Q values
        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q = np.amax(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Create mask for actions taken
        target_q = self.q_network.predict(states, verbose=0)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        target_q[batch_index, actions] = targets
        
        # Train the model
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            loss = tf.keras.losses.MSE(target_q, q_values)
            
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        
        return loss.numpy()
    

    def update_target_network(self):
        # copiar los pesos de la red q a la red objetivo
        self.target_network.set_weights(self.q_network.get_weights())
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        # guardar el modelo en el path indicado
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
        self.target_network.set_weights(self.q_network.get_weights())
        
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
        rewards_history = []
    
        for episode in range(self.episodes):
            state = self.lunar.reset()
            total_reward = 0
            done = False
        
            while not done:
                # Select and take action (aquí es donde necesitas pasar el state)
                action = self.act(state)  # self.act() ahora acepta el parámetro state
                next_state, reward, done,action = self.lunar.take_action()
            
                # Store experience
                self.memory.push(state, action, reward, next_state, done)
            
                state = next_state
                total_reward += reward
            
                # Train the model
                if len(self.memory) >= self.batch_size:
                    for _ in range(self.replays_per_episode):
                        self.update_model()
        
            # Update target network
            if episode % self.target_updt_freq == 0:
                self.update_target_network()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
            rewards_history.append(total_reward)
            avg_reward = np.mean(rewards_history[-100:])
        
            print(f"Episode: {episode+1}/{self.episodes}, Total Reward: {total_reward:.2f}, "
              f"Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        # Early stopping if solved
            if avg_reward >= 200:
                print("Environment solved!")
                self.save_model("lunar_lander_dqn.h5")
                break
            
        return rewards_history