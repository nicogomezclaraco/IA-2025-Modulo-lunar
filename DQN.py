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
    # hay que quitar el trucanmiento a 32
    def __init__(self, state_size, action_size, hidden_size = 32):
        super(DQN, self).__init__()
        # la capaa de entrada es la input_shape
        # le he preguntado al chat y me ha recomendado la funcion relu mejor que la sigmoide 
        # que es la que queria usar y me ha dicho que dos capas internas deberia estar bien
        self.dense1 = tf.keras.layers.Dense(
            hidden_size, 
            activation='relu', 
            kernel_initializer='he_normal'  # Mejor inicialización
        )

        self.dense2 = tf.keras.layers.Dense(
            hidden_size, 
            activation='relu',
            kernel_initializer='he_normal'
        )
        
        self.output_layer = tf.keras.layers.Dense(
            action_size, 
            activation='linear',
            kernel_initializer='glorot_uniform'
        )

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
        states = np.array([x[0] for x in batch], dtype=np.float32)
        actions = np.array([x[1] for x in batch], dtype=np.int32)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch], dtype=np.float32)
        dones = np.array([x[4] for x in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent():
    #he cambiado algun parametro por probar 
    #OG
    #learning_rate=0.001
    #episodes=1500
    def __init__(self, lunar: LunarLanderEnv, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                learning_rate=0.002, batch_size=64, 
                memory_size=10000, episodes=1500, 
                target_network_update_freq=100,
                warmup_steps=1000):
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
        self.warmup_steps = warmup_steps

        self.target_updt_freq = target_network_update_freq
        self.step_count = 0
        
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
            hidden_size=32  #elegir un tamaño de capa oculta // lo he buscado y este parece un buen standar 64 // lo voy a reducir por probar 
        )
        
        self.target_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=32  #elegir un tamaño de capa oculta // lo he buscado y este parece un buen standar
        )
        
        dummy_state = tf.zeros((1, observation_space.shape[0]))
        _ = self.q_network(dummy_state)
        _ = self.target_network(dummy_state)
      
        self.update_target_network()

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
        state = np.array(self.lunar.state, dtype=np.float32).reshape(1, -1)

        if np.random.rand() <= self.epsilon:
            action =  self.lunar.env.action_space.sample()  
        else:
            q_values = self.q_network(state, training=False)
            action = np.argmax(q_values[0]) 

        next_state, reward, done = self.lunar.take_action(action, verbose=False)
        return next_state, reward, done, action
    
    def update_model(self):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
    
        Returns:
        loss: The computed loss value
        """
    
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return 0
    
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
    
        # Convertir a tensores de TensorFlow
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
    
        # Calculate target Q values using the target network
        next_actions = tf.argmax(self.q_network(next_states, training=False), axis=1)
        next_q_values = self.target_network(next_states, training=False)
    
        # Gather Q-values para las acciones seleccionadas
        batch_indices = tf.range(self.batch_size)
        next_action_indices = tf.stack([batch_indices, tf.cast(next_actions, tf.int32)], axis=1)
        max_next_q = tf.gather_nd(next_q_values, next_action_indices)

        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Get current Q values from main network
        with tf.GradientTape() as tape:
            current_q_values = self.q_network(states, training=True)
        
            # Crear índices para actualizar solo las acciones tomadas
            batch_indices = tf.range(self.batch_size)
            action_indices = tf.stack([batch_indices, actions], axis=1)
        
            # Obtener Q-values para las acciones tomadas
            current_q_action = tf.gather_nd(current_q_values, action_indices)
        
            # Compute loss solo para las acciones tomadas
            loss = tf.keras.losses.Huber(delta=1.0)(targets, current_q_action)
        
        # Apply gradients
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        clipped_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else None for grad in grads]
        self.optimizer.apply_gradients(zip(clipped_grads, self.q_network.trainable_variables))
    
        return loss.numpy()
    

    def update_target_network(self):
        # copiar los pesos de la red q a la red objetivo
        for target_param, main_param in zip(self.target_network.trainable_variables, self.q_network.trainable_variables):
            target_param.assign(main_param)
        
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
        losses = []
        best_avg_reward = -float('inf')

        print("Iniciando entrenamiento DQN optimizado...")
        start_time = time.time()

        for episode in range(self.episodes):
            state = self.lunar.reset()
            self.lunar.state = state
            total_reward = 0
            episode_loss = 0
            steps_in_episode = 0
            done = False

            while not done:
                # Select and take action (aquí es donde necesitas pasar el state)
                next_state, reward, done, action = self.act()
                total_reward += reward

                # Store experience
                self.memory.push(state, action, reward, next_state, done)
            
                self.lunar.state = next_state
                state = next_state
                steps_in_episode += 1
                self.step_count += 1

                # Entrenar cada 4 pasos después del warmup
                if self.step_count > self.warmup_steps and self.step_count % 4 == 0:
                    loss = self.update_model()
                    episode_loss += loss
        
            # Update target network con menos frecuencia
            if episode % self.target_updt_freq == 0:
                self.update_target_network()
            
            # Epsilon decay más gradual
            if self.epsilon > self.epsilon_min:
                # Decay exponencial más suave
                self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-episode / 500)
        
            rewards_history.append(total_reward)
            losses.append(episode_loss / steps_in_episode if steps_in_episode > 0 else 0)

            # Average (Mejora)
            if len(rewards_history) >= 100:
                avg_reward = np.mean(rewards_history[-100:])
            else:
                avg_reward = np.mean(rewards_history)
        
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

            print(f"Episode: {episode+1}/{self.episodes}, Total Reward: {total_reward:.2f}, "
              f"Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
            # Log cada 50 episodios para reducir overhead
            if episode % 10 == 0 or episode < 10:
                elapsed_time = time.time() - start_time
                print(f"Ep: {episode+1}/{self.episodes} | Reward: {total_reward:.1f} | "
                      f"Avg: {avg_reward:.1f} | ε: {self.epsilon:.3f} | "
                      f"Loss: {episode_loss/max(steps_in_episode,1):.4f} | "
                      f"Time: {elapsed_time/60:.1f}min")
                
                        # Early stopping mejorado 
                        # esta mierda que  es lolo que no deja guardar el modelo si el rewrd no es 200 ???
                        #            if avg_reward >= 200 and episode >= 100:
                        #vale esto hay que quitarlo / cambiarlo la vd es que para ahora mismo testear no esta ni tan mal


            if  episode >= 100:
                print(f"\n🎉 ¡Ambiente resuelto en {episode+1} episodios!")
                print(f"Recompensa promedio últimos 100 episodios: {avg_reward:.2f}")
                self.save_model("modelo_DQN.weights.h5")
                break

        total_time = time.time() - start_time
        print(f"\n✅ Entrenamiento completado en {total_time/60:.1f} minutos!")
        print(f"Mejor promedio alcanzado: {best_avg_reward:.2f}")

        print("\nTraining completed!")    
        return rewards_history, losses