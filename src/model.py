import numpy as np
import tensorflow as tf
from keras import layers, models
import random
from collections import deque
import matplotlib.pyplot as plt


class PokemonDQN:
    # Action = 10 to account for pokemon switching into themselves
    def __init__(self, action_size = 10, state_size = 45, learning_rate=0.001, gamma=0.99, max_memory_size=2000, batch_size=32):
        self.state_size = state_size  # State: representation of the Pokémon battle, targeting, etc.
        self.action_size = action_size  # 4 moves and up to 5 switches (SINGLE BATTLES ONLY)
        self.memory = []
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.loss_history = [] # For plotting loss later
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05 
        self.epsilon_decay = 0.98
        
        # Build model
        self.model = self._build_model()

    def _build_model(self):
        # Neural network for approximating Q-values
        model = models.Sequential()
        model.add(layers.Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))  # Output layer for Q-values

        # Compile model with Mean Squared Error loss and Adam optimizer
        # model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        
        if np.random.rand() <= self.epsilon:
        # if np.random.rand() <= -100:

            return  np.random.rand(self.action_size).reshape(1, -1)  # Exploration: Random action
        
        q_values = self.model.predict(np.reshape(np.array(list(state.values())), (1, -1)), verbose=0)  # Exploitation: Choose action with highest Q-value

        # choose action based on probability using q_values
        # choice = np.random.choice(self.action_size, p=q_values)

        return q_values

    def replay(self):
        # If not enough samples in memory, return early
        if len(self.memory) < self.batch_size:
            return

        # # Sample a random batch of experiences from memory
        minibatch = random.sample(self.memory, min(self.batch_size*100, len(self.memory)))

        if len(self.memory) > self.batch_size*400:
            self.memory = random.sample(self.memory, self.batch_size*50)

        states = np.array([state for state, _, _, _, _ in minibatch])
        next_states = np.array([next_state for _, _, _, next_state, _ in minibatch])

        # Get predictions for next states (in one batch)
        next_qs = self.model.predict(next_states, verbose=0)

        # Get predictions for current states (in one batch)
        target_fs = self.model.predict(states, verbose=0)

        x = []
        y = []
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_qs[idx])

            target_f = target_fs[idx]
            target_f[action] = target  # Update the Q-value for the taken action

            # Accumulate inputs (x) and targets (y) for model training
            x.append(state)
            y.append(target_f)

        # Convert x and y to numpy arrays and train the model in one batch
        # self.model.fit(np.array(x), np.array(y), epochs=1, verbose=0)

        history = self.model.fit(np.array(x), np.array(y), epochs=10, verbose=1, batch_size=self.batch_size)
        self.loss_history.append(history.history['loss'][0])

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history['loss'][0]

    def load(self, name):
        # Load model from file
        self.model.load_weights(name)

    def save(self, name):
        # Save model to file
        self.model.save(name)
    
    def plot_loss(self):
        # Plot the loss over iterations
        plt.plot(self.loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss Over Iterations')
        plt.show()