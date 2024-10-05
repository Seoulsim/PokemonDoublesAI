import numpy as np
import tensorflow as tf
from keras import layers, models
import random
from collections import deque


class PokemonDQN:
    def __init__(self, action_size = 9, state_size = 45, learning_rate=0.001, gamma=0.99, max_memory_size=2000, batch_size=32):
        self.state_size = state_size  # State: representation of the Pokémon battle, targeting, etc.
        self.action_size = action_size  # 4 moves and up to 5 switches (SINGLE BATTLES ONLY)
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
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
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return  np.random.rand(self.action_size)  # Exploration: Random action
        q_values = self.model.predict(np.expand_dims(state, axis=0))  # Exploitation: Choose action with highest Q-value

        # choose action based on probability using q_values
        # choice = np.random.choice(self.action_size, p=q_values)

        return q_values

    def replay(self):
        # If not enough samples in memory, return early
        if len(self.memory) < self.batch_size:
            return

        # Sample a random batch of experiences from memory
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Compute target using the Q-learning equation
                target = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0))[0])

            # Get the predicted Q-values for the current state
            target_f = self.model.predict(np.expand_dims(state, axis=0))
            target_f[0][action] = target  # Update Q-value for the action taken

            # Train the model on the updated Q-values
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # Load model from file
        self.model.load_weights(name)

    def save(self, name):
        # Save model to file
        self.model.save_weights(name)