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
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.loss_history = [] # For plotting loss later
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        
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

        # Sample a random batch of experiences from memory
        minibatch = random.sample(self.memory, self.batch_size)
        x = []
        y = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Compute target using the Q-learning equation
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0))

            # Get the predicted Q-values for the current state
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)

            target_f[0][action] = target  # Update Q-value for the action taken

            # Train the model on the updated Q-values
            x.append(state)
            y.append(target_f)

        x = tf.stack(x)
        y = tf.stack(y)
        

        history = self.model.fit(x=x, y=y, epochs=1, verbose=1, batch_size=self.batch_size)
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
        self.model.save_weights(name)
    
    def plot_loss(self):
        # Plot the loss over iterations
        plt.plot(self.loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss Over Iterations')
        plt.show()