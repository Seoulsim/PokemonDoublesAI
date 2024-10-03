import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import keras
from keras import layers, models
from collections import deque
import gym
import random

class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, max_memory_size=2000, batch_size=32):

        # State size
        self.state_size = state_size #
        self.action_size = action_size
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
        model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))  # Output layer for Q-values (one per action)

        # Compile model with Mean Squared Error loss and Adam optimizer
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration: Random action
        q_values = self.model.predict(np.expand_dims(state, axis=0))  # Exploitation: Choose action with highest Q-value
        return np.argmax(q_values[0])

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

# Example usage in a Reinforcement Learning environment
import gym

def train_dqn(episodes=1000):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQN(state_size=state_size, action_size=action_size)

    for e in range(episodes):
        state = env.reset()
        done = False
        time = 0

        while not done:
            action = agent.act(state)  # Choose action using DQN
            next_state, reward, done, _ = env.step(action)

            # Reward adjustment for termination cases
            reward = reward if not done else -10

            # Store the experience in memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            time += 1

            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break

        # Train the agent using the replay method
        agent.replay()

    # Save trained model weights
    agent.save('dqn_cartpole.h5')

# Train the DQN
train_dqn(episodes=1000)