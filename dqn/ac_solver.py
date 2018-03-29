import numpy as np
from collections import deque
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import load_model
from keras.models import clone_model
import utility


class AcSolver(object):
    def __init__(self, observation_size, action_size, gamma=0.97, alpha=0.01, alpha_decay=0.01, memory_size=10000, 
            eps=0.01, batch_size=64, freeze_target_frequency=500, verbose=False, load_filename = None):
        print(alpha)
        self.memory = deque(maxlen=memory_size)
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.verbose = verbose
        self.replay_count = 0 # keep track of number of replay calls
        self.freeze_target_frequency = freeze_target_frequency # number of replay calls between each target Q-network
        self.eps = eps
        if (load_filename is not None and utility.file_exists(utility.models_directory + load_filename + "_c_ac.h5") 
                and utility.file_exists(utility.models_directory + load_filename + "_a_ac.h5")):
            self.load_model(load_filename)
        else:
            self.build_model(alpha, alpha_decay)

    def load_model(self, load_filename):
        print("Loading existing model...")
        self.model_critic = load_model(utility.models_directory + load_filename + "_c_ac.h5")
        self.model_actor = load_model(utility.models_directory + load_filename + "_a_ac.h5")
        self.target_model_critic = utility.copy_model(self.model_critic)

    def save_model(self, save_filename):
        self.model_critic.save(utility.models_directory + save_filename + "_c_ac.h5")
        self.model_actor.save(utility.models_directory + save_filename + "_a_ac.h5")

    def build_model(self, alpha, alpha_decay):
        self.model_critic = Sequential()
        self.model_critic.add(Dense(units=64, activation='tanh', input_dim=self.observation_size))
        self.model_critic.add(Dense(units=64, activation='tanh'))
        self.model_critic.add(Dense(units=self.action_size, activation='linear'))
        self.model_critic.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha, decay=alpha_decay))
        self.model_actor = Sequential()
        self.model_actor.add(Dense(units=64, activation='tanh', input_dim=self.observation_size))
        self.model_actor.add(Dense(units=64, activation='tanh'))
        self.model_actor.add(Dense(units=self.action_size, activation='softmax'))
        self.model_actor.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=alpha, decay=alpha_decay))
        self.target_model_critic = utility.copy_model(self.model_critic)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_values(self, state):
        return self.model_critic.predict(state)

    def get_probabilities(self, state):
        return self.model_actor.predict(state)

    """
    Chose an action using the probabilities provided by Policy gradient. With probability eps, pick a random 
    action (to try to avoid getting stuck in local optima)
    """
    def select_action(self, state, eps=None):
        prob = self.model_actor.predict(state)
        return np.random.choice(self.action_size, p=prob[0])

    # Train the agent in a given mini_batch of previous (state,action,reward,next_state)
    def replay(self):
        self.replay_count += 1
        batch_size = min(self.batch_size, len(self.memory))
        x_batch, y_batch = [], []
        actions, weights = [], []
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            y_train = self.model_critic.predict(state)
            y_train[0][action] = reward if done else reward + self.gamma * np.max(self.target_model_critic.predict(next_state))
            x_batch.append(state[0])
            y_batch.append(y_train[0])
            actions.append(action)
            reward = self.target_model_critic.predict(state)[0][action]  # should it use target or current model?
            weights.append(reward)
        weights = np.array(weights)
        weights = (weights - np.mean(weights)) / np.std(weights)  # normalize weights
        self.model_critic.fit(np.array(x_batch), np.array(y_batch), batch_size=batch_size, verbose=self.verbose)
        self.model_actor.fit(np.array(x_batch), np_utils.to_categorical(np.array(actions), self.action_size), sample_weight=weights, batch_size=batch_size, verbose=self.verbose)
        if self.replay_count % self.freeze_target_frequency == 0:
            self.target_model_critic = utility.copy_model(self.model_critic)
