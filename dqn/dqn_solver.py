import numpy as np
from collections import deque
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.models import clone_model
import utility



class DqnSolver(object):
    def __init__(self, observation_size, action_size, gamma=0.97, eps=1.0, eps_decay=0.9995, eps_min=0.1, alpha=0.01,
                 alpha_decay=0.01, memory_size=100000, batch_size=64, freeze_frequency=500, verbose=False, load_filename = None):
        self.memory = deque(maxlen=memory_size)
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.verbose = verbose
        self.freeze_frequency = freeze_frequency
        self.replay_count = 0
        if load_filename is not None and utility.file_exists(utility.models_directory + "dqn_" + load_filename + ".h5"):
            self.load_model(load_filename)
            self.eps = eps_min
        else:
            self.build_model(alpha, alpha_decay)

    def load_model(self, load_filename):
        self.model = load_model(utility.models_directory + "dqn_" + load_filename + ".h5")
        self.target_model = clone_model(self.model)

    def save_model(self, save_filename):
        self.model.save(utility.models_directory + "dqn_" + save_filename + ".h5")

    def build_model(self, alpha, alpha_decay):
        self.model = Sequential()
        self.model.add(Dense(units=200, activation='tanh', input_dim=self.observation_size))
        self.model.add(Dense(units=200, activation='tanh'))
        self.model.add(Dense(units=self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha, decay=alpha_decay))
        self.target_model = self.model

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_values(self, state):
        return self.model.predict(state)

    # select an action according to an eps-greedy policy
    def select_action(self, state, eps=None):
        if eps is None:
            eps = self.eps
        return np.random.randint(0, self.action_size) if eps >= np.random.rand() else np.argmax(self.model.predict(state))

    # Train the agent in a given mini_batch of previous (state,action,reward,next_state)
    def replay(self):
        self.replay_count += 1
        batch_size = min(self.batch_size, len(self.memory))
        x_batch, y_batch = [], []
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            y_train = self.target_model.predict(state)
            y_train[0][action] = reward if done else reward + self.gamma * np.max(self.target_model.predict(next_state))
            x_batch.append(state[0])
            y_batch.append(y_train[0])
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=batch_size, verbose=self.verbose)
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
        if self.replay_count % self.freeze_frequency == 0:
            self.target_model = clone_model(self.model)



