import numpy as np
from collections import deque
import random
import keras
from keras.models import Sequential
from keras.layers import Dense


class DqnSolver(object):
    def __init__(self, observation_size, action_size, gamma=0.97, eps=1.0, eps_decay=0.995, eps_min=0.1, alpha=0.01,
                 alpha_decay=0.01, memory_size=10000, batch_size=64, verbose=False):
        self.memory = deque(maxlen=memory_size)
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.verbose = verbose
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=32, activation='tanh', input_dim=self.observation_size))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.alpha, decay=self.alpha_decay))

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
        batch_size = min(self.batch_size, len(self.memory))
        x_batch, y_batch = [], []
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            y_train = self.model.predict(state)
            y_train[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state))
            x_batch.append(state[0])
            y_batch.append(y_train[0])
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=batch_size, verbose=self.verbose)
        self.eps = max(self.eps * self.eps_decay, self.eps_min)



