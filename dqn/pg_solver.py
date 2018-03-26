import numpy as np
from collections import deque
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import load_model
import utility

class PgSolver(object):
    def __init__(self, observation_size, action_size, gamma=0.97, alpha=0.01, eps=0.01,
                 alpha_decay=0.01, memory_size=10000, batch_size=64, verbose=False, save_filename=None, load_filename=None):
        self.memory = deque(maxlen=memory_size)
        self.ep_step = []
        self.observation_size = observation_size
        self.action_size = action_size
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.verbose = verbose
        if load_filename is not None and utility.file_exists(utility.models_directory +  load_filename + "_pg.h5"):
            self.load_model(load_filename)
        else:
            self.build_model()

    def load_model(self, load_filename):
        self.model = load_model(utility.models_directory + load_filename + "_pg.h5")

    def save_model(self, save_filename):
        self.model.save(utility.models_directory + save_filename + "_pg.h5")

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=200, activation='tanh', input_dim=self.observation_size))
        self.model.add(Dense(units=200, activation='tanh'))
        self.model.add(Dense(units=self.action_size, activation='softmax'))
        print("Creatin", self.alpha)
        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=self.alpha, decay=self.alpha_decay))

    def store(self, state, action, reward, next_state, done):
        self.ep_step.append((state,action,reward,next_state, done))
        if done:
            cur_reward = 0
            for state, action, reward, next_state, done in reversed(self.ep_step):
                cur_reward = self.gamma * cur_reward + reward
                self.memory.append((state, action, cur_reward))
            self.ep_step = []

    def get_probabilities(self, state):
        return self.model.predict(state)

    # select an action according using the probabilities given by the model
    def select_action(self, state, eps=None):
        if eps is None:
            eps = self.eps
        if eps >= np.random.rand():
            return np.random.randint(0, self.action_size)
        else:
            prob = self.model.predict(state)
            return np.random.choice(self.action_size, p=prob[0])

    # Train the agent in a given mini_batch of previous (state,action,reward,next_state)
    def replay(self):
        batch_size = min(self.batch_size, len(self.memory))
        x_batch, y_batch = [], []
        weights = []
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward in mini_batch:
            x_batch.append(state[0])
            y_batch.append(action)
            weights.append(reward)
        weights = np.array(weights)
        weights = (weights - np.mean(weights)) / np.std(weights)
        self.model.fit(np.array(x_batch), np_utils.to_categorical(np.array(y_batch), self.action_size),
                sample_weight=np.array(weights), batch_size=batch_size, verbose=self.verbose)


