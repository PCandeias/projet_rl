import numpy as np
from collections import deque
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


class AcSolver(object):
    def __init__(self, observation_size, action_size, gamma=0.97, alpha=0.01,
                 alpha_decay=0.01, memory_size=10000, batch_size=64, verbose=False):
        self.memory = deque(maxlen=memory_size)
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.verbose = verbose
        self.build_model()
        self.EPS = 1e-8

    def build_model(self):
        self.model_critic = Sequential()
        self.model_critic.add(Dense(units=32, activation='tanh', input_dim=self.observation_size))
        self.model_critic.add(Dense(units=64, activation='tanh'))
        self.model_critic.add(Dense(units=self.action_size, activation='linear'))
        self.model_critic.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.alpha, decay=self.alpha_decay))
        self.model_actor = Sequential()
        self.model_actor.add(Dense(units=32, activation='tanh', input_dim=self.observation_size))
        self.model_actor.add(Dense(units=64, activation='tanh'))
        self.model_actor.add(Dense(units=self.action_size, activation='softmax'))
        self.model_actor.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=self.alpha, decay=self.alpha_decay))


    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_values(self, state):
        return self.model_critic.predict(state)

    def get_probabilities(self, state):
        return self.model_actor.predict(state)

    def select_action(self, state):
        prob = self.model_actor.predict(state)
        return np.random.choice(self.action_size, p=prob[0])

    # Train the agent in a given mini_batch of previous (state,action,reward,next_state)
    def replay(self):
        batch_size = min(self.batch_size, len(self.memory))
        x_batch, y_batch = [], []
        actions, rewards = [], []
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            y_train = self.model_critic.predict(state)
            y_train[0][action] = reward if done else reward + self.gamma * np.max(self.model_critic.predict(next_state))
            x_batch.append(state[0])
            y_batch.append(y_train[0])
            actions.append(action)
            reward = self.model_critic.predict(state)[0][action]
            rewards.append(max(self.EPS, reward) if reward >= 0 else min(-self.EPS, reward))
        self.model_critic.fit(np.array(x_batch), np.array(y_batch), batch_size=batch_size, verbose=self.verbose)
        # for state, action, reward, next_state, done in mini_batch:
        self.model_actor.fit(np.array(x_batch), np_utils.to_categorical(np.array(actions), self.action_size), sample_weight=np.array(rewards), batch_size=batch_size, verbose=self.verbose)
