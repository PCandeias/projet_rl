import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from replay_buffer import ReplayBuffer
import utility
import time


class DqnSolver(object):
    def __init__(self, observation_size, action_size, gamma=0.97, eps=1.0, eps_decay=0.9995, eps_min=0.1, alpha=0.01,
                 memory_size=100000, batch_size=64, double_q=False, freeze_target_frequency=500, verbose=False,
                 load_filename=None):
        self.memory = ReplayBuffer(max_len=memory_size)
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.double_q = double_q
        self.verbose = verbose
        self.freeze_target_frequency = freeze_target_frequency # number of replay calls between each target Q-network
        self.replay_count = 0  # keep track of number of replay calls
        # If trying to load a model from file and file found, load it
        if load_filename is not None and utility.file_exists(utility.models_directory + load_filename + "_dqn.h5"):
            self.load_model(load_filename)
            self.eps = eps_min # If using already partially trained agent, start from eps_min
        else:
            self.build_model(alpha)

    def load_model(self, load_filename):
        print("Loading existing model...")
        self.model = load_model(utility.models_directory + load_filename + "_dqn.h5")
        self.target_model = utility.copy_model(self.model)

    def save_model(self, save_filename):
        self.model.save(utility.models_directory + save_filename + "_dqn.h5")

    def build_model(self, alpha):
        self.model = Sequential()
        self.model.add(Dense(units=200, activation='tanh', input_dim=self.observation_size))
        self.model.add(Dense(units=200, activation='tanh'))
        self.model.add(Dense(units=self.action_size))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha))
        self.target_model = utility.copy_model(self.model) # avoid using target Q-network for first iterations

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # get the predictions for a given state
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
        # Get a batch of state-transitions
        mini_batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, done = zip(*mini_batch)
        y_batch = self.model.predict(np.array(states, copy=False))
        model_pred_after = self.model.predict(np.array(next_states, copy=False))
        target_pred_after = self.target_model.predict(np.array(next_states, copy=False))
        if self.double_q:
            y_batch[np.arange(batch_size),actions] = np.array(rewards, copy=False) + self.gamma * np.invert(np.array(done, copy=False)) * target_pred_after[np.arange(batch_size),np.argmax(model_pred_after, axis=1)]
        else:
            y_batch[np.arange(batch_size),actions] = np.array(rewards, copy=False) + self.gamma * np.invert(np.array(done, copy=False)) * np.max(target_pred_after, axis=1)
        # Train the model
        self.model.fit(np.array(states), np.array(y_batch), batch_size=batch_size, verbose=self.verbose)
        self.eps = max(self.eps * self.eps_decay, self.eps_min) # update eps
        if self.replay_count % self.freeze_target_frequency == 0:
            self.target_model.set_weights(self.model.get_weights())



