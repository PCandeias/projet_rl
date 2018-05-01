from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque
import tensorflow as tf
from keras.backend import tensorflow_backend as K

class MultiFishermanRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('Fisherman-v2')
        self.env.set_environment_variables(max_stock=50, initial_stock=50, population=50, n_agents=self.n_agents,
                growth_rate=2, max_steps=100, n_groups=2)

    def get_action_size(self):
        return 2

    def get_observation_size(self):
        return 1 + self.env.n_groups


with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=16)) as sess:
    K.set_session(sess)

    runner = MultiFishermanRunner(n_agents=1, agent_mode='dqn', save_filename='fisherman', load_filename='fisherman',
                                 save_frequency=500, replay_start_size=20000, gamma=1.0, eps=1.0, eps_decay=0.99995,
                                 eps_min=0.1, alpha=0.0000000025, memory_size=1000000, batch_size=32,
                                 freeze_target_frequency=10000, double_q=True, verbose=False)
    runner.run(n_episodes=1000000, train=True, verbose=True, display_frequency=100, eps=0.00001)
    print(runner.get_predictions([2]))
