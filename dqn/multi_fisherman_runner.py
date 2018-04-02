from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque
import tensorflow as tf
from keras.backend import tensorflow_backend as K

class MultiFishermanRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('Fisherman-v0')
        self.env.set_environment_variables(max_stock=3, initial_stock=3, population=3, n_agents=self.n_agents,
                growth_rate=2, max_steps=100)

    def get_action_size(self):
        return 2

    def get_observation_size(self):
        return 1


with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=16)) as sess:
    K.set_session(sess)

    runner = MultiFishermanRunner(n_agents=3, agent_mode='pg', save_filename='fisherman', load_filename='fisherman',
                                 save_frequency=50000, replay_start_size=100, gamma=0.99, eps=1.0, eps_decay=0.9995,
                                 eps_min=0.05, alpha=0.00000025, memory_size=50000, batch_size=32,
                                 freeze_target_frequency=10000, double_q=True, verbose=False)
    runner.run(n_episodes=1000000, train=True, verbose=True, display_frequency=1000)
    print(runner.get_predictions([2]))
