from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque
import tensorflow as tf
from keras.backend import tensorflow_backend as K

class MultiLunarLanderRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('LunarLander-v2')

    def get_action_size(self):
        return 4

    def get_observation_size(self):
        return 8 

    def _preprocess_reward(self, reward):
        return [reward]

    def _process_actions(self, actions):
        return actions[0]

with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=16)) as sess:
    K.set_session(sess)

    runner = MultiLunarLanderRunner(n_agents=1, agent_mode='dqn', save_filename = 'lunarlander', load_filename='lunarlander',
                                 save_frequency=250, replay_start_size=1000, gamma=1.0, eps=1.0, eps_decay=0.995,
                                 eps_min=0.05, alpha=5e-4, memory_size=50000, batch_size=32,
                                 freeze_target_frequency=1000, verbose=False)
    runner.run(n_episodes=100000, train=True, verbose=True, display_frequency=10)
    runner.run(n_episodes=100, train=False)
