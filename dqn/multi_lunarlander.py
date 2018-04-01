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

    def _stop_condition(self, episode_number):
        return episode_number >= 100 and np.mean(self.scores_episodes[-100:-1]) >= 180

with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=16)) as sess:
    K.set_session(sess)

    runner = MultiLunarLanderRunner(n_agents=1, agent_mode='dqn', save_filename = 'lunarlander', load_filename='lunarlander',
                                 save_frequency=250, replay_start_size=1000, gamma=0.99, eps=1.0, eps_decay=0.99995,
                                 eps_min=0.02, alpha=0.00025, memory_size=100000, batch_size=32,
                                 freeze_target_frequency=10000, verbose=False)
    runner.run(n_episodes=100000, train=True, verbose=True, render=False, display_frequency=100)
    runner.run(n_episodes=100, train=False)
