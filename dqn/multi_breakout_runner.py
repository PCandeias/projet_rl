from multi_gym_runner import MultiGymRunner
import gym
import tensorflow as tf
from keras.backend import tensorflow_backend as K
import numpy as np

class MultiBreakoutRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('Breakout-ram-v0')

    def get_action_size(self):
        return 4

    def get_observation_size(self):
        return 128 

    def _preprocess_reward(self, reward):
        return [reward]

    def _process_actions(self, actions):
        return actions[0]

    def _stop_condition(self, episode_number):
        return episode_number >= 100 and np.mean(self.scores_episodes[-100:-1]) >= 220 and np.mean(self.scores_episodes[-5:-1]) >= 220

with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=16)) as sess:
    K.set_session(sess)

    runner = MultiBreakoutRunner(n_agents=1, agent_mode='dqn', save_filename='breakout', load_filename='breakout',
                                    save_frequency=500, replay_start_size=10000, gamma=0.99, eps=1.0, eps_decay=0.999995,
                                    eps_min=0.1, alpha=0.00025, memory_size=100000, batch_size=32,
                                    freeze_target_frequency=10000, double_q=True, verbose=False)
    if True:
        runner.run(n_episodes=100000, train=True, verbose=True, render=False, display_frequency=100)
    else:
        runner.run(n_episodes=100000, train=False, verbose=True, render=True, display_frequency=5)
