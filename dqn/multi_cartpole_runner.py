from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
import tensorflow as tf
from keras.backend import tensorflow_backend as K

class MultiCartpoleRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('CartPole-v1')

    def get_action_size(self):
        return 2

    def get_observation_size(self):
        return 4

    def _preprocess_reward(self, reward):
        return [reward]

    def _process_actions(self, actions):
        return actions[0]

    def _stop_condition(self, episode_number):
        return episode_number >= 100 and np.mean(self.scores_episodes[-100:-1]) >= 495


with tf.Session(config=tf.ConfigProto(
                    intra_op_parallelism_threads=16)) as sess:
    K.set_session(sess)

    runner = MultiCartpoleRunner(n_agents=1, agent_mode='dqn', save_filename = 'cartpole', load_filename='cartpole',
                                 save_frequency=500, replay_start_size=10000, gamma=1.0, eps=1.0, eps_decay=0.995,
                                 eps_min=0.05, alpha=0.00025, memory_size=1000000, batch_size=32,
                                 freeze_target_frequency=1000, double_q=False, verbose=False)
    runner.run(n_episodes=100000, train=True, verbose=True, display_frequency=20)
    runner.run(n_episodes=100, train=False)
