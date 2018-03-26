from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque

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
        return episode_number >= 100 and np.mean(self.scores_recent) >= 495

#runner = MultiCartpoleRunner(1, agent_mode='pg', load_filename='cartpole', save_filename='cartpole')
#runner = MultiCartpoleRunner(1, agent_mode='dqn')
runner = MultiCartpoleRunner(1, agent_mode='ac',  save_filename='cartpole', load_filename='cartpole',
        save_frequency=1000)
runner.run(n_episodes=200000, train=True, verbose=True, display_frequency=1000)
runner.run(n_episodes=100, train=False)
