from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque

class MultiCartpoleRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('Taxi-v2')

    def get_action_size(self):
        return 6 

    def get_observation_size(self):
        return 1

    def _preprocess_reward(self, reward):
        return [reward]

    def _process_actions(self, actions):
        return actions[0]

#runner = MultiCartpoleRunner(1, agent_mode='pg', load_filename='cartpole', save_filename='cartpole')
#runner = MultiCartpoleRunner(1, agent_mode='dqn')
runner = MultiCartpoleRunner(1, agent_mode='pg',  save_filename='taxi', load_filename='taxi')
runner.run(n_episodes=200000, train=True, verbose=True)
runner.run(n_episodes=100, train=False)
