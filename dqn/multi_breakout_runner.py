from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque

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

runner = MultiBreakoutRunner(1, agent_mode='pg', save_filename='breakout', load_filename='breakout', save_frequency=20000)
runner.run(n_episodes=10000000, train=True, verbose=True)
runner.run(n_episodes=100, train=False)
