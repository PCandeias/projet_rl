from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque

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

# runner = MultiLunarLanderRunner(1, agent_mode='dqn', save_filename='dqn_lunarlander.h5', load_filename='dqn_lunarlander.h5', save_frequency=10000)
runner = MultiLunarLanderRunner(1, agent_mode='dqn',  save_filename='lunar_lander', save_frequency=10000)
runner.run(n_episodes=10000000, train=True, verbose=True)
runner.run(n_episodes=100, train=False)
