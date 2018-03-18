from gym_runner import GymRunner
from dqn_solver import DqnSolver
import numpy as np
import gym


class MountainCarRunner(GymRunner):
    def create_environment(self):
        self.labels = ['Left', 'Do nothing', 'Right']
        self.env = gym.make('MountainCar-v0')

    def create_agent(self):
        self.agent = DqnSolver(action_size=3, observation_size=2)

    def preprocess_state(self, state):
        return np.reshape(state, [1,2])


runner = MountainCarRunner()
runner.run(n_episodes=500000, train=True, render=False)
runner.run(n_episodes=10, train=False, render=True)
