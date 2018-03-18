from gym_runner import GymRunner
from dqn_solver import DqnSolver
import numpy as np
import gym
import parameter_finder


class CartPoleRunner(GymRunner):
    def create_environment(self):
        self.labels = ['Left', 'Right']
        self.env = gym.make('CartPole-v1')


    def get_action_size(self):
        return 2

    def get_observation_size(self):
        return 4

    def preprocess_state(self, state):
        return np.reshape(state, [1,4])


runner = CartPoleRunner()
runner.run(n_episodes=10000, train=True, render=False, goal_score=450, verbose=True)
runner.run(n_episodes=10, train=False, render=True, verbose=True)
