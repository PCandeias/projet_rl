from gym_runner import GymRunner
from dqn_solver import DqnSolver
import numpy as np
import gym


class CartPoleRunner(GymRunner):
    def create_environment(self):
        self.env = gym.make('CartPole-v1')

    def create_agent(self):
        self.agent = DqnSolver(action_size=2, observation_size=4)

    def preprocess_state(self, state):
        return np.reshape(state, [1,4])


runner = CartPoleRunner()
runner.run(n_episodes=2000, train=True, render=False, goal_score=350)
runner.run(n_episodes=10, train=False, render=True)
