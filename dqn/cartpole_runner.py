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
print("Finding params...")
params = parameter_finder.find(runner=runner, v_gamma=np.linspace(0.95, 1.0, num=5), v_eps=np.array([1.0]),
    v_eps_decay=np.linspace(0.99, 1.0, endpoint=False, num=5), v_eps_min=np.linspace(0.05, 0.1, num=5), 
    v_alpha=np.linspace(0.001, 0.01, num=5), v_alpha_decay=np.linspace(0.001, 0.01, num=5), verbose = True)
print(params)
runner.run(n_episodes=1000, train=True, render=False, goal_score=450, verbose=True)
runner.run(n_episodes=10, train=False, render=True, verbose=True)
