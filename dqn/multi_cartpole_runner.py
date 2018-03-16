from multi_gym_runner import MultiGymRunner
import gym

class MultiCartpoleRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('Fisherman-v0')
        self.env.set_environment_variables(10000, 10000, 10000, self.n_agents, 2, 200)

    def get_action_size(self):
        return 2

    def get_observation_size(self):
        return 1


runner = MultiCartpoleRunner(10000)
runner.run(n_episodes=1, train=False)
