from multi_gym_runner import MultiGymRunner
import numpy as np
import gym

class MultiCartpoleRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('Fisherman-v0')
        self.env.set_environment_variables(100, 100, 100, self.n_agents, 2, 200)

    def get_action_size(self):
        return 2

    def get_observation_size(self):
        return 1

    def _reset_metrics(self, r_episodes=False):
        self.avg_scores = []
        if r_episodes:
            self.avg_episodes = []
            
    def _preprocess_state(self, state):
        return np.reshape(state, [1, self.get_observation_size()])

    def get_metrics(self):
        return self.avg_episodes, self.avg_scores

    def _update_metrics(self, state, actions, rewards, next_state, done, score):
        self.avg_scores.append(np.mean(rewards))
        if done:
            self.avg_episodes.append(np.mean(self.avg_scores))

runner = MultiCartpoleRunner(100)
runner.run(n_episodes=1000, train=True)
runner.run(n_episodes=1, train=False)
