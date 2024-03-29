from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque

class MultiEnduroRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('Enduro-ram-v0')

    def get_action_size(self):
        return 9

    def get_observation_size(self):
        return 128 

    def _preprocess_reward(self, reward):
        return [reward]

    def _reset_metrics(self, r_episodes=False):
        self.avg_scores = []
        if r_episodes:
            self.scores_episodes = []
            self.scores_recent = deque(maxlen=100)

    def _display_metrics(self, ep_number):
        print("Episode: %d Average score: %f" % (ep_number, np.mean(self.scores_recent)))
            
    def _update_metrics(self, step, state, actions, rewards, next_state, done, score):
        self.avg_scores.append(np.mean(rewards))
        if done:
            ep_score = np.sum(self.avg_scores)
            self.avg_scores = [] # reset episode scores
            self.scores_episodes.append(ep_score)
            self.scores_recent.append(ep_score)

    def get_metrics(self):
        return self.scores_episodes

    def _process_actions(self, actions):
        return actions[0] 

runner = MultiEnduroRunner(1, agent_mode='pg')
runner.run(n_episodes=10000, train=True,  render=True, verbose=True)
runner.run(n_episodes=100, train=False, render=True)
