from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque

class MultiFishermanRunner(MultiGymRunner):
    def _create_environment(self):
        self.env = gym.make('Fisherman-v0')
        self.env.set_environment_variables(max_stock=10, initial_stock=10, population=10, n_agents=self.n_agents,
                growth_rate=2, max_steps=10)

    def get_action_size(self):
        return 2

    def get_observation_size(self):
        return 1

    def _reset_metrics(self, r_episodes=False):
        self.avg_scores = []
        if r_episodes:
            self.scores_episodes = []
            self.scores_recent = deque(maxlen=100)
            self.steps_episodes = []

    def _display_metrics(self, ep_number):
        print("Episode: %d Average score: %f Last: %f" % (ep_number, np.mean(self.scores_recent), self.scores_recent[-1]))
            
    def _update_metrics(self, step, state, actions, rewards, next_state, done, score):
        self.avg_scores.append(np.mean(rewards))
        if done:
            ep_score = np.sum(self.avg_scores)
            self.avg_scores = [] # reset episode scores
            self.scores_episodes.append(ep_score)
            self.scores_recent.append(ep_score)
            self.steps_episodes.append(step)
            

    def get_metrics(self):
        return self.steps_episodes, self.scores_episodes

runner = MultiFishermanRunner(10, agent_mode='pg')
print("Predictions before")
print(runner.get_predictions(np.array([5,10])))
avg_steps, avg_scores = runner.run(n_episodes=100, train=True, verbose=True)
print("Predictions after 1 training")
print(runner.get_predictions(np.array([5,10])))
print("Steps/Scores in training")
print(avg_steps, avg_scores)
avg_steps, avg_scores = runner.run(n_episodes=100, train=True, verbose=True)
print("Predictions after 2 training")
print(runner.get_predictions(np.array([5,10])))
print("Steps/Scores in training")
print(avg_steps, avg_scores)
avg_steps, avg_scores = runner.run(n_episodes=100, train=True, verbose=True)
print("Predictions after 2 training")
print(runner.get_predictions(np.array([5,10])))
print("Steps/Scores in training")
print(avg_steps, avg_scores)
avg_steps, avg_scores = runner.run(n_episodes=1, train=False, render=True)
print("Steps/Scores in testing")
print(avg_steps, avg_scores)
