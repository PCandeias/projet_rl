import gym
from collections import deque
import numpy as np
from live_graph import LiveGraph
from ac_solver import AcSolver

class GymRunner(object):
    def __init__(self):
        self.labels = None
        self.create_environment()
        self.create_agent()

    def preprocess_reward(self, reward):
        return reward

    def preprocess_state(self, state):
        return state

    def create_environment(self):
        raise NotImplementedError

    def create_agent(self, gamma=1.0, eps=1.0, eps_decay=0.99, eps_min=0.05, alpha=0.01, alpha_decay=0.01):
        # self.agent = DqnSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                # gamma=gamma, eps=eps, eps_decay=eps_decay, eps_min=eps_min, alpha=alpha, alpha_decay=alpha_decay)
        self.agent = AcSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                gamma=gamma,  alpha=alpha, alpha_decay=alpha_decay)

    def get_action_size(self):
        raise NotImplementedError

    def get_observation_size(self):
        raise NotImplementedError

    def run(self, n_episodes, train=False, render=False, goal_score=None, verbose=False):
        recent_scores = deque(maxlen=100)
        all_scores = np.zeros(n_episodes)
        avg_epochs = []
        for e in range(n_episodes):
            score = 0
            steps = 0
            done = False
            state = self.preprocess_state(self.env.reset())
            while not done:

                action = self.agent.select_action(state)
                if render:
                    self.env.render()
                next_state, reward, done, info = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                score += reward
                steps += 1
                self.agent.store(state, action, reward, next_state, done)
                state = next_state
            # Report the current average score after 100 episodes
            if verbose and e % 100 == 0 and e > 0:
                avg = np.mean(np.array(recent_scores))
                print("Episode: %d Score: %d" % (e, avg))
                avg_epochs.append(avg)
                # If achieved the goal reward, stop training
                if goal_score and avg > goal_score:
                    break
            if train:
                self.agent.replay()
            recent_scores.append(score)
            all_scores[e] = score
        return np.mean(np.array(recent_scores))
