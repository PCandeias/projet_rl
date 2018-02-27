import gym
from collections import deque
import numpy as np
from live_graph import LiveGraph

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

    def create_agent(self):
        raise NotImplementedError

    def run(self, n_episodes, train=False, render=False, goal_score=None):
        if render:
            value_graph = LiveGraph(lines=2, labels=self.labels)
            value_graph.show()
        recent_scores = deque(maxlen=100)
        all_scores = np.zeros(n_episodes)
        avg_epochs = []
        for e in range(n_episodes):
            score = 0
            steps = 0
            done = False
            state = self.preprocess_state(self.env.reset())
            while not done:

                action = self.agent.select_action(state, self.agent.eps)
                if render:
                    self.env.render()
                    values = self.agent.get_values(state)[0]
                    value_graph.add_value(values)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                score += reward
                steps += 1
                self.agent.store(state, action, reward, next_state, done)
                state = next_state
            # Report the current average score after 100 episodes
            if e % 100 == 0 and e > 0:
                avg = np.mean(np.array(recent_scores))
                print("Episode: %d Eps: %f Score: %d" % (e, self.agent.eps, avg))
                avg_epochs.append(avg)
                # If achieved the goal reward, stop training
                if goal_score and avg > goal_score:
                    break
            if train:
                self.agent.replay()
            recent_scores.append(score)
            all_scores[e] = score
