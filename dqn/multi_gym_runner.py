import gym
from collections import deque
from dqn_solver import DqnSolver


class MultiGymRunner(object):
    def __init__(self, n_agents=1):
        self.n_agents = n_agents
        self._reset_metrics()
        self._create_environment() 
        self._create_agents()

    def _create_environment(self):
        raise NotImplementedError

    def _create_agents(self, gamma=0.97, eps=1.0, eps_decay=0.995, eps_min=0.1, alpha=0.01,
                 alpha_decay=0.01, memory_size=10000, batch_size=64, verbose=False):
        self.agents = []
        print("Creating agents...")
        for i in range(self.n_agents):
            self.agents.append(DqnSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                gamma=gamma, eps=eps, eps_decay=eps_decay, eps_min=eps_min, alpha=alpha, alpha_decay=alpha_decay,
                memory_size=memory_size, batch_size=batch_size))
        print("Done creating agents.")


    def _preprocess_reward(self, reward):
        return reward

    def _preprocess_state(self, state):
        return state

    def get_action_size(self):
        raise NotImplementedError

    def get_observation_size(self):
        raise NotImplementedError
    
    def _reset_metrics(self):
        self.all_scores = []
        self.avg_scores = []
        self.eps_scores = []

    def _update_metrics(self, state, actions, rewards, next_state, done):
        return

    def get_metrics(self):
        return
    
    def _train_agents(self):
        for agent in agents:
            agent.replay()

    def _store_transitions(state, actions, rewards, next_state, done):
        for i, agent in enumerate(self.agents):
            agent.store(state, actions[i], rewards[i], next_state, done)

    def _select_actions(self, state):
        actions = []
        for agent in self.agents:
            actions.append(agent.select_action(state, agent.eps))
        return actions


    def run(self, n_episodes, train=False, render=False):
        self._reset_metrics()
        for e in range(n_episodes):
            step = 0
            done = False
            state = self._preprocess_state(self.env.reset())
            while not done:
                if render:
                    self.env.render()
                actions = self._select_actions(state)
                next_state, rewards, done, info = self.env.step(actions)
                self._update_metrics(state, actions, rewards, next_state, done)
                if train:
                    self._store_transitions(state, actions, rewards, next_state, done)
                state = next_state
                step += 1
            if train:
                self._train_agents()
        return self.get_metrics()
