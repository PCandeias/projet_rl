import numpy as np
import gym
from collections import deque
from dqn_solver import DqnSolver
from pg_solver import PgSolver
from ac_solver import AcSolver
import time


class MultiGymRunner(object):
    def __init__(self, n_agents=1, agent_mode='dqn'):
        self.n_agents = n_agents
        self.agent_mode = agent_mode
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
            if self.agent_mode == 'pg':
                self.agents.append(PgSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                    gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, memory_size=memory_size, batch_size=batch_size))
            elif self.agent_mode == 'ac':
                self.agents.append(AcSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                    gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, memory_size=memory_size, batch_size=batch_size))
            else:
                self.agents.append(DqnSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                    gamma=gamma, eps=eps, eps_decay=eps_decay, eps_min=eps_min, alpha=alpha, alpha_decay=alpha_decay,
                    memory_size=memory_size, batch_size=batch_size))
        print("Done creating agents.")


    def _preprocess_reward(self, reward):
        return reward

    def _preprocess_state(self, state):
        return np.reshape(state, [1, self.get_observation_size()])

    def get_action_size(self):
        raise NotImplementedError

    def get_observation_size(self):
        raise NotImplementedError

    def _reset_metrics(self, r_episodes=False):
        return

    def _update_metrics(self, step, state, actions, rewards, next_state, done, score):
        return

    def _display_metrics(self, ep_number):
        return

    def get_metrics(self):
        return

    def get_predictions(self, state):
        predictions = []
        for agent in self.agents:
            if self.agent_mode == 'pg':
                predictions.append(agent.get_probabilities(state))
            elif self.agent_mode == 'ac':
                predictions.append(agent.get_values(state), self.agent.get_probabilities(state))
            else:
                predictions.append(agent.get_values(state))
        return np.array(predictions)
    
    def _train_agents(self):
        for agent in self.agents:
            agent.replay()

    def _store_transitions(self, state, actions, rewards, next_state, done):
        for i, agent in enumerate(self.agents):
            agent.store(state, actions[i], rewards[i], next_state, done)

    def _select_actions(self, state):
        actions = []
        for agent in self.agents:
            actions.append(agent.select_action(state))
        return actions

    """
    Do any post processing on the action array required by the environment
    """
    def _process_actions(self, actions):
        return actions

    def run(self, n_episodes, train=False, render=False, verbose=False):
        total_training = 0
        total_running = 0
        self._reset_metrics(r_episodes = True)
        for e in range(n_episodes):
            step = 0
            score = np.zeros(self.n_agents, dtype=np.float64)
            done = False
            state = self._preprocess_state(self.env.reset())
            milli1 = time.time() * 1000
            while not done:
                if render:
                    self.env.render()
                actions = self._select_actions(state)
                next_state, rewards, done, info = self.env.step(self._process_actions(actions))
                next_state = self._preprocess_state(next_state)
                rewards = self._preprocess_reward(rewards)
                step += 1
                score += rewards
                self._update_metrics(step, state, actions, rewards, next_state, done, score)
                if train:
                    self._store_transitions(state, actions, rewards, next_state, done)
                state = next_state
            milli2 = time.time() * 1000
            total_running += (milli2 - milli1)
            if verbose and (e+1)%100 == 0:
                self._display_metrics(e)
            milli1 = time.time() * 1000
            if train:
                self._train_agents()
            milli2 = time.time() * 1000
            total_training += (milli2 - milli1)
        print("Time running: %f  Time training: %f" % (total_running, total_training))
        return self.get_metrics()
