import numpy as np
import gym
from collections import deque
from dqn_solver import DqnSolver
from pg_solver import PgSolver
from ac_solver import AcSolver
import time


class MultiGymRunner(object):
    def __init__(self, n_agents=1, agent_mode='dqn', save_filename = None, load_filename=None, save_frequency = 20000, replay_start_size=10000,
                 gamma=0.99, eps=1.0, eps_decay=0.995, eps_min=0.05, alpha=0.0002, alpha_decay=0.01, memory_size=1000000, batch_size=64, verbose=False):
        self.n_agents = n_agents
        self.agent_mode = agent_mode
        self._create_environment()
        self.agents = []
        self.save_filename = save_filename
        self.save_frequency = save_frequency
        self.replay_start_size = replay_start_size
        if load_filename is not None:
            self._load_agents(load_filename, gamma=gamma, alpha=alpha, alpha_decay=alpha_decay,  eps=eps,
                              eps_decay=eps_decay, eps_min=eps_min, memory_size=memory_size, batch_size=batch_size, verbose=verbose)
        else:
            self._create_agents(gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, eps=eps, eps_decay=eps_decay,
                                eps_min=eps_min,memory_size=memory_size, batch_size=batch_size, verbose=verbose)

    def _create_environment(self):
        raise NotImplementedError

    def _save_agents(self, save_filename):
        print('saving...')
        for i in range(self.n_agents):
            self.agents[i].save_model("" + str(i) + save_filename)

    def _load_agents(self, load_filename, gamma, eps, eps_decay, eps_min, alpha, alpha_decay,
                     memory_size, batch_size, verbose):
        print('loading...')
        self.agents = []
        for i in range(self.n_agents):
            if self.agent_mode == 'pg':
                self.agents.append(PgSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(), eps=0.01,
                                            gamma=gamma, memory_size=memory_size, batch_size=batch_size,
                                            load_filename=str(i) + load_filename, verbose=verbose))
            elif self.agent_mode == 'ac':
                self.agents.append(AcSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                                            gamma=gamma, memory_size=memory_size, batch_size=batch_size,
                                            load_filename=str(i) + load_filename, verbose=verbose))
            else:
                self.agents.append(DqnSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                                             gamma=gamma, eps=eps, eps_decay=eps_decay, eps_min=eps_min, alpha=alpha, alpha_decay=alpha_decay,
                                             memory_size=memory_size, batch_size=batch_size, load_filename=str(i) + load_filename, verbose=verbose))

    def _create_agents(self, gamma, eps, eps_decay, eps_min, alpha,
                       alpha_decay, memory_size, batch_size, verbose):
        print("Creating agents...")
        self.agents = []
        for i in range(self.n_agents):
            if self.agent_mode == 'pg':
                self.agents.append(PgSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                    gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, memory_size=memory_size, batch_size=batch_size, verbose=verbose))
            elif self.agent_mode == 'ac':
                self.agents.append(AcSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                    gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, memory_size=memory_size, batch_size=batch_size, verbose=verbose))
            else:
                self.agents.append(DqnSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                    gamma=gamma, eps=eps, eps_decay=eps_decay, eps_min=eps_min, alpha=alpha, alpha_decay=alpha_decay,
                    memory_size=memory_size, batch_size=batch_size, verbose=verbose))
        print("Done creating agents.")


    def _preprocess_reward(self, reward):
        return reward

    def _preprocess_state(self, state):
        return np.reshape(state, [1, self.get_observation_size()])

    def get_action_size(self):
        raise NotImplementedError

    def get_observation_size(self):
        raise NotImplementedError

    def _reset_metrics(self, display_frequency, r_episodes=False):
        self.avg_scores = []
        if r_episodes:
            self.scores_episodes = []
            self.scores_recent = deque(maxlen=display_frequency)

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

    def get_predictions(self, state):
        predictions = []
        for agent in self.agents:
            if self.agent_mode == 'pg':
                predictions.append(agent.get_probabilities(state))
            elif self.agent_mode == 'ac':
                predictions.append((agent.get_values(state), agent.get_probabilities(state)))
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

    def _stop_condition(self, episode_number):
        return False

    """
    Do any post processing on the action array required by the environment
    """
    def _process_actions(self, actions):
        return actions

    def run(self, n_episodes, train=False, render=False, verbose=False, display_frequency=1000):
        total_steps = 0
        self._reset_metrics(r_episodes = True, display_frequency=display_frequency)
        for e in range(n_episodes):
            step = 0
            score = np.zeros(self.n_agents, dtype=np.float64)
            done = False
            state = self._preprocess_state(self.env.reset())
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
            total_steps += step
            if verbose and (e+1)%display_frequency == 0:
                self._display_metrics(e)
            if train:
                if self._stop_condition(e):
                    break
                elif total_steps >= self.replay_start_size:
                    self._train_agents()
            if self.save_filename is not None and train and (e+1) % self.save_frequency == 0:
                self._save_agents(self.save_filename)
        if self.save_filename is not None and train:
            self._save_agents(self.save_filename)
        return self.get_metrics()
