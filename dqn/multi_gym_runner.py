import numpy as np
from dqn_solver import DqnSolver
from pg_solver import PgSolver
from ac_solver import AcSolver


class MultiGymRunner(object):
    def __init__(self, n_agents=1, agent_mode='dqn', save_filename = None, load_filename=None, save_frequency = 20000, replay_start_size=10000,
                 gamma=0.99, eps=1.0, eps_decay=0.995, eps_min=0.05, alpha=0.01, memory_size=1000000, batch_size=64,
                 update_frequency=1, double_q=False, freeze_target_frequency=500, verbose=False):
        self.n_agents = n_agents
        self.agent_mode = agent_mode
        self.agents = []
        self.save_filename = save_filename
        self.save_frequency = save_frequency
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency

    def _create_environment(self):
        raise NotImplementedError

    def _save_agents(self, save_filename):
        print('saving...')
        for i in range(self.n_agents):
            self.agents[i].save_model(save_filename + str(i))

    def _create_agents(self, load_filename, gamma, eps, eps_decay, eps_min, alpha,
                       memory_size, batch_size, double_q, freeze_target_frequency, verbose):
        print("Creating agents...")
        self.agents = []
        for i in range(self.n_agents):
            if self.agent_mode == 'pg':
                self.agents.append(PgSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                    gamma=gamma, alpha=alpha, memory_size=memory_size, batch_size=batch_size,
                    verbose=verbose, load_filename=load_filename + str(i) if load_filename else None))
            elif self.agent_mode == 'ac':
                self.agents.append(AcSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                    gamma=gamma, alpha=alpha, memory_size=memory_size, batch_size=batch_size,
                    freeze_target_frequency=freeze_target_frequency, verbose=verbose, load_filename=load_filename + str(i) if load_filename else None))
            else:
                self.agents.append(DqnSolver(action_size=self.get_action_size(), observation_size=self.get_observation_size(),
                                             gamma=gamma, eps=eps, eps_decay=eps_decay, eps_min=eps_min, alpha=alpha,
                                             memory_size=memory_size, batch_size=batch_size, double_q=double_q,
                                             freeze_target_frequency=freeze_target_frequency,
                                             verbose=verbose, load_filename=load_filename + str(i) if load_filename else None))
        print("Done creating agents.")


    def _preprocess_reward(self, reward):
        return reward

    def _preprocess_state(self, state):
        return np.reshape(state, [1, self.get_observation_size()])

    def get_action_size(self):
        return self.env.action_space.n

    def get_observation_size(self):
        return self.env.observation_space.n

    def _reset_metrics(self, display_frequency, r_episodes=False):
        self.avg_scores = []
        if r_episodes:
            self.scores_episodes = []
            self.best = -10000000

    def _display_metrics(self, ep_number):
        print("Episode: %d Average score: %f" % (ep_number, np.mean(self.scores_episodes[-100:-1])))

    def _update_metrics(self, step, state, actions, rewards, next_state, done, score, ep_number):
        self.avg_scores.append(np.mean(rewards))
        if done:
            ep_score = np.sum(self.avg_scores)
            self.avg_scores = [] # reset episode scores
            if ep_score > self.best:
                self.best = ep_score
                self.best_ep = ep_number
            self.scores_episodes.append(ep_score)

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

    def _select_actions(self, state, eps):
        actions = []
        for agent in self.agents:
            actions.append(agent.select_action(state, eps))
        return actions

    def _stop_condition(self, episode_number):
        return False

    def _save_condition(self, episode_number):
        return False
    """
    Do any post processing on the action array required by the environment
    """
    def _process_actions(self, actions):
        return actions

    def run(self, n_episodes, train=False, render=False, verbose=False, display_frequency=1000, eps=None):
        total_steps = 0
        self._reset_metrics(r_episodes=True, display_frequency=display_frequency)
        r_eps = None if train else eps
        for e in range(n_episodes):
            step = 0
            score = np.zeros(self.n_agents, dtype=np.float64)
            done = False
            state = self._preprocess_state(self.env.reset())
            while not done:
                if render:
                    self.env.render()
                actions = self._select_actions(state, r_eps)
                next_state, rewards, done, info = self.env.step(self._process_actions(actions))
                next_state = self._preprocess_state(next_state)
                rewards = self._preprocess_reward(rewards)
                step += 1
                total_steps += 1
                score += rewards
                self._update_metrics(step, state, actions, rewards, next_state, done, score, e)
                if train:
                    self._store_transitions(state[0], actions, rewards, next_state[0], done)
                state = next_state
                if train and total_steps % self.update_frequency == 0 and total_steps >= self.replay_start_size:
                    self._train_agents()
            if verbose and (e+1)%display_frequency == 0:
                self._display_metrics(ep_number=e)
            if train and self._stop_condition(e):
                break
            if self.save_filename is not None and train:
                if (e+1) % self.save_frequency == 0:
                    self._save_agents(self.save_filename)
                    if self._save_condition(e):
                        self._save_agents(self.save_filename + "_best")

        if self.save_filename is not None and train:
            self._save_agents(self.save_filename)
        return self.get_metrics()
