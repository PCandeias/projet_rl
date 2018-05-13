import gym
import random
from gym import spaces, logger
import numpy as np

class RandomFishermanCorrelatedEnv(gym.Env):
    def __init__(self):
        # environment specific variables
        self.max_stock = 10000
        self.initial_stock = 10000
        self.cur_stock = self.initial_stock
        self.population = 10000
        self.n_agents = 1
        self.growth_rate = 3
        self.base_pr_fish = 1.0 / self.growth_rate
        self.cur_step = 0
        self.max_steps = 100
        self.action_space = spaces.Discrete(2)
        self.n_groups = 1
        self.observation_space = spaces.Box(0, self.max_stock, shape=(1 + self.n_groups,))

    def set_environment_variables(self, max_stock, initial_stock, population, n_agents, growth_rate, max_steps,
            n_groups, g_consumption = None):
        if(n_agents > population or n_groups <= 0):
            raise ValueError("N. Agents > Population")
        self.max_stock = max_stock
        self.initial_stock = initial_stock
        self.cur_stock = self.initial_stock
        self.population = population
        self.n_agents = n_agents
        self.n_non_agents = population - n_agents
        self.growth_rate = growth_rate
        self.base_pr_fish = 1.0 / self.growth_rate
        self.max_steps = max_steps
        self.n_groups = n_groups
        self.cur_group = 0
        self.observation_space = spaces.Box(0, self.max_stock, shape=(1 + n_groups,))
        if g_consumption is None:
            self.g_consumption = [int(self.n_non_agents / self.n_groups) + (self.n_non_agents % self.n_groups if i == 0 else 0) for i in
                    range(self.n_groups)]
        else:
            self.g_consumption = g_consumption

    def step(self, actions):
        if self.done:
            return np.array(self.cur_stock), np.zeros(self.n_agents), self.done, {0}
        players_want = np.sum(actions)
        total_want = players_want + self.g_consumption[self.cur_step % self.n_groups]
        if self.cur_stock >= total_want:
            self.cur_stock = min((self.cur_stock - total_want) * 2, self.max_stock)
            rewards = np.array(actions)
            consumed = total_want
        else:
            chosen_rewards = np.zeros(total_want)
            chosen_rewards[0:self.cur_stock] = 1
            np.random.shuffle(chosen_rewards)
            consumed = self.cur_stock
            self.cur_stock = 0
            rewards = np.zeros(self.n_agents)
            cur = 0
            for i in range(self.n_agents):
                if actions[i] == 1:
                    rewards[i] = chosen_rewards[cur]
                    cur += 1
        self.cur_group = (self.cur_group + 1) % self.n_groups
        group = np.zeros(self.n_groups)
        group[self.cur_group] = 1
        self.cur_step += 1
        self.done = self.cur_stock == 0 or self.cur_step >= self.max_steps
        return np.append(np.array(self.cur_stock), group), rewards, self.done, {consumed}

    def reset(self):
        self.cur_stock = self.initial_stock
        self.cur_step = 0
        self.done = False
        self.cur_group = 0
        group = np.zeros(self.n_groups)
        group[self.cur_group] = 1
        return np.append(np.array(self.cur_stock), group)

    def render(self):
        print("Current stock: %d" % (self.cur_stock))

    def close(self):
        return None

