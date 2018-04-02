import gym
import random
from gym import spaces, logger
import numpy as np

class RandomFishermanEnv(gym.Env):
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
        self.observation_space = spaces.Box(0, self.max_stock, shape=(1,))

    def set_environment_variables(self, max_stock, initial_stock, population, n_agents, growth_rate, max_steps):
        if(n_agents > population):
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

    def step(self, actions):
        if self.done:
            return np.array(self.cur_stock), np.zeros(self.n_agents), self.done, {0}
        self.cur_step += 1
        pr_fish = min(1,max(0, (self.cur_stock -
            min(self.max_stock * self.base_pr_fish, self.population / float(self.growth_rate -1))) / self.population))
        players_want = np.sum(actions)
        pop_want = np.random.binomial(self.n_non_agents, pr_fish)
        total_want = players_want + pop_want
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
        self.done = self.cur_stock == 0 or self.cur_step >= self.max_steps
        return np.array(self.cur_stock), rewards, self.done, {consumed}

    def reset(self):
        self.cur_stock = self.initial_stock
        self.cur_step = 0
        self.done = False
        return np.array(self.cur_stock)

    def render(self):
        print("Current stock: %d" % (self.cur_stock))

    def close(self):
        return None

