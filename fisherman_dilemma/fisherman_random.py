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
        self.growth_rate = growth_rate 
        self.base_pr_fish = 1.0 / self.growth_rate
        self.max_steps = max_steps 

    def step(self, actions):
        if self.done:
            return np.array(self.cur_stock), rewards, self.done, {consumed}
        self.cur_step += 1
        actions_population = np.random.rand(self.population)
        agents_spots = random.sample(range(self.population), self.n_agents)
        rewards = np.zeros(self.n_agents, dtype=np.float64)
        pr_fish = max(0, (self.cur_stock - 
            min(self.max_stock * self.base_pr_fish, self.population / float(self.growth_rate -1))) / self.population)
        starting_stock = self.cur_stock
        for i in range(self.population):
            if self.cur_stock <= 0:
                self.done = True
                break
            if i in agents_spots:
                player_i = agents_spots.index(i)
                if actions[player_i] == 1:
                    rewards[player_i] = 1
                    self.cur_stock -= 1
            else:
                if np.random.rand() < pr_fish:
                    self.cur_stock -= 1
        consumed = starting_stock-self.cur_stock
        self.cur_stock = min(self.max_stock, self.cur_stock * self.growth_rate)
        if self.cur_step >= self.max_steps:
            self.done = True
        print("Step: %d Before: %d After: %d Consumed: %d" % (self.cur_step, starting_stock, self.cur_stock, consumed))
        return np.array(self.cur_stock), rewards, self.done, {consumed}

    def reset(self):
        self.cur_stock = self.initial_stock
        self.cur_step = 0
        self.done = False
        return np.array(self.cur_stock, dtype=np.float64)

    def render(self):
        print("Current stock: %d" % (self.cur_stock))

    def close(self):
        return None

