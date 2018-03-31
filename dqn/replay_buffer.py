import random


class ReplayBuffer(object):
    def __init__(self, max_len=None):
        self.max_len = max_len
        self.memory = []
        self.cur = 0


    def __len__(self):
        return len(self.memory)

    def sample(self, n_samples):
        return random.sample(self.memory, n_samples)

    def append(self, element):
        if self.max_len is None or self.max_len >= len(self.memory):
            self.memory.append(element)
        else:
            self.memory[self.cur] = element
            self.cur = (self.cur + 1) % self.max_len