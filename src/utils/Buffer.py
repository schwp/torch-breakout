from collections import deque
import random

import numpy as np


class Buffer():
    def __init__(self, capicity):
        self.buffer = deque(maxlen=capicity)

    def add(self, x):
        self.buffer.append(x)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)
