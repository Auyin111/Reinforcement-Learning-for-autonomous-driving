import random
import math
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def find_eps_thres(cfg, steps_done):
    return cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) \
           * math.exp(-1. * steps_done / cfg.EPS_DECAY)
