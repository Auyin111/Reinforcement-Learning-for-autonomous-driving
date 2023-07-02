import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hid_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hid_size)
        hid_size_half = hid_size // 2
        self.layer2 = nn.Linear(hid_size, hid_size_half)
        self.layer3 = nn.Linear(hid_size_half, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


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


def select_action(policy_net, env, state, steps_done,
                  eps_end, eps_start, eps_decay,
                  device):
    """select the action based on the number of steps_done and random value"""

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) \
                    * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1

    # pick action with the larger expected reward
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), steps_done
    # select random action
    else:
        return torch.tensor([[env.action_space.sample()]],
                            device=device, dtype=torch.long), steps_done
