import torch
import random
import torch.nn as nn
import numpy as np
from dp_util.dqn import Transition


def optimize_model(policy_net, target_net, memory, optimizer,
                   BATCH_SIZE, GAMMA, device):
    """performs a single step of the optimization"""

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) by states and actions
    state_action_values = policy_net(state_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss


def select_action(policy_net, states, eps_threshold):
    """select the action based on the number of steps_done and random value"""

    sample = random.random()

    # pick action with the larger expected reward
    if sample > eps_threshold:
        with torch.no_grad():
            return torch.argmax(policy_net(states).flatten()).item()

    # select random action
    else:
        # increaee the chance of using gas
        int_random_act = np.random.choice(np.arange(0, 5), p=[0.05, 0.15, 0.15, 0.5, 0.15])

        return int_random_act
