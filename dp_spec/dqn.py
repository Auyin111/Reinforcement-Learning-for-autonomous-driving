import torch
import random
import torch.nn as nn
import numpy as np
from dp_util.dqn import Transition


def select_greedy_actions(states, q_network):
    """select the greedy action base on Q-values"""

    with torch.no_grad():
        _, actions = q_network(states).max(dim=1, keepdim=True)

    return actions


def eval_actions(states, actions, rewards, dones, gamma, q_network):
    """computes the Q-values by evaluating the actions given the current states and Q-netwoek"""

    with torch.no_grad():
        next_q_values = q_network(states).gather(1, actions)
    q_values = rewards + (gamma * next_q_values * (1 - dones))

    return q_values


def q_learning_update(states, rewards, dones, gamma, q_network):
    actions = select_greedy_actions(states, q_network)
    q_values = eval_actions(states, actions, rewards, dones, gamma, q_network)

    return q_values


def double_q_learning_update(states, rewards, dones, gamma, q_network_1, q_network_2):
    """use q_network_1 to select actions and q_network_2 to evaulate"""

    actions = select_greedy_actions(states, q_network_1)
    q_values = eval_actions(states, actions, rewards, dones, gamma, q_network_2)

    return q_values


def optimize_model(use_double_dqn, policy_net, target_net, memory, optimizer,
                   BATCH_SIZE, GAMMA, device):
    """performs a single step of the optimization"""

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    next_state_batch = torch.cat(batch.next_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    done_batch = torch.cat(batch.done)

    # Compute Q(s_t, a) by states and actions
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    if use_double_dqn:
        target_q_values = double_q_learning_update(next_state_batch,
                                                   reward_batch, done_batch, GAMMA,
                                                   target_net, policy_net)
    else:
        target_q_values = q_learning_update(next_state_batch,
                                            reward_batch, done_batch, GAMMA,
                                            target_net)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss


def select_action(policy_net, states, eps_threshold, device):
    """select the action based on the number of steps_done and random value"""

    sample = random.random()

    # pick action with the larger expected reward
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(states).max(1)[1].view(1, 1)

    # select random action
    else:
        # increase the chance of using gas
        int_random_act = np.random.choice(np.arange(0, 5), p=[0.05, 0.15, 0.15, 0.5, 0.15])

        return torch.tensor([[int_random_act]], device=device, dtype=torch.long)

