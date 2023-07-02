import gymnasium as gym
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count

from model import DQN, ReplayMemory, select_action, Transition
from cfg import Cfg


def optimize_model(memory, batch_size, gamma):
    """performs a single step of the optimization"""

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) by states and actions
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    cfg = Cfg(test_mode=False, dir_data='./')
    md_name = f'{cfg.version}_test' if cfg.test_mode else cfg.version
    wandb.init(project=cfg.project, entity=cfg.entity,
               group=f'{cfg.user}_{cfg.model}', job_type="train",
               name= md_name)

    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions, cfg.HID_SIZE).to(device)
    target_net = DQN(n_observations, n_actions, cfg.HID_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=cfg.LR, amsgrad=True)
    memory = ReplayMemory(cfg.MEMORY_CAP)

    steps_done = 0
    episode_durations = []

    for i_episode in range(1, cfg.NUM_EPISODES + 1):

        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action, steps_done = select_action(policy_net, env, state, steps_done,
                                               cfg.EPS_END, cfg.EPS_START, cfg.EPS_END, device)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation,
                                          dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            state = next_state

            if len(memory) >= cfg.BATCH_SIZE:
                # Perform one step of the optimization (on the policy network)
                optimize_model(memory, cfg.BATCH_SIZE, cfg.GAMMA)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * cfg.TAU \
                                             + target_net_state_dict[key] * (1 - cfg.TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                wandb.log({"episode_durations": t + 1})

                if i_episode % cfg.EPISODE_PRINT == 0:
                    print(f'i_episode: {i_episode}, duration: {t + 1}')
                break

    torch.save(policy_net.state_dict(), f'{md_name}.pth')
