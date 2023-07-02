import gymnasium as gym
import time
import torch

from model import DQN, select_action
from cfg import Cfg


def sim_game(policy_net, cfg, name_gym, num_episodes, max_steps):
    """simulate the game"""

    env2 = gym.make(name_gym, render_mode='human')

    steps_done = 0

    for episode in range(1, num_episodes + 1):

        (state, _) = env2.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        appendedObservations = []
        for step in range(1, max_steps + 1):
            if step % 20 == 0:
                print(f'step: {step}')

            action, steps_done = select_action(policy_net, env, state, steps_done,
                                               cfg.EPS_END, cfg.EPS_START, cfg.EPS_END, device)

            observation, reward, terminated, truncated, info = env2.step(action.item())
            appendedObservations.append(observation)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Move to the next state
            state = next_state

            time.sleep(0.1)
            if terminated:
                time.sleep(1)
                break
    print('End !')
    env2.close()


if __name__ == '__main__':
    cfg = Cfg(test_mode=False, dir_data='./')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    md_name = f'{cfg.version}_test' if cfg.test_mode else cfg.version
    name_game = "CartPole-v1"

    env = gym.make(name_game)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    model = DQN(n_observations, n_actions, cfg.HID_SIZE).to(device)
    model.load_state_dict(torch.load(f'{md_name}.pth'))

    sim_game(model, cfg, name_game, num_episodes=3, max_steps=500)