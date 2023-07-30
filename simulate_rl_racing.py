import pickle
import torch
import torch.optim as optim
import improved_gym
from collections import namedtuple
from itertools import count
from models.on_track_cls import OnTrackClsNet
from models.rl_car_racing import DDQN
from dp_spec.img import preprocess_img
from dp_spec.dqn import select_action
from cfgs.rl_car_racing import RlCarRacingCfg


load_prev_md = True
# number of simulation episodes
num_episodes = 10
# select specific trained model to simulate
i_episode = 663
cls_version = 'v_2_0_7'

collect_img = False
step_waiting = 50


if __name__ == "__main__":

    on_track_cls_file = f'on_track_classifier_{cls_version}.pth'
    dir_auto_img = r'D:\file\data\racing\auto_img'
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward', 'done'))

    cfg = RlCarRacingCfg(test_mode=False, dir_data='./')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    env = improved_gym.make('CarRacing-v2', render_mode='human', continuous=False)
    n_actions = env.action_space.n
    state_dim = (2, 84, 84)
    input_size = state_dim[1]

    # load the trained rl model
    policy_net = DDQN(state_dim[0], input_size, n_actions).to(device)
    policy_net.load_state_dict(torch.load(f'poli_{i_episode}_{cfg.md_name}'))

    # simulate
    print(f'start to simulate {num_episodes} times')
    for num_episode in range(1, num_episodes + 1):

        steps_done_round = 0
        state, info = env.reset()
        prev_state = None

        # as the image is zooming in the first 50 steps and it wil affect learning, do not interact with env
        for i in range(step_waiting):
            state, _, _, _, _ = env.step(0)
            steps_done_round += 1

        state = preprocess_img(state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():

            steps_done_round += 1
            # no need any random action
            eps_threshold = 0

            if prev_state is None:
                prev_state = state
            states = torch.concat((prev_state, state)).unsqueeze(0)
            action = select_action(policy_net, states, eps_threshold, device)
            observation, reward, terminated, truncated, _ = env.step(action.item())

            observation = preprocess_img(observation)
            next_state = torch.tensor(observation,
                                      dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated

            # Store the transition in memory
            next_states = torch.concat((state, next_state)).unsqueeze(0)

            prev_state = state
            state = next_state

            if done:
                break

    env.close()
