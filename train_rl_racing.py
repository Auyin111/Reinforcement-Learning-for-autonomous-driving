import matplotlib.pyplot as plt
import wandb
import pickle
import torch
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from PIL import Image
from collections import namedtuple
from itertools import count

import improved_gym

from models.on_track_cls import OnTrackClsNet
from models.rl_car_racing import DDQN
from dp_spec.img import preprocess_img
from dp_util.img import output_to_arr
from dp_util.dqn import ReplayMemory, find_eps_thres
from dp_spec.dqn import optimize_model, select_action
from cfgs.rl_car_racing import RlCarRacingCfg


def check_offtrack(on_track_cls_net, off_track_label, next_state, reward):
    """if off track penalise"""

    with torch.no_grad():
        outputs = on_track_cls_net(next_state.unsqueeze(0))
    on_off_track = output_to_arr(outputs)[0]
    if on_off_track == off_track_label:
        reward = reward - 0.5

    return reward


# load_prev_md = True: load previous model and further train it
# load_prev_md = False: train model from scratch
load_prev_md = False
collect_img = False
cls_version = 'v_2_0_7'

use_double_dqn = True
step_waiting = 50
img_num = 1
dir_auto_img = r'D:\file\data\racing\auto_img'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


if __name__ == "__main__":

    on_track_cls_file = f'on_track_classifier_{cls_version}.pth'
    cfg = RlCarRacingCfg(test_mode=False, dir_data='./')

    wandb.init(project=cfg.project, entity=cfg.entity,
               group=cfg.model,
               job_type="train_model",
               name=cfg.version)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    # load the on track classifier model and encoder
    ###########################################################
    with open(f'ont_hot_encoder_{cls_version}.pkl', 'rb') as f:
        encoder = pickle.load(f)
    # extract definition of prediction output
    on_track_label = encoder.categories_[0].tolist().index('on')
    off_track_label = encoder.categories_[0].tolist().index('off')
    on_track_cls_net = OnTrackClsNet().to(device)
    on_track_cls_net.load_state_dict(torch.load(on_track_cls_file))

    # preview the eps threshold
    ###########################################################
    list_steps_done = np.linspace(1, 200000, num=40)
    list_eps_threshold = []

    for x in list_steps_done:
        list_eps_threshold.append(find_eps_thres(cfg, x))

    plt.style.use('dark_background')
    plt.figure(figsize=(20,10))
    plt.plot(list_steps_done, list_eps_threshold, c ="white", linestyle='dashed', marker='o')
    plt.title("eps againt steps_done", fontsize=30)
    # plt.show()

    # prepare policy and target net
    ###########################################################
    n_actions = 5
    state_dim = (2, 84, 84)
    input_size = state_dim[1]
    policy_net = DDQN(state_dim[0], input_size, n_actions).to(device)
    target_net = DDQN(state_dim[0], input_size, n_actions).to(device)

    if load_prev_md:

        i_episode = 663
        print(f'load the previous model {cfg.md_name.split(".")[0]}')

        policy_net.load_state_dict(torch.load(f'poli_{i_episode}_{cfg.md_name}'))
        target_net.load_state_dict(torch.load(f'tg_{i_episode}_{cfg.md_name}'))

        with open(f"dict_cp_{i_episode}_{cfg.md_name.split('.')[0]}.pkl", 'rb') as f:
            dict_cp = pickle.load(f)

        memory = dict_cp['memory']
        list_reward_round_sum = dict_cp['list_reward_round_sum']
        list_round_avg_loss = dict_cp['list_round_avg_loss']
        steps_done = dict_cp['steps_done']
        i_episode = dict_cp['i_episode']
        eps_threshold = dict_cp['eps_threshold']

    # start from a new records
    else:
        print('start to train a new model')
        target_net.load_state_dict(policy_net.state_dict())

        memory = ReplayMemory(cfg.MEMORY_CAP)
        list_reward_round_sum = []
        list_round_avg_loss = []
        steps_done = 0
        i_episode = 1
        eps_threshold = cfg.EPS_START

    # training
    ###########################################################
    optimizer = optim.AdamW(policy_net.parameters(), lr=cfg.LR, amsgrad=True)
    # Log the network weight histograms
    wandb.watch(policy_net)
    env = improved_gym.make("CarRacing-v2",
                            # render_mode='human',
                            continuous=False, )

    while i_episode <= cfg.NUM_EPISODES:

        steps_done_round = 0
        list_reward_round = []
        list_loss_round = []
        list_speed = []
        print(f'i_episode: {i_episode}, eps_threshold: {eps_threshold:.{cfg.SF}f}')

        state, info = env.reset()
        prev_state = None

        # as the image is zooming in the first 50 steps and it wil affect learning, do not interact with env
        for i in range(step_waiting):
            _, _, _, _, _ = env.step(0)
            steps_done_round += 1

        state = preprocess_img(state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():

            steps_done += 1
            steps_done_round += 1
            eps_threshold = find_eps_thres(cfg, steps_done)

            if prev_state is None:
                prev_state = state
            states = torch.concat((prev_state, state)).unsqueeze(0)
            action = select_action(policy_net, states, eps_threshold, device)
            observation, reward, terminated, truncated, _ = env.step(action.item())

            # save the img to train a on track classifier model
            if collect_img:
                img_num += 1
                if img_num % 10 == 0:
                    img = Image.fromarray(observation, 'RGB')
                    img.save(os.path.join(dir_auto_img, f'v4_{img_num}.png'))

            list_reward_round.append(reward)
            list_speed.append(env.true_speed)

            reward = torch.tensor([[reward]], device=device)
            observation = preprocess_img(observation)
            next_state = torch.tensor(observation,
                                      dtype=torch.float32, device=device).unsqueeze(0)
            reward = check_offtrack(on_track_cls_net, off_track_label, next_state / 255, reward)

            done = terminated or truncated
            ts_done = torch.tensor([[done]], device=device, dtype=torch.long)

            # Store the transition in memory
            next_states = torch.concat((state, next_state)).unsqueeze(0)
            memory.push(states, action, next_states, reward, ts_done)
            prev_state = state
            state = next_state

            if len(memory) >= 1000:
                if len(memory) == 1000:
                    print('start to train model')
                # Perform one step of the optimization (on the policy network)
                loss = optimize_model(use_double_dqn, policy_net, target_net, memory, optimizer,
                                      cfg.BATCH_SIZE, cfg.GAMMA, device)
                list_loss_round.append(loss)

            if steps_done % 1 == 0:
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * cfg.TAU \
                                                 + target_net_state_dict[key] * (1 - cfg.TAU)
                target_net.load_state_dict(target_net_state_dict)

            if done:
                sum_reward_round = np.sum(list_reward_round)
                list_reward_round_sum.append(sum_reward_round)
                avg_loss_round = np.mean([x.cpu().detach().numpy() for x in list_loss_round])
                list_round_avg_loss.append(avg_loss_round)
                avg_speed_round = np.mean(list_speed)

                wandb.log({'i_episode': i_episode,
                           "reward_round": sum_reward_round, 'avg_loss_round': avg_loss_round,
                           'avg_speed': avg_speed_round})

                i_episode += 1

                break

    torch.save(policy_net.state_dict(), f'poli_{i_episode}_{cfg.md_name}')
    torch.save(target_net.state_dict(), f'tg_{i_episode}_{cfg.md_name}')

    dict_cp = {
        'memory': memory,
        'list_reward_round_sum': list_reward_round_sum,
        'list_round_avg_loss': list_round_avg_loss,
        'steps_done': steps_done,
        'i_episode': i_episode,
        'eps_threshold': eps_threshold
    }

    with open(f"dict_cp_{i_episode}_{cfg.md_name.split('.')[0]}.pkl", 'wb') as output:
        pickle.dump(dict_cp, output)

    df_reward_round_sum = pd.DataFrame(
        {'reward_round_sum': list_reward_round_sum,
         'round_avg_loss': list_round_avg_loss
         })

    ax = df_reward_round_sum.reward_round_sum.plot(legend=True)
    df_reward_round_sum.round_avg_loss.plot(ax=ax,
                                            color='r',secondary_y=True, legend=True)
    plt.title("reward and loss against episode ", fontsize=20)
    # plt.show()
