from models import non_spiking
from models import SSA
from models import TTSA
from utils.lidar2image import LidarToOccupancyGrid
import highway_env

import argparse
import os.path
import os
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from collections import deque
import numpy as np
import random
import torch
import gymnasium as gym
import einops
import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch

msgpack_numpy_patch()

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# ARG PARSER FOR MULTIPLE SEEDS
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=1,
                    help="Number of seeds to run the experiment")
parser.add_argument("--scenario", type=str, default="highway-v0",
                    help="Scenario: highway-v0 or roundabout-v0 (default)")
parser.add_argument("--mode", type=str, default="non-spiking",
                    help="Mode: non-spiking (default)")
args = parser.parse_args()


# -----------------------------
# HELPER: SET GLOBAL SEEDS
# -----------------------------
def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# =============================
# RUN EXPERIMENT FOR EACH SEED
# =============================

for run_index in range(args.seeds):

    seed = run_index * 10
    print(f"\n===== RUN {run_index + 1}/{args.seeds} — Using Seed {seed} =====\n")
    set_global_seed(seed)

    # ---- Training Hyperparameters ----
    GAMMA = 0.99
    BATCH_SIZE = 32
    BUFFER_SIZE = int(5e4)
    MIN_REPLAY_SIZE = 10000

    EPSILON_START = 1.0
    EPSILON_END = 0.1
    NUM_ENV = 1
    EPSILON_DECAY = int(7e4)
    TARGET_UPDATE_FREQ = 100 // NUM_ENV
    LR = 1e-4
    SAVE_INTERVAL = 5000
    LOG_INTERVAL = 1000



    # ========== ENV CONFIGS ==========

    if args.scenario == "highway-v0":

        if args.mode == "non-spiking":
            LOG_DIR = f'logs/non_spiking_Highway/_seed_{seed}'
        elif args.mode == "SSA":
            LOG_DIR = f'logs/SSA_Highway/_seed_{seed}'
        elif args.mode == "TTSA":
            LOG_DIR = f'logs/TTSA_Highway/_seed_{seed}'
        config1 = {
            "observation": {
                "type": "LidarObservation",
                "features": ["presence", "distance"],
                "cells": 800,
                "maximum_range": 60
            },
            "action": {"type": "DiscreteMetaAction"},
            "vehicles_density": 1,
            "duration": 50,
        }

        config = {
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 128),
                "stack_size": 1,
                "weights": [0.299, 0.587, 0.114],
                "scaling": 2,
                "centering_position": [.5, .5]
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 50,
        }

        config2 = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ["heading"],
                "absolute": False,
                "order": "sorted"
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 50,
        }

        lidar_converter = LidarToOccupancyGrid(output_range=[50, 200], v_max=30)

        # ====== ENVIRONMENTS ======
        env = gym.make("highway-v0", render_mode="rgb_array", config=config)
        env2 = gym.make("highway-v0", render_mode="rgb_array", config=config1)
        env3 = gym.make("highway-v0", render_mode="rgb_array", config=config2)

        tenv = gym.make("highway-v0", render_mode="rgb_array", config=config)
        tenv2 = gym.make("highway-v0", render_mode="rgb_array", config=config1)
        tenv3 = gym.make("highway-v0", render_mode="rgb_array", config=config2)

    elif args.scenario == "roundabout-v0":

        if args.mode == "non-spiking":
            LOG_DIR = f'logs/non_spiking_roundabout/_seed_{seed}'
        elif args.mode == "SSA":
            LOG_DIR = f'logs/SSA_roundabout/_seed_{seed}'
        elif args.mode == "TTSA":
            LOG_DIR = f'logs/TTSA_roundabout/_seed_{seed}'


        config1 = {
            "observation": {
                "type": "LidarObservation",
                "features": ["presence", "distance"],
                "cells": 800,
                "maximum_range": 60
            },
            "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 8, 16]},
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,
            "normalize_reward": True,
        }

        config = {
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 128),
                "stack_size": 1,
                "weights": [0.299, 0.587, 0.114],  # weights for RGB conversion
                "scaling": 2,
                "centering_position": [.5, .5]
            },
            "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 8, 16]},
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,
            "normalize_reward": True,
        }

        config2 = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ["heading"],
                "absolute": False,
                "order": "sorted"
            }, "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 8, 16]},
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,
            "normalize_reward": True,
        }

        lidar_converter = LidarToOccupancyGrid(output_range=[50, 200], v_max=16)

        env = gym.make("roundabout-v0", render_mode="rgb_array", config=config)
        env2 = gym.make("roundabout-v0", render_mode="rgb_array", config=config1)
        env3 = gym.make("roundabout-v0", render_mode="rgb_array", config=config2)

        tenv = gym.make("roundabout-v0", render_mode="rgb_array", config=config)
        tenv2 = gym.make("roundabout-v0", render_mode="rgb_array", config=config1)
        tenv3 = gym.make("roundabout-v0", render_mode="rgb_array", config=config2)

    # ======= REPLAY BUFFER =======
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epinfos_buffer = deque([], maxlen=100)
    rews_buffer_ = []

    summary_writer = SummaryWriter(LOG_DIR)

    # ======================================
    # ======= NETWORKS (non-spiking) =======
    # ======================================
    if args.mode == "non-spiking":

        os.makedirs(args.scenario + '/' + '-' + args.mode + '/'+ '-' + str(seed), exist_ok=True)


        online_net = non_spiking.Network(env, env2, device=device,
                             depths1=(8, 16, 16),
                             depths2=(8, 16, 16),
                             final_layer=512, scenario=args.scenario, mode=args.mode, seed=seed)

        target_net = non_spiking.Network(env, env2, device=device,
                             depths1=(8, 16, 16),
                             depths2=(8, 16, 16),
                             final_layer=512, scenario=args.scenario, mode=args.mode, seed=seed)
    elif args.mode == "SSA":

        os.makedirs(args.scenario + '/' + '-' + args.mode + '/'+ '-' + str(seed), exist_ok=True)


        online_net = SSA.Network(env, env2, device=device,
                             depths1=(8, 16, 16),
                             depths2=(8, 16, 16),
                             final_layer=512, num_steps=5, scenario=args.scenario, mode=args.mode, seed=seed)

        target_net = SSA.Network(env, env2, device=device,
                             depths1=(8, 16, 16),
                             depths2=(8, 16, 16),
                             final_layer=512, num_steps=5, scenario=args.scenario, mode=args.mode, seed=seed)

    elif args.mode == "TTSA":

        os.makedirs(args.scenario + '/' + '-' + args.mode + '/'+ '-' + str(seed), exist_ok=True)


        online_net = TTSA.Network(env, env2, device=device,
                             depths1=(8, 16, 16),
                             depths2=(8, 16, 16),
                             final_layer=512, num_steps=5, scenario=args.scenario, mode=args.mode, seed=seed)

        target_net = TTSA.Network(env, env2, device=device,
                             depths1=(8, 16, 16),
                             depths2=(8, 16, 16),
                             final_layer=512, num_steps=5, scenario=args.scenario, mode=args.mode, seed=seed)

    online_net = online_net.to(device)
    target_net = target_net.to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    # ======= INITIAL RESET =======
    h_seed = np.random.randint(0, 2 ** 63)
    obs1, infos = env.reset(seed=h_seed)
    obs2, _ = env2.reset(seed=h_seed)
    obs3, _ = env3.reset(seed=h_seed)

    num_rays = config1["observation"]["cells"]
    angle_range = [0, -2 * np.pi]
    angles = np.linspace(angle_range[0], angle_range[1], num_rays)

    heading = obs3[0][0]
    v_vel = infos['speed']
    occupancy_grid = lidar_converter.process(obs2, angles, v_vel, heading)
    obsp = occupancy_grid

    obs = [obs1, obsp]

    episode_count = 0
    t = 0

    # =====================================================
    # ==================== TRAINING LOOP ==================
    # =====================================================

    for step in range(int(1e5)):

        epsilon = np.interp(step * NUM_ENV,
                            [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])

        action = online_net.act(obs, epsilon)

        new_obs1, rew, done, termin, infos = env.step(action)
        new_obs2, _, _, _, _ = env2.step(action)
        new_obs3, _, _, _, _ = env3.step(action)

        heading = new_obs3[0][0]
        v_vel = infos['speed']
        occupancy_grid = lidar_converter.process(new_obs2, angles, v_vel, heading)
        obsp = occupancy_grid

        new_obs = [new_obs1, obsp]

        rews_buffer_.append(rew)
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)

        obs = new_obs
        t += 1

        # -------- EPISODE END --------
        if done or termin:
            eprew = sum(rews_buffer_)
            eplen = len(rews_buffer_)
            epinfos_buffer.append({"r": round(eprew, 6), "l": eplen})

            rews_buffer_ = []
            episode_count += 1

            h_seed = np.random.randint(0, 2 ** 63)
            obs1, infos = env.reset(seed=h_seed)
            obs2, _ = env2.reset(seed=h_seed)
            obs3, _ = env3.reset(seed=h_seed)

            v_vel = infos['speed']
            heading = obs3[0][0]
            occupancy_grid = lidar_converter.process(obs2, angles, v_vel, heading)
            obsp = occupancy_grid

            obs = [obs1, obsp]
            t = 0

        # -------- TRAIN NETWORK --------
        if len(replay_buffer) >= 1000:
            transitions = random.sample(replay_buffer, BATCH_SIZE)
            loss = online_net.compute_loss(transitions, target_net)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------- TARGET NET UPDATE --------
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # -------- LOGGING --------
        if step % LOG_INTERVAL == 0:
            rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
            len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0

            print()
            print('Seed:', seed)
            print('Step:', step)
            print('Avg Rew:', rew_mean)
            print('Avg Ep len:', len_mean)
            print('Episodes:', episode_count)

            summary_writer.add_scalar('AvgRew', rew_mean, step)
            summary_writer.add_scalar('AvgLen', len_mean, step)

        # -------- SAVE & TEST --------
        if step % SAVE_INTERVAL == 0 or step == 0:

            test_reward_sum = 0

            for _ in range(20):
                tseed = np.random.randint(0, 2 ** 63)
                o1, tinfo = tenv.reset(seed=tseed)
                o2, _ = tenv2.reset(seed=tseed)
                o3, _ = tenv3.reset(seed=tseed)

                headingt = o3[0][0]
                v_velt = tinfo['speed']
                occupancy_grid = lidar_converter.process(o2, angles, v_velt, headingt)
                opt = occupancy_grid

                obst = [o1, opt]
                d = False
                ep_ret = 0
                ep_len = 0
                ttermin = 0

                while not (d or ttermin):
                    a = online_net.act(obst, 0)
                    tnew_obs1, trew, d, ttermin, tinfos = tenv.step(a)
                    tnew_obs2, _, _, _, _ = tenv2.step(a)
                    tnew_obs3, _, _, _, _ = tenv3.step(a)

                    headingn = tnew_obs3[0][0]
                    v_veln = tinfos['speed']
                    occupancy_grid = lidar_converter.process(tnew_obs2, angles, v_veln, headingn)
                    tobsp = occupancy_grid

                    obst = [tnew_obs1, tobsp]

                    ep_ret += trew
                    ep_len += 1

                test_reward_sum += ep_ret

            test_reward_sum /= 20
            summary_writer.add_scalar('TestRew', test_reward_sum, step)
            print('TestRew:', test_reward_sum)
            print('Saving model...')
            online_net.save(step)
