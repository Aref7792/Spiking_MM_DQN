import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gym
import gym_carla
import carla
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()


# =========================================================
# Args
# =========================================================
@dataclass
class Args:
    exp_name: str = "carla_birdeye_stack_dqn"
    seed: int = 20
    torch_deterministic: bool = True
    cuda: bool = True

    total_timesteps: int = int(1e5)
    learning_rate: float = 1e-4
    gamma: float = 0.99

    batch_size: int = 32
    buffer_size: int = int(2e5)
    learning_starts: int = 10000
    train_frequency: int = 4
    target_update_freq: int = 200

    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = int(1e5)

    log_interval: int = 1000
    save_interval: int = 5000

    save_dir: str = "models_birdeye_DQN"
    log_dir: str = "runs"

    final_layer: int = 512

    number_of_vehicles: int = 100
    number_of_walkers: int = 0
    display_size: int = 256
    max_past_step: int = 1
    dt: float = 0.1
    port: int = 2000
    town: str = "Town03"
    task_mode: str = "random"
    max_time_episode: int = 500
    max_waypt: int = 12
    obs_range: int = 32
    lidar_bin: float = 0.5
    d_behind: int = 12
    out_lane_thres: float = 2.0
    desired_speed: int = 8
    max_ego_spawn_times: int = 200
    pixor_size: int = 64
    pixor: bool = False
    use_radar: bool = True
    radar_height: float = 1.0
    radar_x: float = 2.0
    radar_hfov: int = 60
    radar_vfov: int = 20
    radar_range: int = 32
    radar_pps: int = 3000
    radar_vmax: float = 30.0
    render_panels: int = 3
    render: bool = False
    enable_pygame: bool = False
    frame_stack: int = 4


# =========================================================
# Utils
# =========================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def reset_env(env, seed=None):
    if seed is None:
        out = env.reset()
    else:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            out = env.reset()

    if isinstance(out, tuple):
        obs, _info = out
        return obs
    return out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 4:
        obs, rew, done, info = out
        return obs, rew, done, info
    elif len(out) == 5:
        obs, rew, terminated, truncated, info = out
        return obs, rew, (terminated or truncated), info
    raise ValueError("Unexpected env.step output format")


def obs_to_tensor(obs, device):
    birdeye = torch.as_tensor(obs["birdeye"], dtype=torch.float32, device=device)
    return birdeye


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# =========================================================
# Replay Buffer
# =========================================================
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, obs_dtype=np.uint8):
        self.capacity = int(capacity)
        self.ptr = 0
        self.full = False

        self.obs = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)
        self.nobs = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)

        self.act = np.empty((self.capacity,), dtype=np.int64)
        self.rew = np.empty((self.capacity,), dtype=np.float32)
        self.done = np.empty((self.capacity,), dtype=np.float32)
        self.k = np.empty((self.capacity,), dtype=np.int64)

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def add(self, obs, action, reward, done, next_obs, k):
        i = self.ptr
        self.obs[i] = obs
        self.nobs[i] = next_obs
        self.act[i] = int(action)
        self.rew[i] = float(reward)
        self.done[i] = float(done)
        self.k[i] = int(k)

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.done[idx],
            self.nobs[idx],
            self.k[idx],
        )


# =========================================================
# Model
# =========================================================
class QNetwork(nn.Module):
    def __init__(self, action_dim, birdeye_shape, final_layer=512, depths=(16, 32, 64)):
        super().__init__()
        self.action_dim = action_dim

        n_input_channels = birdeye_shape[-1]
        h, w = birdeye_shape[0], birdeye_shape[1]
        dim_h, dim_w = self._conv_output_hw(h, w)
        conv_out_dim = depths[2] * dim_h * dim_w

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, depths[0], kernel_size=5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(depths[0], depths[1], kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.q_head = nn.Sequential(
            layer_init(nn.Linear(conv_out_dim, final_layer)),
            nn.ReLU(),
            layer_init(nn.Linear(final_layer, action_dim), std=0.01),
        )

    @staticmethod
    def _conv2d_out(size, kernel, stride, padding=0, dilation=1):
        return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    def _conv_output_hw(self, h, w):
        h = self._conv2d_out(h, 5, 2)
        w = self._conv2d_out(w, 5, 2)

        h = self._conv2d_out(h, 3, 2)
        w = self._conv2d_out(w, 3, 2)

        h = self._conv2d_out(h, 3, 1)
        w = self._conv2d_out(w, 3, 1)
        return h, w

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float() / 255.0
        z = self.encoder(x)
        q = self.q_head(z)
        return q

    def act(self, x, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            q_values = self.forward(x)
            return int(torch.argmax(q_values, dim=1).item())

    def save(self, path):
        torch.save(self.state_dict(), path)


# =========================================================
# Environment factory
# =========================================================
def make_env(args):
    params = {
        "number_of_vehicles": args.number_of_vehicles,
        "number_of_walkers": args.number_of_walkers,
        "display_size": args.display_size,
        "max_past_step": args.max_past_step,
        "dt": args.dt,
        "discrete": True,
        "discrete_acc": [
            (0.00, 0.00),
            (0.25, 0.00),
            (0.40, 0.00),
            (0.60, 0.00),
            (0.80, 0.00),
            (1.00, 0.00),
            (0.00, 0.20),
            (0.00, 0.50),
        ],
        "discrete_steer": [-0.2, -0.10, -0.05, 0.0, 0.05, 0.10, 0.2],
        "continuous_accel_range": [-3.0, 3.0],
        "continuous_steer_range": [-0.3, 0.3],
        "ego_vehicle_filter": "vehicle.lincoln*",
        "port": args.port,
        "town": args.town,
        "task_mode": args.task_mode,
        "max_time_episode": args.max_time_episode,
        "max_waypt": args.max_waypt,
        "obs_range": args.obs_range,
        "lidar_bin": args.lidar_bin,
        "d_behind": args.d_behind,
        "out_lane_thres": args.out_lane_thres,
        "desired_speed": args.desired_speed,
        "max_ego_spawn_times": args.max_ego_spawn_times,
        "display_route": True,
        "pixor_size": args.pixor_size,
        "pixor": args.pixor,
        "use_radar": args.use_radar,
        "radar_height": args.radar_height,
        "radar_x": args.radar_x,
        "radar_hfov": args.radar_hfov,
        "radar_vfov": args.radar_vfov,
        "radar_range": args.radar_range,
        "radar_pps": args.radar_pps,
        "radar_vmax": args.radar_vmax,
        "render_panels": args.render_panels,
        "render": args.render,
        "enable_pygame": args.enable_pygame,
        "frame_stack": args.frame_stack,
    }
    return gym.make("carla-v0", params=params), params


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])
        ),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device:", device)

    env, env_params = make_env(args)
    try:
        env.action_space.seed(args.seed)
    except Exception:
        pass

    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    initial_obs = reset_env(env, args.seed)
    obs_shape = initial_obs["birdeye"].shape

    print("birdeye shape:", obs_shape)
    print("num actions:", env.action_space.n)

    online_net = QNetwork(
        action_dim=env.action_space.n,
        birdeye_shape=obs_shape,
        final_layer=args.final_layer,
        depths=(16, 32, 64),
    ).to(device)

    target_net = QNetwork(
        action_dim=env.action_space.n,
        birdeye_shape=obs_shape,
        final_layer=args.final_layer,
        depths=(16, 32, 64),
    ).to(device)

    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=args.learning_rate)

    replay = ReplayBuffer(
        capacity=args.buffer_size,
        obs_shape=obs_shape,
        obs_dtype=np.uint8,
    )

    epinfos_buffer = deque([], maxlen=100)

    # -----------------------------------------------------
    # Prefill replay buffer
    # -----------------------------------------------------
    obs = initial_obs
    for step in trange(args.learning_starts, desc="Replay Warmup"):
        action = env.action_space.sample()
        next_obs, reward, done, info = step_env(env, action)

        k = info["action_repeat"] if isinstance(info, dict) and "action_repeat" in info else 1
        replay.add(
            obs["birdeye"],
            action,
            reward,
            done,
            next_obs["birdeye"],
            k,
        )

        obs = next_obs
        if done:
            obs = reset_env(env)

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------
    obs = reset_env(env)
    episode_count = 0
    episode_return = 0.0
    episode_length = 0
    start_time = time.time()

    for global_step in trange(args.total_timesteps, desc="BirdEye DQN Training"):
        epsilon = linear_schedule(
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay,
            global_step,
        )

        obs_b = obs_to_tensor(obs, device)
        action = online_net.act(
            obs_b.unsqueeze(0),
            epsilon=epsilon,
        )

        next_obs, reward, done, info = step_env(env, action)
        k = info["action_repeat"] if isinstance(info, dict) and "action_repeat" in info else 1

        replay.add(
            obs["birdeye"],
            action,
            reward,
            done,
            next_obs["birdeye"],
            k,
        )

        obs = next_obs
        episode_return += reward
        episode_length += 1

        terminal = done or (episode_length >= args.max_time_episode)
        if terminal:
            epinfo = {"r": round(episode_return, 6), "l": episode_length}
            epinfos_buffer.append(epinfo)

            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)

            episode_count += 1
            obs = reset_env(env)
            episode_return = 0.0
            episode_length = 0

        loss = torch.tensor(0.0, device=device)

        if global_step % args.train_frequency == 0:
            obs1, act, rew, done_batch, nobs1, kk = replay.sample(args.batch_size)

            obses_t1 = torch.from_numpy(obs1).to(device, non_blocking=True).float()
            actions_t = torch.from_numpy(act).to(device, non_blocking=True).view(-1, 1).long()
            rews_t = torch.from_numpy(rew).to(device, non_blocking=True).view(-1, 1).float()
            dones_t = torch.from_numpy(done_batch).to(device, non_blocking=True).view(-1, 1).float()
            new_obses_t1 = torch.from_numpy(nobs1).to(device, non_blocking=True).float()
            k_t = torch.from_numpy(kk).to(device, non_blocking=True).view(-1, 1).float()

            with torch.no_grad():
                max_target_q = target_net(new_obses_t1).amax(dim=1, keepdim=True)
                targets = rews_t + (args.gamma ** k_t) * (1.0 - dones_t) * max_target_q

            q_values = online_net(obses_t1)
            action_q_values = q_values.gather(1, actions_t)

            loss = torch.nn.functional.smooth_l1_loss(action_q_values, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            writer.add_scalar("losses/td_loss", loss.item(), global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

        if global_step % args.target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        if global_step % args.log_interval == 0:
            rew_mean = np.mean([e["r"] for e in epinfos_buffer]) if len(epinfos_buffer) > 0 else 0.0
            len_mean = np.mean([e["l"] for e in epinfos_buffer]) if len(epinfos_buffer) > 0 else 0.0
            sps = int((global_step + 1) / (time.time() - start_time + 1e-8))

            print()
            print(f"global_step={global_step}, SPS={sps}")
            print(f"epsilon={epsilon:.6f}")
            print(f"avg_reward={rew_mean:.6f}")
            print(f"avg_ep_len={len_mean:.2f}")
            print(f"episodes={episode_count}")
            print(f"loss={loss.item():.6f}")

            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/avg_reward_100", rew_mean, global_step)
            writer.add_scalar("charts/avg_ep_len_100", len_mean, global_step)

        if global_step % args.save_interval == 0 and global_step > 0:
            save_path = os.path.join(args.save_dir, f"{run_name}_{global_step}.pth")
            print(f"saving model to {save_path}")
            online_net.save(save_path)

    final_path = os.path.join(args.save_dir, f"{run_name}_final.pth")
    print(f"saving final model to {final_path}")
    online_net.save(final_path)

    env.close()
    writer.close()