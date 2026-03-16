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
import torch as th
import torch.nn as nn
import torch.optim as optim
import einops
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import snntorch
from snntorch import spikegen

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()


# =========================================================
# Args
# =========================================================
@dataclass
class Args:
    exp_name: str = "carla_multimodal_ssa_dqn"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True

    total_timesteps: int = int(1e5)
    learning_rate: float = 1e-4
    gamma: float = 0.99

    batch_size: int = 32
    buffer_size: int = int(2e5)
    learning_starts: int = 5000
    train_frequency: int = 4
    target_update_freq: int = 200

    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = int(1e5)

    log_interval: int = 1000
    save_interval: int = 5000

    save_dir: str = "models_SSA_DQN"
    log_dir: str = "runs"

    latent_size: int = 64
    final_layer: int = 512
    num_heads: int = 8
    num_steps: int = 5

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
    frame_stack: int = 1


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


def obs_to_tensors(obs, device):
    birdeye = torch.as_tensor(obs["birdeye"], dtype=torch.float32, device=device)
    radar = torch.as_tensor(obs["radar"], dtype=torch.float32, device=device)
    return birdeye, radar


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# =========================================================
# Replay Buffer
# =========================================================
class ReplayBuffer:
    def __init__(self, capacity, obs1_shape, obs2_shape, obs1_dtype=np.uint8, obs2_dtype=np.uint8):
        self.capacity = int(capacity)
        self.ptr = 0
        self.full = False

        self.obs1 = np.empty((self.capacity, *obs1_shape), dtype=obs1_dtype)
        self.obs2 = np.empty((self.capacity, *obs2_shape), dtype=obs2_dtype)
        self.nobs1 = np.empty((self.capacity, *obs1_shape), dtype=obs1_dtype)
        self.nobs2 = np.empty((self.capacity, *obs2_shape), dtype=obs2_dtype)

        self.act = np.empty((self.capacity,), dtype=np.int64)
        self.rew = np.empty((self.capacity,), dtype=np.float32)
        self.done = np.empty((self.capacity,), dtype=np.float32)
        self.k = np.empty((self.capacity,), dtype=np.int64)

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def add(self, obs1, obs2, action, reward, done, next_obs1, next_obs2, k):
        i = self.ptr
        self.obs1[i] = obs1
        self.obs2[i] = obs2
        self.nobs1[i] = next_obs1
        self.nobs2[i] = next_obs2
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
            self.obs1[idx],
            self.obs2[idx],
            self.act[idx],
            self.rew[idx],
            self.done[idx],
            self.nobs1[idx],
            self.nobs2[idx],
            self.k[idx],
        )


# =========================================================
# Model blocks
# =========================================================
class SSA(nn.Module):
    def __init__(self, dim, num_heads, num_steps):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_steps = num_steps
        self.scale = 0.18

        self.q_conv = layer_init(nn.Linear(dim, dim))
        self.q_lif = snntorch.Leaky(0.6, 1)

        self.k_conv = layer_init(nn.Linear(dim, dim))
        self.k_lif = snntorch.Leaky(0.6, 1)

        self.v_conv = layer_init(nn.Linear(dim, dim))
        self.v_lif = snntorch.Leaky(0.6, 1)

        self.attn_lif = snntorch.Leaky(0.6, 1)

        self.proj_conv = layer_init(nn.Linear(dim, dim))
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = snntorch.Leaky(0.6, 1)

    def forward(self, q, k, v):
        mem_q = self.q_lif.reset_mem()
        mem_k = self.k_lif.reset_mem()
        mem_v = self.v_lif.reset_mem()
        mem_att = self.attn_lif.reset_mem()
        mem_proj = self.proj_lif.reset_mem()

        att_spk_out = []

        # q, k, v: (T, B, N, C)
        for step in range(self.num_steps):
            b, n, c = q[step].shape

            q_conv_out = self.q_conv(q[step])
            q_conv_out, mem_q = self.q_lif(q_conv_out, mem_q)
            q_heads = q_conv_out.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

            k_conv_out = self.k_conv(k[step])
            k_conv_out, mem_k = self.k_lif(k_conv_out, mem_k)
            k_heads = k_conv_out.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

            v_conv_out = self.v_conv(v[step])
            v_conv_out, mem_v = self.v_lif(v_conv_out, mem_v)
            v_heads = v_conv_out.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

            x = (q_heads @ k_heads.transpose(-2, -1)) * self.scale
            x = x @ v_heads

            b2, h, l, dh = x.shape
            x = x.permute(0, 2, 1, 3).contiguous().view(b2, l, h * dh)
            x, mem_att = self.attn_lif(x, mem_att)

            x = self.proj_conv(x)
            cur = x.permute(0, 2, 1).contiguous()
            cur = self.proj_bn(cur)
            cur = cur.permute(0, 2, 1).contiguous()
            spk_proj, mem_proj = self.proj_lif(cur, mem_proj)

            att_spk_out.append(spk_proj)

        return th.stack(att_spk_out, dim=0)


class EncoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads, num_steps):
        super().__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.num_steps = num_steps

        self.norm1 = nn.LayerNorm(latent_size)
        self.norm2 = nn.LayerNorm(latent_size)
        self.norm3 = nn.LayerNorm(latent_size)

        self.multihead = SSA(latent_size, num_heads, num_steps)

        self.enc_mlp1 = layer_init(nn.Linear(latent_size, latent_size * 4))
        self.gelu = snntorch.Leaky(0.6, 1)
        self.enc_mlp2 = layer_init(nn.Linear(latent_size * 4, latent_size))

    def forward(self, embedded_patches1, embedded_patches2):
        mem = self.gelu.reset_mem()
        spk_out = []

        q = self.norm1(embedded_patches1)
        v = self.norm2(embedded_patches2)
        attention_out = self.multihead(q, v, v)

        for step in range(self.num_steps):
            first_added = attention_out[step] + embedded_patches1[step]
            first_added = self.norm3(first_added)
            cur = self.enc_mlp1(first_added)
            spk, mem = self.gelu(cur, mem)
            ff_out = self.enc_mlp2(spk)
            final_out = ff_out + first_added
            spk_out.append(final_out)

        return th.stack(spk_out, dim=0)


class InputEmbedding(nn.Module):
    def __init__(self, n_channels, latent_size, dim1, dim2, num_steps):
        super().__init__()
        self.linear_projection = layer_init(nn.Linear(n_channels, latent_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim1, dim2))
        self.num_steps = num_steps

    def forward(self, input_data):
        # input_data: (T, B, C, H, W)
        patches = einops.rearrange(input_data, "T b c h w -> T b h w c")
        patches = patches + self.pos_embedding.unsqueeze(-1)
        patches = einops.rearrange(patches, "T b h w c -> T b (h w) c")
        return self.linear_projection(patches)


class SpikingMultiModalEncoder(nn.Module):
    def __init__(
        self,
        birdeye_shape,
        radar_shape,
        latent_size=64,
        num_heads=8,
        num_steps=5,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
        radar_div_255=True,
    ):
        super().__init__()

        n_input_channels1 = birdeye_shape[-1]
        n_input_channels2 = radar_shape[-1]

        h1, w1 = birdeye_shape[0], birdeye_shape[1]
        h2, w2 = radar_shape[0], radar_shape[1]

        dim1_h, dim1_w = self._conv_output_hw(h1, w1)
        dim2_h, dim2_w = self._conv_output_hw(h2, w2)

        if (dim1_h != dim2_h) or (dim1_w != dim2_w):
            raise ValueError(
                f"Birdeye and radar conv outputs must match spatially. "
                f"Got birdeye=({dim1_h}, {dim1_w}) and radar=({dim2_h}, {dim2_w})."
            )

        self.dim_h = dim1_h
        self.dim_w = dim1_w
        self.output_dim = latent_size * self.dim_h * self.dim_w
        self.num_steps = num_steps
        self.radar_div_255 = radar_div_255

        self.bev_conv1 = layer_init(nn.Conv2d(n_input_channels1, depths1[0], kernel_size=5, stride=2))
        self.bev_lif1 = snntorch.Leaky(0.6, 1)
        self.bev_conv2 = layer_init(nn.Conv2d(depths1[0], depths1[1], kernel_size=3, stride=2))
        self.bev_lif2 = snntorch.Leaky(0.6, 1)
        self.bev_conv3 = layer_init(nn.Conv2d(depths1[1], depths1[2], kernel_size=3, stride=1))
        self.bev_lif3 = snntorch.Leaky(0.6, 1)
        self.bev_emb = InputEmbedding(depths1[2], latent_size, self.dim_h, self.dim_w, num_steps)

        self.radar_conv1 = layer_init(nn.Conv2d(n_input_channels2, depths2[0], kernel_size=5, stride=2))
        self.radar_lif1 = snntorch.Leaky(0.6, 1)
        self.radar_conv2 = layer_init(nn.Conv2d(depths2[0], depths2[1], kernel_size=3, stride=2))
        self.radar_lif2 = snntorch.Leaky(0.6, 1)
        self.radar_conv3 = layer_init(nn.Conv2d(depths2[1], depths2[2], kernel_size=3, stride=1))
        self.radar_lif3 = snntorch.Leaky(0.6, 1)
        self.radar_emb = InputEmbedding(depths2[2], latent_size, self.dim_h, self.dim_w, num_steps)

        self.cross = EncoderBlock(latent_size, num_heads, num_steps)

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

    def _normalize_birdeye(self, x):
        return x.float() / 255.0

    def _normalize_radar(self, x):
        if self.radar_div_255:
            return x.float() / 255.0
        return x.float()

    def forward(self, x1, x2):
        x1 = self._normalize_birdeye(x1.permute(0, 3, 1, 2))
        x2 = self._normalize_radar(x2.permute(0, 3, 1, 2))

        x1_spikes = spikegen.rate(x1, self.num_steps).to(device=x1.device)
        x2_spikes = spikegen.rate(x2, self.num_steps).to(device=x2.device)

        mem_b1 = self.bev_lif1.reset_mem()
        mem_b2 = self.bev_lif2.reset_mem()
        mem_b3 = self.bev_lif3.reset_mem()

        mem_r1 = self.radar_lif1.reset_mem()
        mem_r2 = self.radar_lif2.reset_mem()
        mem_r3 = self.radar_lif3.reset_mem()

        spk_bev = []
        spk_radar = []

        for step in range(self.num_steps):
            cur_b1 = self.bev_conv1(x1_spikes[step])
            spk_b1, mem_b1 = self.bev_lif1(cur_b1, mem_b1)
            cur_b2 = self.bev_conv2(spk_b1)
            spk_b2, mem_b2 = self.bev_lif2(cur_b2, mem_b2)
            cur_b3 = self.bev_conv3(spk_b2)
            spk_b3, mem_b3 = self.bev_lif3(cur_b3, mem_b3)
            spk_bev.append(spk_b3)

            cur_r1 = self.radar_conv1(x2_spikes[step])
            spk_r1, mem_r1 = self.radar_lif1(cur_r1, mem_r1)
            cur_r2 = self.radar_conv2(spk_r1)
            spk_r2, mem_r2 = self.radar_lif2(cur_r2, mem_r2)
            cur_r3 = self.radar_conv3(spk_r2)
            spk_r3, mem_r3 = self.radar_lif3(cur_r3, mem_r3)
            spk_radar.append(spk_r3)

        spk_bev = th.stack(spk_bev, dim=0)
        spk_radar = th.stack(spk_radar, dim=0)

        emb_bev = self.bev_emb(spk_bev)
        emb_radar = self.radar_emb(spk_radar)

        fused = self.cross(emb_radar, emb_bev)
        fused = einops.rearrange(
            fused, "T b (h w) c -> T b c h w", h=self.dim_h, w=self.dim_w
        )
        return fused


# =========================================================
# Agent
# =========================================================
class QNetwork(nn.Module):
    def __init__(
        self,
        action_dim,
        birdeye_shape,
        radar_shape,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
        final_layer=512,
        latent_size=64,
        num_heads=8,
        num_steps=5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_steps = num_steps

        self.encoder = SpikingMultiModalEncoder(
            birdeye_shape=birdeye_shape,
            radar_shape=radar_shape,
            latent_size=latent_size,
            num_heads=num_heads,
            num_steps=num_steps,
            depths1=depths1,
            depths2=depths2,
            radar_div_255=True,
        )

        self.fc8 = layer_init(nn.Linear(self.encoder.output_dim, final_layer))
        self.fc9 = snntorch.Leaky(0.6, 1)
        self.fc10 = layer_init(nn.Linear(final_layer, action_dim), std=0.01)

    def forward(self, x1, x2):
        fused = self.encoder(x1, x2)

        mem9 = self.fc9.reset_mem()
        out = []

        for step in range(self.num_steps):
            cur = self.fc8(fused[step].reshape(fused.shape[1], -1))
            spk9, mem9 = self.fc9(cur, mem9)
            q = self.fc10(spk9)
            out.append(q)

        return th.stack(out, dim=0).sum(dim=0)

    def act(self, x1, x2, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            q_values = self.forward(x1, x2)
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
        "discrete_steer": [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
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
    obs_b_shape = initial_obs["birdeye"].shape
    obs_r_shape = initial_obs["radar"].shape

    print("birdeye shape:", obs_b_shape)
    print("radar shape:", obs_r_shape)
    print("num actions:", env.action_space.n)

    online_net = QNetwork(
        action_dim=env.action_space.n,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        num_steps=args.num_steps,
    ).to(device)

    target_net = QNetwork(
        action_dim=env.action_space.n,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        num_steps=args.num_steps,
    ).to(device)

    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=args.learning_rate)

    replay = ReplayBuffer(
        capacity=args.buffer_size,
        obs1_shape=obs_b_shape,
        obs2_shape=obs_r_shape,
        obs1_dtype=np.uint8,
        obs2_dtype=np.uint8,
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
            obs["radar"],
            action,
            reward,
            done,
            next_obs["birdeye"],
            next_obs["radar"],
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

    for global_step in trange(args.total_timesteps, desc="SSA-DQN Training"):
        epsilon = linear_schedule(
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay,
            global_step,
        )

        obs_b, obs_r = obs_to_tensors(obs, device)
        action = online_net.act(
            obs_b.unsqueeze(0),
            obs_r.unsqueeze(0),
            epsilon=epsilon,
        )

        next_obs, reward, done, info = step_env(env, action)
        k = info["action_repeat"] if isinstance(info, dict) and "action_repeat" in info else 1

        replay.add(
            obs["birdeye"],
            obs["radar"],
            action,
            reward,
            done,
            next_obs["birdeye"],
            next_obs["radar"],
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
            obs1, obs2, act, rew, done_batch, nobs1, nobs2, kk = replay.sample(args.batch_size)

            obses_t1 = torch.from_numpy(obs1).to(device, non_blocking=True).float()
            obses_t2 = torch.from_numpy(obs2).to(device, non_blocking=True).float()
            actions_t = torch.from_numpy(act).to(device, non_blocking=True).view(-1, 1).long()
            rews_t = torch.from_numpy(rew).to(device, non_blocking=True).view(-1, 1).float()
            dones_t = torch.from_numpy(done_batch).to(device, non_blocking=True).view(-1, 1).float()
            new_obses_t1 = torch.from_numpy(nobs1).to(device, non_blocking=True).float()
            new_obses_t2 = torch.from_numpy(nobs2).to(device, non_blocking=True).float()
            k_t = torch.from_numpy(kk).to(device, non_blocking=True).view(-1, 1).float()

            with torch.no_grad():
                max_target_q = target_net(new_obses_t1, new_obses_t2).amax(dim=1, keepdim=True)
                targets = rews_t + (args.gamma ** k_t) * (1.0 - dones_t) * max_target_q

            q_values = online_net(obses_t1, obses_t2)
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