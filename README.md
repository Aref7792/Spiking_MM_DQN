## Introduction

This repository contains the official implementation of the paper:

**“New Spiking Architecture for Multi-Modal Decision-Making in Autonomous Vehicles”**  
*(submitted to IROS 2026)*

This work introduces a spiking-based multi-modal deep reinforcement learning framework for autonomous driving decision-making. The proposed architecture integrates heterogeneous sensory modalities and leverages neuromorphic computation to efficiently capture both spatial and temporal dynamics in complex traffic environments.

For experimental evaluation, we use two simulation platforms:

- **Highway Environment** (Farama Foundation)  
  For details about the reward function, observation space, and environment configuration, please refer to:  
  https://highway-env.farama.org/index.html

- **Gym-based CARLA Simulator**  
  We build on the CARLA Gym interface available at:  
  https://github.com/cjy1992/gym-carla/tree/master

To support multi-modal perception, we modify the CARLA Gym environment such that the observation space includes both **LiDAR** and **radar** information. In particular, relative velocity measurements from **four radar sensors** surrounding the ego vehicle are incorporated and encoded into pixel-space representations, enabling the model to capture dynamic scene information alongside spatial features.

The repository is structured as a minimal and reproducible implementation. The main functionality is provided through **two core code files**, which include:

- Model architecture and training pipeline  
- Hyperparameter configuration  
- Environment setup and preprocessing  
- Logging, evaluation, and experiment utilities  

## CARLA Model Architecture and Training Hyperparameters

The CARLA experiments use a multi-modal DQN architecture with bird’s-eye-view (BEV) and radar observations, cross-attention fusion, and a discrete-action control setup in the gym-based CARLA simulator.


| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Input modalities | — | Bird’s-eye view (BEV) and radar |
| Fusion type | — | Cross-attention fusion |
| Latent embedding size | latent_size | 64 |
| Number of attention heads | num_heads | 8 |
| Final hidden layer size | final_layer | 512 |
| Q-network output | — | Discrete action values |
| BEV encoder channels | — | 16 → 32 → 64 |
| Radar encoder channels | — | 16 → 32 → 64 |
| BEV convolution kernels | — | 5×5, 3×3, 3×3 |
| Radar convolution kernels | — | 5×5, 3×3, 3×3 |
| BEV convolution strides | — | 2, 2, 1 |
| Radar convolution strides | — | 2, 2, 1 |
| Encoder activation | — | ReLU |
| Attention block normalization | — | LayerNorm |
| Transformer MLP activation | — | GELU |
| Positional encoding | — | Learnable positional embedding |
| Output head | — | Linear Q-head with ReLU hidden layer |

---

### Reinforcement Learning Parameters

| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Algorithm | — | Deep Q-Network (DQN) |
| Total training steps | — | 1 × 10⁵ |
| Learning rate | — | 1 × 10⁻⁴ |
| Discount factor | gamma | 0.99 |
| Batch size | — | 32 |
| Replay buffer size | — | 2 × 10⁵ transitions |
| Replay warmup / learning starts | — | 10000 steps |
| Training frequency | — | Every 4 environment steps |
| Target network update frequency | — | Every 200 steps |
| Loss function | — | Smooth L1 loss (Huber loss) |
| Optimizer | — | Adam |
| Exploration strategy | — | Epsilon-greedy |
| Epsilon start | — | 1.0 |
| Epsilon end | — | 0.1 |
| Epsilon decay duration | — | 1 × 10⁵ steps |
| Model save interval | — | Every 5000 steps |
| Logging interval | — | Every 1000 steps |

---


## Highway-Env Model Architecture and Training Hyperparameters

The Highway-Env experiments use a multi-modal DQN architecture with bird’s-eye-view (BEV) and LiDAR observations, cross-attention fusion, and a discrete-action control setup in the Highway-Env simulator.

---

### Model Architecture

| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Input modalities | — | Bird’s-eye view (BEV) and LiDAR |
| Fusion type | — | Cross-attention fusion |
| Output embedding size | d_model | 32 |
| Number of attention heads | Nh | 8 |
| Fusion feed-forward dimension | d_ff | 128 |
| Decision head hidden size | d_ff | 512 |
| Q-network output | — | 5 discrete actions (Highway-Env) |
| BEV encoder channels | — | 8 → 16 → 16 |
| LiDAR encoder channels | — | 8 → 16 → 16 |
| BEV convolution kernels | — | 5×5, 3×3, 3×3 |
| LiDAR convolution kernels | — | 7×7, 5×5, 3×3 |
| BEV convolution strides | — | 3, 2, 1 |
| LiDAR convolution strides | — | 3, 3, 1 |
| Encoder activation | — | ReLU / Binary LIF |
| Attention block normalization | — | LayerNorm |
| Transformer MLP activation | — | ReLU |
| Positional encoding | — | Learnable positional embedding |
| Output head | — | Linear Q-head with ReLU hidden layer |

---

### Reinforcement Learning Parameters

| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Algorithm | — | Deep Q-Network (DQN) |
| Total training steps | — | 1 × 10⁵ per scenario |
| Learning rate | — | 1 × 10⁻⁴ |
| Discount factor | gamma | 0.99 |
| Batch size | — | 64 |
| Replay buffer size | — | 5 × 10⁴ transitions |
| Target network update frequency | — | Every 100 steps |
| Loss function | — | DQN temporal-difference loss |
| Optimizer | — | Adam |
| Exploration strategy | — | Epsilon-greedy |
| Epsilon schedule | — | Linear decay from 1.0 to 0.1 over 7 × 10⁴ steps |
| Reward weights | — | Default Highway-Env reward (speed, collision, lane-change) |
| Evaluation episodes | — | 20–50 |

----
### Spiking-Specific Hyperparameters

The spiking neuron configuration is summarized as follows:

| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Neuron type | — | Binary LIF and ternary LIF neurons (asymmetric thresholds) |
| Usage across modules | — | Binary LIF used in feature extraction and decision layers; ternary LIF used in cross-attention fusion |
| Membrane time constant | tau_m | 2 |
| Positive spike threshold | Vth⁺ | 1.0 |
| Negative spike threshold | Vth⁻ | −4 |
| Reset mechanism | Vreset | Subtractive reset (V ← V − Vth±) |
| Output spike representation | — | Binary {0, +1} and ternary {−1, 0, +1} |
| Surrogate gradient type | — | Arctangent |
| Simulation time window | Ts | 5 time steps |


