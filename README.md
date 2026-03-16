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

The repository is structured as a minimal and reproducible implementation. The main functionality is provided through **two sub-sections**, which include implementation for the Highway Env and CARLA Gym. 

## CARLA Model Architecture and Training Hyperparameters

The CARLA experiments use a multi-modal DQN architecture with bird’s-eye-view (BEV) and radar observations, cross-attention fusion, and a discrete-action control setup in the gym-based CARLA simulator.

---

### Model Architecture

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



