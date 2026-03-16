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

## Model Architecture and Training Hyperparameters

The multi-modal architecture and its training configuration are summarized below.

---

### Feature Extraction (per Modality)

| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Number of convolutional layers | — | 3 × Conv2D per modality |
| **BEV kernel sizes** | — | [5×5, 3×3, 3×3] |
| **LiDAR kernel sizes** | — | [7×7, 5×5, 3×3] |
| Stride | — | [3, 2, 1] / [3, 3, 1] |
| Channels per layer | — | 8 → 16 → 16 |
| Activation function | — | ReLU / Binary LIF |
| Output embedding dimension | d_model | 32 |

---

### Cross-Attention Fusion Module

| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Number of attention heads | Nh | 8 |
| Feed-forward dimension | d_ff | 128 |
| Positional encoding | — | Learnable positional encoding |
| Normalization | — | LayerNorm |

---

### Decision Head

| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Activation function | — | ReLU / Binary LIF |
| Feed-forward dimension | d_ff | 512 |
| Output dimension | — | 5 (discrete actions in Highway-Env) |

---

### Reinforcement Learning Parameters

| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Algorithm | — | Deep Q-Network (DQN) |
| Discount factor | gamma | 0.99 |
| Replay buffer size | — | 5 × 10⁴ transitions |
| Batch size | — | 64 |
| Target network update frequency | — | Every 100 steps |
| Learning rate | eta₀ | 1 × 10⁻⁴ |
| Optimizer | — | Adam |
| Exploration schedule | epsilon-greedy | Linear decay from 1.0 to 0.1 over 7 × 10⁴ steps |
| Reward weights | — | Default Highway-Env (speed, collision, lane-change) |
| Training steps per scenario | — | 1 × 10⁵ |
| Evaluation episodes | — | 20–50 |

---

### Spiking-Specific Parameters

| Parameter | Setting | Value / Description |
|-----------|---------|---------------------|
| Membrane time constant (tau_m) | — | 2 |
| Spike threshold (positive) (Vth⁺) | — | 1.0 |
| Spike threshold (negative) (Vth⁻) | — | −4 |
| Reset mechanism (Vreset) | — | Subtractive reset (V ← V − Vth±) |
| Output spikes | — | Binary {0, +1}, Ternary {−1, 0, +1} |
| Surrogate gradient type | — | Arctangent |
| Simulation time window (Ts) | — | 5 |
| Neuron model | — | Binary LIF and ternary LIF (asymmetric thresholds) |
