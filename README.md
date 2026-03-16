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

Additional implementation details (training configurations, reward shaping, and ablations) will be released upon acceptance.
