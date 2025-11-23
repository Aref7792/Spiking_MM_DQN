# Spiking Multi-Modal Deep Q-Network (MM-DQN)

This repository contains the official implementation of the paper:

**“A New Spiking Architecture for Multi-Modal Decision-Making in Autonomous Vehicles”**  
Submitted to **AAMAS 2026 (The 25th International Conference on Autonomous Agents and Multi-Agent Systems)**.

The framework introduces a multi-modal spiking deep Q-network that integrates BEV camera representations and LiDAR-derived occupancy maps using novel spiking cross-attention mechanisms.

---

## Table of Contents
- Overview
- Repository Structure
- Environments
- Architecture Variants
- Installation & Requirements
- Torch & CUDA Versions
- How to Run
- Results
- Citation
- License

---

# Overview

This repository includes two versions of the implementation:

1. Submitted Code — the exact version included during the conference submission (located in `/Submitted_Code`).  
2. Modular Implementation — a cleaner, extensible, and maintainable version intended as the long-term official release.

The project supports two gymnasium-based autonomous driving environments:
- Multi-lane Highway
- Roundabout Navigation

Both scenarios involve driving interactions that require fast temporal reasoning and multimodal fusion.

---

# Repository Structure

MM-DQN/
│
├── Submitted_Code/
│   ├── highway/
│   ├── roundabout/
│   └── ...
│
├── Modular_Code/
│   ├── models/
│   ├── utils/
│   ├── configs/
│   ├── train_test.py
│   └── ...
│
├── results/
│   ├── highway/
│   ├── roundabout/
│   └── ...
│
└── README.md

---

# Environments

1. Highway-v0 — A multi-lane, multi-agent driving environment, requiring rapid decision-making and safe lane transitions.

2. Roundabout-v0 — A more complex scenario requiring negotiation with multiple agents while navigating circular traffic flows.

Both are based on the `highway_env` library.

---

# Architecture Variants

Three architectures are provided for each environment:

1. MM-DQN (Non-Spiking Baseline) — A conventional deep Q-network operating on BEV and LiDAR image observations.

2. SSA — Standard Spiking Attention — A spiking extension using binary spikes and standard spike-based attention.

3. TTSA — Temporal-Aware Ternary Spiking Attention (Proposed) — Provides temporal-aware ternary spikes, cross-modal spiking attention, improved reward performance, and higher representational capacity.

---

# Installation & Requirements

Python Version: 3.9 – 3.11

Install Core Dependencies:
pip install highway_env gymnasium snntorch numpy torch einops msgpack msgpack_numpy tensorboard

Optional:
pip install opencv-python matplotlib

---

# Torch & CUDA Versions

The repository was tested with:

torch: 2.7.1
CUDA Toolkit: 12.8
numpy: 2.2.6
snntorch: 0.9.4
gymnasium: 1.2.0
highway-env: 1.10.1
einops: 0.8.1
tensorboard: 2.20.0


---

# How to Run

python train_test.py --seeds <number_of_seeds> --mode <nonspiking | SSA | TTSA> --scenario <highway-v0 | roundabout-v0>

Examples:
python train_test.py --seeds 5 --mode TTSA --scenario highway-v0  
python train_test.py --seeds 3 --mode SSA --scenario roundabout-v0  

---

# Results

Highway Scenario:  
results/highway/nonspiking.gif  
results/highway/ssa.gif  
results/highway/ttsa.gif  

Roundabout Scenario:  
results/roundabout/nonspiking.gif  
results/roundabout/ssa.gif  
results/roundabout/ttsa.gif  

---

# Citation


---

# License


