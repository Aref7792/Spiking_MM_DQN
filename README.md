# Spiking Multi-Modal Deep Q-Network (MM-DQN)

This repository provides the official implementation of the **New Spiking Architecture for Multi-Modal Decision-Making in Autonomous Vehicles** paper submitted to The 25th International Conference on Autonomous Agents and Multi-Agent Systems. 

# Submitted Code 
You can find the code attached to the paper during submission in the "Submitted_Code" folder.  
Two distinct environments are included:

- **Highway** — Straight multi-lane highway driving.  
- **Roundabout** — Complex decision-making and interaction with multiple agents.

Each environment contains three code variants:
- **MM_DQN** — Non-spiking baseline model.  
- **SSA** — Spiking DQN with *Standard Spiking Attention*.  
- **TTSA** — Spiking DQN with *Temporal-Aware Ternary Spiking Attention* (proposed).

Before running the code, please install the following dependencies: 
```bash
pip install highway_env snntorch gymnasium

```
---

# Modular Implementation

A more professional version of the code is provided to be released as the formal repository of the paper. 

To run the code please run: 

python train_test.py --seeds # of seeds --mode nonspiking, SSA, or TTSA --scenario highway-v0, roundabout-v0

# Results

The renderings of the best trained model on each enviroment is as follows: 

## Highway Scenario

![Spiking](results/spiking_smooth.gif)
![Non-Spiking](results/nonspiking_smooth.gif)

## 🧩 Dependencies

Before running the code, please install the following dependencies:

```bash
pip install highway_env snntorch gymnasium


