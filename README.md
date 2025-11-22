# Spiking Multi-Modal Deep Q-Network (MM-DQN)

This repository provides the implementation of the **Multi-Modal Deep Q-Network (MM-DQN)** and its **spiking variants** designed for autonomous driving scenarios using the *Highway-Env* simulator.  
Two distinct environments are included:

- **Highway** — Straight multi-lane highway driving.  
- **Roundabout** — Complex decision-making and interaction with multiple agents.

Each environment contains three code variants:
- **MM_DQN** — Non-spiking baseline model.  
- **SSA** — Spiking DQN with *Standard Spiking Attention*.  
- **TTSA** — Spiking DQN with *Temporal-Aware Ternary Spiking Attention* (proposed).

---

## 🧠 Spiking DSQN Result
! [Spiking Result](results/spiking.gif)

## ⚙️ Non-Spiking DQN Result
! [Non-Spiking Result](results/nonspiking.gif)

## 🧩 Dependencies

Before running the code, please install the following dependencies:

```bash
pip install highway_env snntorch gymnasium


