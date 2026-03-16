Multi-Modal Deep Q-Learning for Autonomous Driving in CARLA

(MM-DQN · SSA-DQN · TTSA-DQN)

Overview

This repository provides implementations of multi-modal Deep Q-Learning pipelines for autonomous driving in CARLA 0.9.13 using the gym-carla interface. The methods operate under a discrete control formulation and leverage synchronized multi-sensor observations, including:

Bird’s-Eye-View (BEV) images

Radar / LiDAR representations

The repository includes three comparable frameworks:

MM-DQN — Standard multi-modal Deep Q-Network baseline

SSA-DQN — Spiking multi-modal DQN with Spiking Self-Attention

TTSA-DQN (Ours) — Spiking multi-modal DQN with Ternary Temporal Spiking Attention

All methods are evaluated on CARLA Town03 under identical observation and action configurations to ensure fair comparison.

Repository Structure
.
├── MM_DQN.py              # Multi-Modal DQN baseline
├── SSA_DQN.py             # SSA-based spiking DQN
├── TTSA_DQN.py            # TTSA-based spiking DQN (proposed)
├── test.py                # Environment validation script
├── models_DQN/            # Saved MM-DQN checkpoints
├── models_SSA_DQN/        # Saved SSA-DQN checkpoints
├── models_TTSA_DQN/       # Saved TTSA-DQN checkpoints
├── runs/                  # TensorBoard logs
└── README.md
System Requirements

Operating System: Ubuntu 24.04

Python: 3.8

Simulator: CARLA 0.9.13

GPU: NVIDIA GPU recommended (especially for spiking models)

Environment Manager: Conda (recommended)

Installation
1. Create Conda Environment
conda create -n carla_rl python=3.8 -y
conda activate carla_rl
2. Clone Repository
git clone <your-repository-url>
cd <your-repository-folder>
3. Install Dependencies
pip install -r requirements.txt
CARLA 0.9.13 Setup

Download and extract CARLA 0.9.13 from the official release.

Add the CARLA Python API to your environment:

export PYTHONPATH=$PYTHONPATH:/path/to/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg

(Optional — make permanent)

echo 'export PYTHONPATH=$PYTHONPATH:/path/to/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg' >> ~/.bashrc
source ~/.bashrc

Note: The .egg filename may differ slightly depending on your installation. Verify inside:
/path/to/CARLA_0.9.13/PythonAPI/carla/dist/

Running CARLA
Windowed Mode
cd /path/to/CARLA_0.9.13
./CarlaUE4.sh -windowed -carla-port=2000
Headless Mode (Recommended for Training)
cd /path/to/CARLA_0.9.13
SDL_VIDEODRIVER=dummy ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port=2000
Alternative Headless Configuration
DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000
Usage

Ensure the CARLA server is running before launching training.

MM-DQN (Baseline)
python MM_DQN.py --render False --enable-pygame False --port 2000
SSA-DQN
python SSA_DQN.py --render False --enable-pygame False --port 2000
TTSA-DQN (Proposed)
python TTSA_DQN.py --render False --enable-pygame False --port 2000
TensorBoard Logging

Training logs are stored in:

runs/

Launch TensorBoard:

tensorboard --logdir runs

Typical metrics include:

Episodic return

Episodic length

TD loss

Epsilon schedule

Training throughput

Saved Models

Checkpoints are stored in:

models_DQN/
models_SSA_DQN/
models_TTSA_DQN/

Example:

<exp_name>__seed0__5000.pth
<exp_name>__seed0__10000.pth
Quick Environment Test

After completing setup:

python test.py

This verifies:

CARLA connectivity

Observation pipeline

Action execution

Common Issues
1. gym_carla not found
pip install -e gym-carla
2. CARLA import error
echo $PYTHONPATH

Ensure the CARLA egg is correctly appended.

3. Address already in use
./CarlaUE4.sh -carla-port=2001
python MM_DQN.py --port 2001
4. Slow Training Performance

Reduce:

number_of_vehicles

display_size

render_panels

Keep:

--render False --enable-pygame False
