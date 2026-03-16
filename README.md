Multi-Modal DQN for CARLA (Vanilla, SSA, TTSA)

This repository provides three Deep Q-Learning pipelines for multi-modal autonomous driving in CARLA 0.9.13 using bird’s-eye-view (BEV) and radar observations:

1. Vanilla Multi-Modal DQN
2. SSA-based Spiking Multi-Modal DQN
3. TTSA-based Spiking Multi-Modal DQN (Ours)

All models use the gym-carla wrapper with a discrete control space and are tested on Town03.

---

Repository Structure

.
├── dqn_carla.py
├── ssa_dqn_carla.py
├── ttsa_dqn_carla.py
├── test.py
├── models_DQN/
├── models_SSA_DQN/
├── models_TTSA_DQN/
├── runs/
└── README.txt

---

System Requirements

OS: Ubuntu 24.04
Python: 3.8
CARLA: 0.9.13
GPU: NVIDIA GPU recommended
Conda: recommended for environment management

---

Installation

1. Create Conda Environment

conda create -n carla_rl python=3.8 -y
conda activate carla_rl

---

2. Clone Repository

git clone <your-repository-url>
cd <your-repository-folder>

---

3. Install Dependencies

If using requirements:

pip install -r requirements.txt

Or manually:

pip install torch torchvision torchaudio
pip install numpy gym tqdm tensorboard einops msgpack msgpack-numpy tyro snntorch pygame scikit-image

If packaged:

pip install -e .

---

Install gym-carla

git clone https://github.com/cjy1992/gym-carla.git
cd gym-carla
pip install -r requirements.txt
pip install -e .
cd ..

---

CARLA 0.9.13 Setup

Download and extract CARLA 0.9.13.

Add the CARLA Python API to your environment:

export PYTHONPATH=$PYTHONPATH:/path/to/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg

(Optional) Make permanent:

echo 'export PYTHONPATH=$PYTHONPATH:/path/to/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg' >> ~/.bashrc
source ~/.bashrc

Note: the .egg filename may differ slightly. Check:

/path/to/CARLA_0.9.13/PythonAPI/carla/dist/

---

Running CARLA

Windowed Mode:

cd /path/to/CARLA_0.9.13
./CarlaUE4.sh -windowed -carla-port=2000

Headless Mode (Recommended):

cd /path/to/CARLA_0.9.13
SDL_VIDEODRIVER=dummy ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port=2000

Alternative:

DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000

---

Usage

Make sure CARLA is already running before training.

---

1. Vanilla Multi-Modal DQN

python dqn_carla.py --render False --enable-pygame False --port 2000

---

2. SSA-Based Spiking DQN

python ssa_dqn_carla.py --render False --enable-pygame False --port 2000

---

3. TTSA-Based Spiking DQN (Ours)

python ttsa_dqn_carla.py --render False --enable-pygame False --port 2000

---

Headless Execution (Recommended)

SDL_VIDEODRIVER=dummy python dqn_carla.py --render False --enable-pygame False --port 2000

---

TensorBoard Logging

Logs are saved in:

runs/

Launch TensorBoard:

tensorboard --logdir runs

Metrics include:

episodic return
episodic length
TD loss
epsilon schedule
training speed

---

Saved Models

Checkpoints are saved in:

models_DQN/
models_SSA_DQN/
models_TTSA_DQN/

Example:

<exp_name>__seed0__5000.pth
<exp_name>__seed0__10000.pth

---

Quick Environment Test

After setup:

python test.py

This verifies:

CARLA connection
observation format
action execution

---

Common Issues

1. gym_carla not found

pip install -e gym-carla

---

2. CARLA import error

Check:

echo $PYTHONPATH

---

3. Address already in use

Use another port:

./CarlaUE4.sh -carla-port=2001

Then:

python dqn_carla.py --port 2001

---

4. Slow training

Reduce:

number_of_vehicles
display_size
render_panels

Keep:

--render False --enable-pygame False

---

Recommended Workflow

For stable experimentation:

1. Run Vanilla DQN first
2. Run SSA-DQN
3. Run TTSA-DQN

This isolates environment issues from spiking-attention implementation issues.

---

Citation

If you use this repository in academic work, please cite the associated paper or implementation.
