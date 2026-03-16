Multi-Modal Deep Q-Learning for Autonomous Driving in CARLA
(MM-DQN, SSA-DQN, TTSA-DQN)

Overview
This section provides Deep Q-Learning pipelines for multi-modal autonomous driving using CARLA 0.9.13. The implementations leverage bird’s-eye-view (BEV) and radar/Lidar observations through the gym-carla interface and operate under a discrete control formulation. The repository includes:

1. Multi-Modal DQN
2. SSA-Based Spiking Multi-Modal DQN
3. TTSA-Based Spiking Multi-Modal DQN (proposed approach)

All methods are validated on CARLA Town03 with consistent observation and action configurations to ensure fair comparison.

---

Repository Structure

.
├── MM_DQN.py
├── SSA_DQN.py
├── TTSA_DQN.py
├── test.py
├── models_DQN/
├── models_SSA_DQN/
├── models_TTSA_DQN/
├── runs
└── README.txt

---

System Requirements

Operating System: Ubuntu 24.04
Python Version: 3.8
Simulator: CARLA 0.9.13
GPU: NVIDIA GPU recommended for spiking variants
Environment Manager: Conda (recommended)

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

Using requirements file:

pip install -r requirements.txt


---

CARLA 0.9.13 Setup

Download and extract CARLA 0.9.13 from the official release.

Add the CARLA Python API to your environment:

export PYTHONPATH=$PYTHONPATH:/path/to/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

(Optional) Make permanent:

echo 'export PYTHONPATH=$PYTHONPATH:/path/to/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg' >> ~/.bashrc
source ~/.bashrc

Note: The egg filename may vary slightly depending on your installation. Verify inside:

/path/to/CARLA_0.9.13/PythonAPI/carla/dist/

---

Running CARLA

Windowed Mode:

cd /path/to/CARLA_0.9.13
./CarlaUE4.sh -windowed -carla-port=2000

Headless Mode (Recommended for Training):

cd /path/to/CARLA_0.9.13
SDL_VIDEODRIVER=dummy ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port=2000

Alternative headless configuration:

DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000

---

Usage

Ensure the CARLA server is running prior to launching training.

Vanilla Multi-Modal DQN:

python MM_DQN.py --render False --enable-pygame False --port 2000

SSA-Based Spiking DQN:

python SSA_DQN.py --render False --enable-pygame False --port 2000

TTSA-Based Spiking DQN:

python TTSA_DQN.py --render False --enable-pygame False --port 2000



---

TensorBoard Logging

Training logs are stored in:

runs/

Launch TensorBoard:

tensorboard --logdir runs

Typical metrics include:

* Episodic return
* Episodic length
* TD loss
* Epsilon schedule
* Training throughput

---

Saved Models

Checkpoints are saved in:

models_DQN/
models_SSA_DQN/
models_TTSA_DQN/

Example filenames:

<exp_name>__seed0__5000.pth
<exp_name>__seed0__10000.pth

---

Quick Environment Test

After setup:

python test.py

This verifies:

* CARLA connectivity
* Observation pipeline
* Action execution

---

Common Issues

1. gym_carla not found:

pip install -e gym-carla

2. CARLA import error:

echo $PYTHONPATH

3. Address already in use:

./CarlaUE4.sh -carla-port=2001
python dqn_carla.py --port 2001

4. Slow training performance:

Reduce:

* number_of_vehicles
* display_size
* render_panels

Keep:

--render False --enable-pygame False

---

Recommended Workflow

For reproducible experimentation:

1. Run Vanilla DQN to verify environment configuration
2. Run SSA-DQN
3. Run TTSA-DQN

This sequence isolates environment-level issues from model-level implementation effects.


