# EC_swarm
Pipeline for using Evolutionary Computing techniques applied on swarm robotics.

------
####This branch is directly related to the following paper:
https://arxiv.org/pdf/2203.11585.pdf


To run the same evolutionary experiments as presented in the paper run the following line after installation:

```
./run-experiment.sh EC_swarm_EvoExp
```

To retest the best controllers run:
```
./run-experiment.sh RetestBest
```

---
REQUIREMENTS
------------

This EC pipeline requires the following for the simulator [Isaac Gym](https://developer.nvidia.com/isaac-gym):
* Ubuntu 18.04 or 20.04
* CUDAnn
* Python 3.6, 3.7, 3.8

## Installation
- clone the repository
```bash
git clone https://github.com/fudavd/EC_swarm
```

- Download and extract Isaac Gym in the `/thirdparty/` folder (can be downloaded from [here](https://developer.nvidia.com/isaac-gym))
- Create a Python virtual environment in the `EC_swarm' root directory:
```bash
virtualenv -p=python3.8 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

