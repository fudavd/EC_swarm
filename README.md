# EC_swarm
Pipeline for using Evolutionary Computing techniques applied on swarm robotics

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

