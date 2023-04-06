# EC_swarm
Pipeline for using Evolutionary Computing techniques applied on swarm robotics.

---
REQUIREMENTS
------------

This EC pipeline requires the following for the simulator [Isaac Gym](https://developer.nvidia.com/isaac-gym):
* Ubuntu 18.04 or 20.04
* CUDAnn (only an installation is required, there is no need to use it)
* Python 3.6, 3.7, 3.8

## Installation
- clone the repository
```bash
git clone https://github.com/fudavd/EC_swarm
```

- Download and extract Isaac Gym in the `/thirdparty/` folder (can be downloaded from [here](https://developer.nvidia.com/isaac-gym))
- Create a Python virtual environment in the `EC_swarm` root directory:
```bash
virtualenv -p=python3.8 .venv
source .venv/bin/activate
pip install -r requirements.txt
```
---
Publications
------
#### This repo is directly related to the following paper:
* Van Diggelen, F., Luo, J., Karagüzel, T. A., Cambier, N., Ferrante, E., & Eiben, A. E. (2022, July). Environment induced emergence of collective behavior in evolving swarms with limited sensing. In _Proceedings of the Genetic and Evolutionary Computation Conference_ (pp. 31-39). https://doi.org/10.1145/3512290.3528735. [**[Branch]**](https://github.com/fudavd/EC_swarm/tree/GECCO_2022)

---