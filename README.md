# EC_swarm
Pipeline for using Evolutionary Computing techniques applied on swarm robotics.

---
REQUIREMENTS
------------

This EC pipeline requires the following for the simulator <a href="https://developer.nvidia.com/isaac-gym" target="_blank">Isaac Gym</a>:
* Ubuntu 18.04 or 20.04
* CUDAnn (only an installation is required, there is no need to use it)
* Python 3.8

## Installation
- clone the repository
```bash
git clone https://github.com/fudavd/EC_swarm
```

- Download and extract Isaac Gym in the `/thirdparty/` folder (can be downloaded from <a href="https://developer.nvidia.com/isaac-gym" target="_blank">here</a>)
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

[//]: # (* Van Diggelen, F., De Carlo, M., Cambier, N., Ferrante, E., & Eiben, A. E. &#40;2024, July&#41;. Environment induced emergence of collective behavior in evolving swarms with limited sensing. In _Proceedings of the Genetic and Evolutionary Computation Conference_ &#40;pp. 31-39&#41;. https://arxiv.org/abs/2402.04763. [**[Branch]**]&#40;https://github.com/fudavd/EC_swarm/tree/GECCO_2024&#41;)
* Van Diggelen, F., De Carlo, M., Cambier, N., Ferrante, E., & Eiben, A. E. (2024). Emergence of specialized Collective Behaviors in Evolving Heterogeneous Swarms. _Arxiv_: https://arxiv.org/abs/2402.04763. [**[Branch]**](https://github.com/fudavd/EC_swarm/tree/GECCO_2024)

---