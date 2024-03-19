# EC_swarm
Pipeline for using Evolutionary Computing techniques applied on swarm robotics.

------
#### This branch is directly related to the following paper:
### Emergence of specialized Collective Behaviors in Evolving Heterogeneous Swarms

Link to this work can be found here
https://arxiv.org/abs/2402.04763

To run the Evolutionary experiment for the _Baseline_ as presented in the paper run the following line after [Installation](#installation):

```
./run-experiment.sh Baseline_swarm_EvoExp
```
To run the Evolutionary experiment for the _Heterogeneous controller_
```
./run-experiment.sh Hetero_swarm_EvoExp
```

To retest the best controllers and re-run the validation experiments:
```
wget https://dataverse.nl/dataset.xhtml?persistentId=doi%3A10.34894%2FBSZNMT#
unzip ./results.zip
python ./results/RetestBest.py
```


### Citation:
```
@article{van2024emergence,
  title={ Learning online adaptive emergent control for heterogeneous robot swarms},
  author={van Diggelen, Fuda and Alperen Karagüzel, Tugay and Rincón, Andrés García and Ferrante, Eliseo},
  year={2024}
}
```

Replication data can be downloaded from here https://doi.org/10.34894/2FBSZNMT

---
REQUIREMENTS
------------

This EC pipeline requires the following for the simulator <a href="https://developer.nvidia.com/isaac-gym" target="_blank">Isaac Gym</a>:
* Ubuntu 18.04 or 20.04
* CUDAnn (only an installation is required, there is no need to use it)
* Python 3.8

## Installation
- Clone the repository in the CrazyThymio branch 
```bash
git clone -b CrazyThymio https://github.com/fudavd/EC_swarm
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
#### This repo is directly related to the following papers:
* Van Diggelen, F., Luo, J., Karagüzel, T. A., Cambier, N., Ferrante, E., & Eiben, A. E. (2022, July). Environment induced emergence of collective behavior in evolving swarms with limited sensing. In _Proceedings of the Genetic and Evolutionary Computation Conference_ (pp. 31-39). https://doi.org/10.1145/3512290.3528735. [**[Branch]**](https://github.com/fudavd/EC_swarm/tree/GECCO_2022)
* Van Diggelen, F., De Carlo, M., Cambier, N., Ferrante, E., & Eiben, A. E. (2024). Emergence of specialized Collective Behaviors in Evolving Heterogeneous Swarms. _Arxiv_: https://arxiv.org/abs/2402.04763. [**[Branch]**](https://github.com/fudavd/EC_swarm/tree/GECCO_2024)
---
