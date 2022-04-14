# EC_swarm
Pipeline for using Evolutionary Computing techniques applied on swarm robotics.

------
#### This branch is directly related to the following paper:
### Environment induced emergence of collective behaviour in evolving swarms with limited sensing

Link to this work can be found here
https://doi.org/10.1145/3512290.3528735

To run the same evolutionary experiments as presented in the paper run the following line after [Installation](#installation):

```
./run-experiment.sh EC_swarm_EvoExp
```

To retest the best controllers run:
```
./run-experiment.sh RetestBest
```


### Citation:
```
@inproceedings{van2022environment,
  title     = {Environment induced emergence of collective behaviour in evolving swarms with limited sensing},
  author    = {van Diggelen, Fuda and Luo, Jie and Alperen Karag{\"u}zel, Tugay and Cambier, Nicolas and Ferrante, Eliseo and Eiben, A.E.},
  journal   = {Proceedings of the Genetic and Evolutionary Computation Conference},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3512290.3528735},
  doi       = {10.1145/3512290.3528735},
  location  = {Boston, USA},
  series    = {GECCO '22}
  year      = {2022}
}
```

Replication data can be dowloaded from here https://doi.org/10.34894/IREUW1

---
REQUIREMENTS
------------

This EC pipeline requires the following for the simulator [Isaac Gym](https://developer.nvidia.com/isaac-gym):
* Ubuntu 18.04 or 20.04
* CUDAnn (only an install is required, there is no need to use it)
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

