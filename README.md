# EC_swarm
Webpage for using Evolutionary Computing techniques applied on swarm robotics.

------
#### This branch is directly related to the following paper:
### Emergent Heterogeneous Swarm Control Through Hebbian Learning 

Link to this work can be found here <a href="https://arxiv.org/pdf/2507.11566" target="_blank">https://arxiv.org/pdf/2507.11566</a>


To run the same evolutionary experiments as presented in the paper run the following line after [Installation](#installation):

```
./run-experiment.sh Hebbian_swarm_EvoExp
```

[//]: # (To retest the best controllers and re-run the validation experiments:)

[//]: # (```)

[//]: # (wget https://dataverse.nl/dataset.xhtml?persistentId=doi%3A10.34894%2F0VSN8Z#)

[//]: # (unzip ./results.zip)

[//]: # (python ./results/RetestBest.py)

[//]: # (```)


### Citation:
```
@article{van2025hebbianswarm,
  title={Emergent Heterogeneous Swarm Control Through Hebbian Learning},
  author={van Diggelen, Fuda and Karag{\"u}zel, Tugay Alperen and Rincon, Andres Garcia and Eiben, AE and Floreano, Dario and Ferrante, Eliseo},
  journal={arXiv preprint arXiv:2507.11566},
  year={2025}
}
```


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
#### This repo is directly related to the following papers:
* Van Diggelen, F., Karagüzel, T. A., Rincon, A. G., Eiben, A. E., Floreano, D., & Ferrante, E., (2025). Emergent Heterogeneous Swarm Control Through Hebbian Learning. _Nature Communications_ (15), 6534. https://doi.org/10.1145/3512290.3528735. [**[Branch]**](https://github.com/fudavd/EC_swarm/tree/Hebbian)
* van Diggelen, F., De Carlo, M., Cambier, N., Ferrante, E., & Eiben, G. (2024, September). Emergence of Specialised Collective Behaviors in Evolving Heterogeneous Swarms. In International Conference on Parallel Problem Solving from Nature (pp. 53-69). Cham: Springer Nature Switzerland. https://doi.org/10.1007/978-3-031-70068-2_4. [**[Branch]**](https://github.com/fudavd/EC_swarm/tree/PPSN_2024)
* Van Diggelen, F., Luo, J., Karagüzel, T. A., Cambier, N., Ferrante, E., & Eiben, A. E. (2022, July). Environment induced emergence of collective behavior in evolving swarms with limited sensing. In _Proceedings of the Genetic and Evolutionary Computation Conference_ (pp. 31-39). https://doi.org/10.1145/3512290.3528735. [**[Branch]**](https://github.com/fudavd/EC_swarm/tree/GECCO_2022)
---

## Acknowledgments
Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.