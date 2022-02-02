# Few-Task Meta Learning.

This repository contains the code used for task selectin for "Few-Task Meta Learning". The script shown is a generic template to work for any domain defined following the standard Open AI gym template. For training the individual tasks and the meta-RL agents, as well as testing the agents [garage](https://github.com/rlworkgroup/garage) was used.

### Dependencies
Our code requires:
* python 3.*
* numpy
* tensorflow v1.7+
* scipy
* garage

### Domains
For the domains used in the experiments please take a look at:

[Ant](https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/ant_env_rand.py)

[Half-Cheetah](https://github.com/rlworkgroup/garage/blob/93d1d6f0d546b544ab52bc399cacad3f0c696849/src/garage/envs/mujoco/half_cheetah_vel_env.py)

[KrazyWorld](https://github.com/bstadie/krazyworld)

[MiniGrid](https://github.com/maximecb/gym-minigrid)

