## DBQPG
Scalable Bayesian Quadrature framework.

Our code is uses files from the popular public implementation of TRPO [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)

```bac_main.py```: DBQPG algorithm. Uses GAE to estimate the advantage and the a GP function approximation for estimating policy gradient. Uses two critics (i), NN-based critic for estimating the advantage and (ii) GP-based critic (Deep RBF kernel + Fisher kernel).

```models.py```: Defines the actor and NN-based critic network definitions. Used with ```ac_main.py```.

```fbpca_mod.py```: A tensor-based randomized SVD implementation, adapted from [fbpca repository](https://github.com/facebook/fbpca).

For running Monte-Carlo algorithm baselines, use [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)

Kindly take a look at the arguments of ```bac_main.py``` file to select the experiment that you want to run. The experiment will run for 1000 policy updates and the logs get stored in ```session_logs/``` folder.

### Requirements
1) GPU hardware (for fast kernel learning and effective use of GPyTorch)
2) PyTorch (1.1.0)
3) GPyTorch library (0.3.5)
4) OpenAI Gym (0.15.4)
5) MuJoCo physics engine
6) mujoco-py (2.0.2.9)
7) TensorboardX (1.9)
8) Numpy (1.18.1)
9) Scipy (1.4.1)