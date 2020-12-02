Deep Bayesian Quadrature Policy Optimization
============================================

<h4>
Akella Ravi Tej<sup>1</sup>, Kamyar Azizzadenesheli<sup>2</sup>, Mohammad Ghavamzadeh<sup>3</sup>, Anima Anandkumar<sup>4</sup>, Yisong Yue<sup>4</sup>
</br>
<span style="font-size: 14pt; color: #555555">
<sup>1</sup>Indian Institute of Technology Roorkee, <sup>2</sup>Purdue University, <sup>3</sup>Google Research, <sup>4</sup>Caltech
</span>
</h4>
<hr>

**Preprint:** [arxiv.org/abs/2006.15637](https://arxiv.org/abs/2006.15637)<br>
**Publication:** [AAAI-21](https://aaai.org/Conferences/AAAI-21/), [NeurIPS Deep RL Workshop 2020](https://sites.google.com/view/deep-rl-workshop-neurips2020/home), [NeurIPS Real-World RL Workshop 2020](https://sites.google.com/view/neurips2020rwrl)<br>
**Project Website:** [akella17.github.io/publications/Deep-Bayesian-Quadrature-Policy-Optimization/](https://akella17.github.io/publications/Deep-Bayesian-Quadrature-Policy-Optimization/)

![Bayesian Quadrature for Policy Gradient](/imgs/BQforPG.png)

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Akella17/Deep-Bayesian-Quadrature-Policy-Optimization/blob/master/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)

**Bayesian quadrature** is an approach in probabilistic numerics for approximating a numerical integration. When estimating the policy gradient integral, replacing standard Monte-Carlo estimation with Bayesian quadrature provides
1. more accurate gradient estimates with a significantly lower variance
2. a consistent improvement in the sample complexity and average return for several policy gradient algorithms
3. a methodological way to quantify the uncertainty in gradient estimation.

This repository contains a computationally efficient implementation of BQ for estimating the **policy gradient integral** (gradient vector) and the **estimation uncertainty** (gradient covariance matrix). The source code is written in a **modular fashion**, currently supporting three policy gradient estimators and three policy gradient algorithms (9 combinations overall):

**Policy Gradient Estimators** :-
1. *Monte-Carlo Estimation*
2. *Deep Bayesian Quadrature Policy Gradient (DBQPG)*
3. *Uncertainty Aware Policy Gradient (UAPG)*
<!-- | (i) **Monte-Carlo Estimation** | (ii) **Deep Bayesian Quadrature Policy Gradient (DBQPG)** | (iii) **Uncertainty Aware Policy Gradient (UAPG)** |
| --------------------------- |:-----------|:---------------------------------------:| -->


**Policy Gradient Algorithms** :-
1. *Vanilla Policy Gradient*
2. *Natural Policy Gradient (NPG)*
3. *Trust-Region Policy Optimization (TRPO)*
<!-- | (i) **Vanilla Policy Gradient** | (ii) **Natural Policy Gradient (NPG)** | (iii) **Trust-Region Policy Optimization (TRPO)** |
| --------------------------- |:-----------|:---------------------------------------:| -->

Project Setup
-------------

This codebase requires Python 3.6 (or higher). We recommend using Anaconda or Miniconda for setting up the virtual environment. Here's a walk through for the installation and project setup.

```setup
git clone https://github.com/Akella17/Deep-Bayesian-Quadrature-Policy-Optimization.git
cd Deep-Bayesian-Quadrature-Policy-Optimization
conda create -n DBQPG python=3.6
conda activate DBQPG
pip install -r requirements.txt
```
Supported Environments
----------------------

1. [Classic Control](https://gym.openai.com/envs/#classic_control) 
2. [MuJoCo](http://www.mujoco.org/)
3. [PyBullet](http://pybullet.org/)
4. [Roboschool](https://github.com/openai/roboschool)
5. [DeepMind Control Suite](https://github.com/deepmind/dm_control) (via [dm_control2gym](https://github.com/martinseilair/dm_control2gym))

Training
--------

Modular implementation:
```train
python agent.py --env-name <gym_environment_name> --pg_algorithm <VanillaPG/NPG/TRPO> --pg_estimator <MC/BQ> --UAPG_flag
```
All the experiments will run for 1000 policy updates and the **logs** get stored in ```session_logs/``` folder. To reproduce the results in the paper, refer the following command:
```train
# Running Monte-Carlo baselines
python agent.py --env-name <gym_environment_name> --pg_algorithm <VanillaPG/NPG/TRPO> --pg_estimator MC
# DBQPG as the policy gradient estimator
python agent.py --env-name <gym_environment_name> --pg_algorithm <VanillaPG/NPG/TRPO> --pg_estimator BQ
# UAPG as the policy gradient estimator
python agent.py --env-name <gym_environment_name> --pg_algorithm <VanillaPG/NPG/TRPO> --pg_estimator BQ --UAPG_flag
```
For more customization options, kindly take a look at the ```arguments.py```.

Visualization
-------------

```visualize.ipynb``` can be used to visualize the Tensorboard files stored in ```session_logs/``` (requires ```jupyter``` and ```tensorboard``` installed).

Results
-------

### Vanilla Policy Gradient

![Average of 10 runs.](/imgs/VanillaPG_plot.png)

### Natural Policy Gradient

![Average of 10 runs.](/imgs/NPG_plot.png)

### Trust Region Policy Optimization

![Average of 10 runs.](/imgs/TRPO_plot.png)

Implementation References
-------------------------
- [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)
	- TRPO and NPG implementation.
- [GPyTorch library](https://gpytorch.ai/)
	- Structured kernel interpolation (SKI) with Toeplitz method for RBF kernel. 
	- Kernel learning with GPU acceleration.
- [fbpca](https://research.fb.com/blog/2014/09/fast-randomized-svd/)
	- Fast randomized singular value decomposition (SVD) through implicit matrix-vector multiplications.
- ["A new trick for calculating Jacobian vector products"](https://j-towns.github.io/2017/06/12/A-new-trick.html)
	- Efficient *Jvp* computation through regular reverse-mode autodiff (more details in Appendix D of [our paper](https://arxiv.org/abs/2006.15637)).

Contributing
------------

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also see the todo list below.

TODO
----

- Implement policy network for discrete action space and test on Arcade Learning Environment (ALE).
- Add other policy gradient algorithms.

Citation
--------

If you find this work useful, please consider citing:

```text
@article{ravi2020DBQPG,
    title={Deep Bayesian Quadrature Policy Optimization},
    author={Akella Ravi Tej and Kamyar Azizzadenesheli and Mohammad Ghavamzadeh and Anima Anandkumar and Yisong Yue},
    journal={arXiv preprint arXiv:2006.15637},
    year={2020}
}
```
