Deep Bayesian Quadrature Policy Optimization
============================================

<h4>
Akella Ravi Tej<sup>1</sup>, Kamyar Azizzadenesheli<sup>3</sup>, Mohammad Ghavamzadeh<sup>2</sup>, Anima Anandkumar<sup>3</sup>, Yisong Yue<sup>3</sup>
</br>
<span style="font-size: 14pt; color: #555555">
<sup>1</sup>Indian Institute of Technology Roorkee, <sup>2</sup>Google Research, <sup>3</sup>Caltech
</span>
</h4>
<hr>

**Preprint:** [arxiv.org/abs/2006.15637](https://arxiv.org/abs/2006.15637)

![Bayesian Quadrature for Policy Gradient](/imgs/BQforPG.png)

**Bayesian quadrature** is an approach from probabilistic numerics for approximating a numerical integration. When estimating the policy gradient integral, replacing standard Monte-Carlo estimation with Bayesian quadrature provides
1. more accurate gradient estimates with a significantly lower variance
2. a consistent improvement in the sample complexity and average return for several policy gradient algorithms
3. a methodological way to quantify the uncertainty in gradient estimation.

This repository provides a computationally efficient high-dimensional generalization implementation of BQ for estimating the **policy gradient integral** (gradient vector) and the **estimation uncertainty** (gradient covariance matrix). Provides a **modular implementation** of several policy gradient methods, currently supporting three policy gradient estimators and three policy gradient algorithms (9 combinations in total):

**Policy gradient estimators** :-
1. *Monte-Carlo*
2. *deep Bayesian quadrature policy gradient (DBQPG)*
3. *Uncertainty Aware Policy Gradient (UAPG)*

**Policy gradient algorithms** :-
1. *Vanilla policy gradient*
2. *natural policy gradient (NPG)*
3. *trust-region policy optimization (TRPO) algorithms*

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

1. [MuJoCo](http://www.mujoco.org/)
2. [PyBullet](http://pybullet.org/)
3. [Roboschool](https://github.com/openai/roboschool) [DEPRECATED]

Training
--------

Modular implementation:
```train
python agent.py --env-name <gym_environment_name> --pg_algorithm <VanillaPG/NPG/TRPO> --pg_estimator <MC/BQ> --UAPG_flag
```
All the experiments will run for 1000 policy updates and the logs get stored in ```session_logs/``` folder. Kindly take a look at the ```arguments.py``` file to select the experiment that you want to run. To reproduce the results in the paper, refer the following command:
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

The results get stored in ```session_logs/``` folder. These results can be visualized using Tensorboard. The IPython notebook provides additional details regarding this.

Results
-------

### Vanilla Policy Gradient

![Average of 10 runs.](/imgs/VanillaPG_plot.png)

### Natural Policy Gradient

![Average of 10 runs.](/imgs/NPG_plot.png)

### Trust Region Policy Optimization

![Average of 10 runs.](/imgs/TRPO_plot.png)

Acknowledgements
----------------

I would like to thank the numerous open-source projects that were instrumental in the development of this project. To name a few,
- Our codebase is built on top of [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo), a popular open-source implementation of TRPO
- For scaling-up Bayesian quadrature and for GPU acceleration to kernel learning, we heavily rely on [GPyTorch library](https://gpytorch.ai/).
- For fast low-rank svd using just implicit matrix-vector multiplications, I refactored Facebook's [fbpca](https://research.fb.com/blog/2014/09/fast-randomized-svd/) code in PyTorch.
- For fast *Jvp* computation, ["A new trick for calculating Jacobian vector products"](https://j-towns.github.io/2017/06/12/A-new-trick.html) (Appendix D in our paper).

Contributing
------------

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also see a todo list below.

TODO
----

- Add other policy gradient algorithms

Citation
--------

If you find this code useful, please consider citing:

```text
@article{ravi2020DBQPG,
    title={Deep Bayesian Quadrature Policy Optimization},
    author={Akella Ravi Tej and Kamyar Azizzadenesheli and Mohammad Ghavamzadeh and Anima Anandkumar and Yisong Yue},
    journal={arXiv preprint arXiv:2006.15637},
    year={2020}
}
```
