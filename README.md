# Deep Bayesian Quadrature Policy Optimization

This repository is the official implementation of Deep Bayesian Quadrature Policy Gradient (DBQPG) and Uncertainty Aware Policy Gradient methods (https://arxiv.org/abs/20xx.xxxx).
![Bayesian Quadrature for Policy Gradient](/imgs/BQforPG.png)

## Project Setup

This codebase requires Python 3.6 (or higher). We recommend using Anaconda or Miniconda for setting up the virtual environment. Here, we walk through the installation and project setup.

```setup
git clone https://github.com/Akella17/Deep-Bayesian-Quadrature-Policy-Optimization.git
cd Deep-Bayesian-Quadrature-Policy-Optimization
conda create -n DBQPG python=3.6
conda activate DBQPG
pip install -r requirements.txt
```
## Supported Environments
1. [MuJoCo](http://www.mujoco.org/)
2. [PyBullet](http://pybullet.org/)
3. [Roboschool](https://github.com/openai/roboschool) [DEPRECATED]

## Training

Kindly take a look at the arguments of ```bac_main.py``` file to select the experiment that you want to run. The experiment will run for 1000 policy updates and the logs get stored in ```session_logs/``` folder. To train the model(s) in the paper, run this command:

```train
# General Command
python agent.py --env-name <gym_environment_name> --pg_algorithm <VanillaPG/NPG/TRPO> --pg_estimator <MC/BQ> --UAPG_flag
# Running Monte-Carlo baselines
python agent.py --env-name <gym_environment_name> --pg_algorithm <VanillaPG/NPG/TRPO> --pg_estimator MC
# DBQPG as the policy gradient estimator
python agent.py --env-name <gym_environment_name> --pg_algorithm <VanillaPG/NPG/TRPO> --pg_estimator BQ
# UAPG as the policy gradient estimator
python agent.py --env-name <gym_environment_name> --pg_algorithm <VanillaPG/NPG/TRPO> --pg_estimator BQ --UAPG_flag
```

## Visualization

The results get stored in ```session_logs/``` folder. These results can be visualized using Tensorboard. The IPython notebook provides additional details regarding this.

## Results

### Vanilla Policy Gradient

![Average of 10 runs.](/imgs/VanillaPG_plot.png)

### Natural Policy Gradient

![Average of 10 runs.](/imgs/NPG_plot.png)

### Trust Region Policy Optimization

![Average of 10 runs.](/imgs/TRPO_plot.png)

## Acknowledgements

I would like to thank the numerous open-source projects that were instrumental in the development of this project. Namely, our codebase is built on top of [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo), a popular open-source implementation of TRPO. For scaling-up Bayesian quadrature and for GPU acceleration to kernel learning, we heavily rely on GPyTorch library. For fast low-rank svd using just implicit matrix-vector multiplications, I refactored Facebook's [fbpca](https://research.fb.com/blog/2014/09/fast-randomized-svd/) code in PyTorch. Lastly, I would like to thank Jamie Townsend for ["A new trick for calculating Jacobian vector products"](https://j-towns.github.io/2017/06/12/A-new-trick.html) (Appendix D in our paper).

## Contributing

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also see a todo list below.