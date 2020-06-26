import os
import json
import argparse
import numpy as np
from tensorboardX import SummaryWriter

def get_args():
	parser = argparse.ArgumentParser(description='PyTorch Policy Gradient (Bayesian Quadrature/Monte Carlo)')
	#--------------------------------------------------------------------------------------------------------------------------------------------------------
	# General arguments
	parser.add_argument('--env-name',
						default="Swimmer-v2", metavar='G',
	                    help='Name of the gym environment to run')
	parser.add_argument('--gamma', type=float,
						default=0.995, metavar='G',
	                    help='discount factor (default: 0.995)')
	parser.add_argument('--batch-size', type=int,
						default=15000, metavar='N',
	                    help='state-action sample size (default: 15000)')
	parser.add_argument('--pg_algorithm',
						default="VanillaPG",
						help='TRPO | VanillaPG | NPG. Selecting the policy optimization technique')
	parser.add_argument('--render',
						action='store_true',
	                    help='renders the policy roll-out in the environment')
	parser.add_argument('--output_directory',
						default="session_logs/", metavar='G',
	                    help='writes the session logs to this directory')
	parser.add_argument('--gpu_id',
						default="0", metavar='G',
						help='Mention the target GPU for deployment. Our GP kernel learning does not support multi-gpu training.')
	parser.add_argument('--seed', type=int,
						default=-1, metavar='N',
	                    help='random seed (default: 1). Useful for debugging.')
	#--------------------------------------------------------------------------------------------------------------------------------------------------------
	# GAE arguments
	parser.add_argument("--advantage_flag", action='store_true',
	                    help="Replaces Monte-Carlo/TD(1) action-value estimates with generalized advantage estimates (GAE)")
	parser.add_argument('--tau', type=float,
						default=0.97, metavar='G',
	                    help='GAE exponentially-weighted average coefficient (default: 0.97)')
	#--------------------------------------------------------------------------------------------------------------------------------------------------------
	# LR for VanillaPG and NPG
	parser.add_argument('--lr', type=float,
						default=7e-4, metavar='G',
	                    help='learning rate (default: 1e-1)')
	#--------------------------------------------------------------------------------------------------------------------------------------------------------
	# TRPO arguments
	parser.add_argument('--max-kl', type=float,
						default=1e-2, metavar='G',
	                    help='Trust region size, i.e., the max allowed KL divergence between the old and updated policy (default: 1e-2)')
	parser.add_argument('--damping', type=float,
						default=1e-1, metavar='G',
	                    help='Damping coefficient. For numerical stablility and quick convergence of Fisher inverse computation using Conjugate Gradient.')
	#--------------------------------------------------------------------------------------------------------------------------------------------------------
	# Policy gradient estimator arguments
	parser.add_argument('--pg_estimator',
						default="BQ", metavar='G',
	                    help='BQ: Bayesian Quadrature | MC: Monte Carlo - Selects the PG estimator.')
	if parser.parse_known_args()[0].pg_estimator == 'BQ':
		parser.add_argument('--svd_low_rank', type=int,
							default=-1, metavar='N',
		                    help='specified the (low) rank for approximating the U and Cov matrices with FastSVD')
		parser.add_argument('--fisher_coefficient', type = float,
							default=5e-5, metavar='G',
		                    help="The coefficient of Fisher kernel, i.e. c_2, in the PG estimate U(c_1 K_s + c_2 K_f + sigma^2 I)^{-1} A^{GAE}")
		parser.add_argument('--state_coefficient', type = float,
							default=1, metavar='G',
		                    help="The coefficient of State kernel, i.e. c_1, in the PG estimate U(c_1 K_s + c_2 K_f + sigma^2 I)^{-1} A^{GAE}")
		parser.add_argument('--likelihood_noise_level', type = float,
							default=1e-4, metavar='G',
		                    help='GPs noise variance sigma^2')
		parser.add_argument("--UAPG_flag", action='store_true',
							help="If true then the gradient covariance is used for computing UAPG updates")
		parser.add_argument('--UAPG_epsilon', type=float,
							default=3.0, metavar='G',
		                    help='Maximum factor by which a DBQPG compoment stepsize is increased during the UAPG update (for NPG or TRPO)')
	
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
	if args.seed == -1:
		args.seed = int(np.random.randint(low = 0, high = 100000000, size = 1)[0])
	#--------------------------------------------------------------------------------------------------------------------------------------------------------
	# Hyperparameter Helper for few MuJoCo environments (feel free to comment this section for a manual overide)
	with open('helper_config.json') as config:
		config = json.load(config)
		if args.env_name in config:
			args.advantage_flag = config[args.env_name]["advantage_flag"]
			args.svd_low_rank = config[args.env_name][args.pg_algorithm]["svd_low_rank"]
			if args.pg_algorithm != 'TRPO':
				args.lr = config[args.env_name][args.pg_algorithm]["lr"]
			if args.pg_estimator == 'MC' and 'MC_lr' in config[args.env_name][args.pg_algorithm]:
				args.lr = config[args.env_name][args.pg_algorithm]["MC_lr"]
	#--------------------------------------------------------------------------------------------------------------------------------------------------------
	# Logs the cumulative reward statistics over episodes
	prefix = args.pg_estimator + "_" + args.pg_algorithm
	write_filename = prefix + "_" + args.env_name +"_GAE("+str(args.advantage_flag)+ ")_LR("+str(args.lr)
	if args.pg_estimator == "BQ":
		write_filename = write_filename +")_SVDrank("+str(args.svd_low_rank)+")_UAPG("+str(args.UAPG_flag)+")_UAPGeps(" + str(args.UAPG_epsilon)
	write_filename = write_filename + ")_batchsize("+str(args.batch_size)+")_seed("+str(args.seed)+ ")"

	if not os.path.exists(args.output_directory + write_filename):
	    os.makedirs(args.output_directory + write_filename)
	summa_writer = SummaryWriter(logdir = args.output_directory + write_filename, comment= args.pg_estimator + "-PG")
	#--------------------------------------------------------------------------------------------------------------------------------------------------------
	
	return args, summa_writer