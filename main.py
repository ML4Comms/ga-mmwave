#!/usr/bin/env python
# Created by Samuel Gomes 06/06/2023 ---------------%
#       Email: samuelbfgomes@gmail.com              %
#       Github: https://github.com/SamuelBFG        %
# --------------------------------------------------%

import datetime
import warnings
import models.ga as ga
from ypstruct import structure
from arguments import parse_args
from utils.data_load import load_datasets
from utils.funcs import tic, toc, get_kde, compute_AIC, get_nls, get_mle, fitness, plot_results


warnings.filterwarnings("ignore", category=RuntimeWarning) 

if __name__ == '__main__':
	args = parse_args()
	print('Called with args:')
	print(args)

	curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	common_dir = './' + args.path_for_common_dir + curr_time + '/'

	for dist in args.dists:
		# Problem Definition
		if dist == 'a-u':
			mode = 'small_scale_fading'
			problem = structure()
			problem.costfunc = fitness
			problem.dist = dist
			problem.nvar = len(args.min_dists[0]) 
			problem.varmin = args.min_dists[0]
			problem.varmax = args.max_dists[0]
			problem.tol = args.tolerance
			problem.ntol = args.ntol
			params = structure()
			params.maxit = args.maxit[0]
			params.npop = args.npop[0]
			params.pc = args.pc[0]
			params.beta = args.xbeta[0] 
			params.gamma = args.gamma[0]
			params.mu = args.mu[0]
			params.sigma = args.sigma[0]

		elif dist == 'k-u':
			mode = 'small_scale_fading'
			problem = structure()
			problem.costfunc = fitness
			problem.dist = dist 
			problem.nvar = len(args.min_dists[1]) 
			problem.varmin = args.min_dists[1]
			problem.varmax = args.max_dists[1]
			problem.tol = args.tolerance
			problem.ntol = args.ntol
			params = structure()
			params.maxit = args.maxit[1]
			params.npop = args.npop[1]
			params.pc = args.pc[1]
			params.beta = args.xbeta[1] 
			params.gamma = args.gamma[1]
			params.mu = args.mu[1]
			params.sigma = args.sigma[1]

		elif dist == 'sk-u_t1':
			mode = 'composite_fading'
			problem = structure()
			problem.costfunc = fitness
			problem.dist = dist 
			problem.nvar = len(args.min_dists[2]) 
			problem.varmin = args.min_dists[2]
			problem.varmax = args.max_dists[2]
			problem.tol = args.tolerance
			problem.ntol = args.ntol
			params = structure()
			params.maxit = args.maxit[2]
			params.npop = args.npop[2]
			params.pc = args.pc[2]
			params.beta = args.xbeta[2] 
			params.gamma = args.gamma[2]
			params.mu = args.mu[2]
			params.sigma = args.sigma[2]

		elif dist == 'sk-u_t2':
			mode = 'composite_fading'
			problem = structure()
			problem.costfunc = fitness
			problem.dist = dist 
			problem.nvar = len(args.min_dists[3]) 
			problem.varmin = args.min_dists[3]
			problem.varmax = args.max_dists[3]
			problem.tol = args.tolerance
			problem.ntol = args.ntol
			params = structure()
			params.maxit = args.maxit[3]
			params.npop = args.npop[3]
			params.pc = args.pc[3]
			params.beta = args.xbeta[3] 
			params.gamma = args.gamma[3]
			params.mu = args.mu[3]
			params.sigma = args.sigma[3]

		elif dist == 'sk-u_t3':
			mode = 'composite_fading'
			problem = structure()
			problem.costfunc = fitness
			problem.dist = dist 
			problem.nvar = len(args.min_dists[4]) 
			problem.varmin = args.min_dists[4]
			problem.varmax = args.max_dists[4]
			problem.tol = args.tolerance
			problem.ntol = args.ntol
			params = structure()
			params.maxit = args.maxit[4]
			params.npop = args.npop[4]
			params.pc = args.pc[4]
			params.beta = args.xbeta[4] 
			params.gamma = args.gamma[4]
			params.mu = args.mu[4]
			params.sigma = args.sigma[4]

		nAPs = ['AP{}'.format(x) for x in range(1, args.n_considered_rx + 1)]

		for path in args.paths:
			
			datasets = {}
			datasets = load_datasets(mode)

			print('DISTRIBUTION: ', dist)
			for nAP in nAPs:

				data = datasets[path][nAP]
				n = len(data)
				Xdata, Ydata = get_kde(data.dropna(), args.n_kde_samples, path, nAP) # Get histogram
				
				# Run GA - MSE
				problem.fitness_type = 'MSE'
				tic()
				out_mse = ga.run(problem, dist, params, Xdata, Ydata)
				toc()
				print(out_mse.bestsol)
				
				# Run GA - RAD
				problem.fitness_type = 'RAD'
				tic()
				out_rad = ga.run(problem, dist, params, Xdata, Ydata)
				toc()
				print(out_rad.bestsol)

				# Run NLS
				tic()
				initial_conditions = get_mle(data, dist, problem.varmin)
				nls_params = get_nls(initial_conditions, Xdata, Ydata, dist, problem.varmin, problem.varmax).x
				toc()


				print('NLS Est. Params:', nls_params)
				compute_AIC(data, dist, n, problem.nvar, nls_params, out_mse.bestsol.position, out_rad.bestsol.position)
				plot_results(dist, Xdata, Ydata, nls_params, out_mse.bestsol.position, out_rad.bestsol.position, path, nAP, common_dir)
