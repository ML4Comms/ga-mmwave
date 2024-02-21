import argparse

def parse_args():
	''' 
	Please note that every configuration is w.r.t [a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3] dists in this order.
	Regarding parameters within each distribution, we have the following order:
	1. a-u [alpha, mu, r_hat]
	2. k-u [kappa, mu, omega]
	3. shadowed k-u T1 [kappa, mu, omega, md]
	4. shadowed k-u T2 [kappa, mu, omega, ms]
	5. shadowed k-u T3 [kappa, mu, r_hat, mt]
	'''
	parser = argparse.ArgumentParser(description='ga_mmwave_fading_parameter_estimation')

	# Considered distributions settings
	parser.add_argument('--dists', type=list, default=['a-u', 'k-u', 'sk-u_t1', 'sk-u_t2'], \
	 help='Distributions to fit (available: a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
	parser.add_argument('--min_dists', type=list, \
		default=[[0.01, 0.01, 0.01], [0.001, 0.01, 0.01], [0.001, 0.01, 0.01, 0.1], [0.001, 0.01, 0.01, 1.001], [0.001, 0.01, 0.01, 1.001]], \
		help='Min est. params. bounds (for respectively: a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
			# min param. bounds [alpha, mu, r_hat], [kappa, mu, omega] , ... 
	parser.add_argument('--max_dists', type=list, \
		default=[[10.0, 3.0, 3.0], [70, 3.0, 3.0], [15, 3.0, 2.5, 5.0], [70, 3.0, 2.5, 100], [70, 3.0, 2.5, 100]], \
		help='Max est. params. bounds (for respectively: a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
			# max param. bounds [alpha, mu, r_hat], [kappa, mu, omega] , ... 
	parser.add_argument('--n_kde_samples', type=int, default=81,\
	 help='Number of Kernel Density Estimation samples')

	# DAS scenario (RXs and paths)
	parser.add_argument('--paths', type=list, default=['AB', 'BA'], help='DAS paths (AB and/or BA)')
	parser.add_argument('--n_considered_rx', type=int, default=9, help='Number of considered RXS for each path')

	# Genetic Algorithms settings
	parser.add_argument('--maxit', type=list, default=[500, 500, 500, 600, 600], \
		help='Num. of generations (a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
	parser.add_argument('--npop', type=list, default=[70, 70, 70, 70, 80], \
		help='Solutions\' population size (a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
	parser.add_argument('--pc', type=list, default=[1, 1, 1, 1, 1], \
		help='Percentage of child pop. w.r.t global pop. (a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
	parser.add_argument('--xbeta', type=list, default=[1, 1, 1, 1, 1], \
		help='Roullete wheel selection parameter (a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
	parser.add_argument('--gamma', type=list, default=[.2, .2, .2, .3, .3], \
		help='Offspring shifter (a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
	parser.add_argument('--mu', type=list, default=[.2, .2, .2, .3, .3], \
		help='Mutation rate (a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
	parser.add_argument('--sigma', type=list, default=[.2, .2, .2, .3, .3], \
		help='Mutation stepsize (a-u, k-u, sk-u_t1, sk-u_t2, sk-u_t3)')
	parser.add_argument('--tolerance', type=float, default=1e-08, \
		help='Tolerance level to early stopping')
	parser.add_argument('--ntol', type=int, default=300, \
		help='Number of epochs checking if the fitness function difference is lower than tolerance')

	# Main folder path
	parser.add_argument('--path_for_common_dir', dest='path_for_common_dir',
		default='runs/', type=str)

	# Fixed seeds settings
	parser.add_argument('--fix_seed', type=bool, default=False, \
		help='True for fixed seeds, False otherwise')
	parser.add_argument('--seed_number', type=int, default=0, \
		help='Fixed seed number')

	args = parser.parse_args()

	return args