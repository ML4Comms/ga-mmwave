#!/usr/bin/env python
# Created by Samuel Gomes 06/06/2023 ---------------%
#       Email: samuelbfgomes@gmail.com              %
#       Github: https://github.com/SamuelBFG        %
# --------------------------------------------------%

import pdb
import math
import os, sys
import numpy as np
from ypstruct import structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import parse_args

args = parse_args()
if args.fix_seed:
	np.random.seed(args.seed_number)

def run(problem, dist, params, X, Y):

	# Problem information
	costfunc = problem.costfunc
	fitness_type = problem.fitness_type
	dist = problem.dist
	nvar = problem.nvar
	varmin = problem.varmin
	varmax = problem.varmax
	tol = problem.tol
	ntol = problem.ntol

	# Parameters 
	maxit = params.maxit
	npop = params.npop
	beta = params.beta
	pc = params.pc
	nc = int(np.round(pc*npop/2)*2)
	gamma = params.gamma
	mu = params.mu
	sigma = params.sigma

	# Empty Individual Template
	empty_individual = structure()
	empty_individual.position = None
	empty_individual.cost = None

	# Best Solution ever found
	bestsol = empty_individual.deepcopy()
	bestsol.cost = np.inf

	# Initialize Population
	pop = empty_individual.repeat(npop)
	for i in range(npop):
		pop[i].position = np.random.uniform(varmin, varmax, nvar)
		pop[i].cost = costfunc(pop[i].position, dist, fitness_type, X, Y)
		if math.isnan(pop[i].cost):
			pop[i].cost = np.inf
		if pop[i].cost < bestsol.cost:
			bestsol = pop[i].deepcopy()
		
	# Best Cost of Iterations
	bestcost = np.full(maxit, float('inf'))

	counter_tol = 0

	# Main Loop
	for it in range(maxit):

		costs = np.array([x.cost for x in pop])
		avg_cost = np.mean(costs)
		if avg_cost != 0:
			costs = costs/avg_cost
		probs = np.exp(-beta*costs)

		popc = []
		for _ in range(nc//2):

			# Random Selection of Parents
			# q = np.random.permutation(npop)
			# p1 = pop[q[0]]
			# p2 = pop[q[1]]

			# Perform Roulette Wheel Selection
			p1 = pop[roulette_wheel_selection(probs)]
			p2 = pop[roulette_wheel_selection(probs)]

			# # Perform Stochastic Universal Sampling
			# i1, i2 = stochastic_universal_sampling(probs)
			# p1 = pop[i1]
			# p2 = pop[i2]

			# Perform Crossover
			c1, c2 = crossover(p1, p2, gamma)

			# Perform Mutation
			c1 = mutate(c1, mu, sigma)
			c2 = mutate(c2, mu, sigma)

			# Apply Bounds
			apply_bounds(c1, varmin, varmax)
			apply_bounds(c2, varmin, varmax)

			# Evaluate First Offspring
			c1.cost = costfunc(c1.position, dist, fitness_type, X, Y)
			if c1.cost < bestsol.cost:
				bestsol = c1.deepcopy()

			# Evaluate Second Offspring
			c2.cost = costfunc(c2.position, dist, fitness_type, X, Y)
			if c2.cost < bestsol.cost:
				bestsol = c2.deepcopy()

			# Add Offspring to Population
			popc.append(c1)
			popc.append(c2)

		# Merge, Sort and Select
		pop += popc
		pop = sorted(pop, key=lambda x:x.cost)
		pop = pop[:npop]

		# Store Best Cost
		bestcost[it] = bestsol.cost

		# # Show Iteration Information
		# print('Iteration {}: Best Cost = {}'.format(it, bestcost[it]))
		# pdb.set_trace()
		if it > 2 and abs(bestcost[it-1] - bestcost[it]) < tol:
			counter_tol += 1
			if counter_tol > ntol:
				bestcost = bestcost[:it]
				break
		else:
			counter_tol = 0

	# Output
	out = structure()
	out.pop = pop
	out.bestsol = bestsol
	out.bestcost = bestcost
	return out

def crossover(p1, p2, gamma):
	c1 = p1.deepcopy()
	c2 = p1.deepcopy()
	alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape)
	c1.position = alpha*p1.position + (1-alpha)*p2.position
	c2.position = alpha*p2.position + (1-alpha)*p1.position
	return c1, c2

def mutate(x, mu, sigma):
	y = x.deepcopy()
	flag = np.random.rand(*x.position.shape) <= mu
	ind = np.argwhere(flag)
	# y.position[ind] = x.position[ind] + sigma*np.random.randn(*ind.shape)
	y.position[ind] += sigma*np.random.randn(*ind.shape)
	return y

def apply_bounds(x, varmin, varmax):
	x.position = np.maximum(x.position, varmin)
	x.position = np.minimum(x.position, varmax)

def roulette_wheel_selection(p):
	c = np.cumsum(p)
	r = sum(p)*np.random.rand()
	ind = np.argwhere(r <= c)
	if len(ind) == 0:
		return 0
	else:
		return ind[0][0]

def stochastic_universal_sampling(p):
	lambd = 2 
	c = np.cumsum(p)
	r = sum(p)*np.random.uniform(0,1/lambd)
	r2 = r + sum(p)/lambd
	ind = np.argwhere(r <= c)
	ind2 = np.argwhere(r2 <= c)
	if len(ind) == 0:
		indx1 = 0
	else:
		indx1 = ind[0][0]

	if len(ind2) == 0:
		indx2 = 0
	else:
		indx2 = ind[0][0]

	return indx1, indx2