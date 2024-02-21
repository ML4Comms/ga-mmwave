from sklearn.neighbors import KernelDensity
from utils.distributions import alpha_mu_pdf, kappa_mu_pdf, shad_ku_T1_pdf, shad_ku_T2_pdf, shad_ku_T3_pdf
from scipy.stats import nakagami, rice
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pdb

def tic():
	'''
	Homemade version of matlab tic and toc functions.
	'''
	import time
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()

def toc():
	import time
	if 'startTime_for_tictoc' in globals():
	    print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
	else:
	    print("Toc: start time not set") 

def get_kde(linear_data, nExtractedSamples, path, nAP):
	'''
	Extracted KDE uniformly spaced samples from the data.
	'''
	print('Path: {} | AP: {}'.format(path, nAP))
	Xdata = np.linspace(linear_data.min(), linear_data.max(), nExtractedSamples).reshape(-1, 1)
	kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(linear_data.values.reshape(-1, 1))

	logkde = kde.score_samples(Xdata)
	Ydata = np.exp(logkde).reshape(-1,1)
	# plt.figure(figsize=(10,7))
	# plt.scatter(20*np.log10(Xdata), Ydata, c='magenta')
	# plt.title(f'Path {path} | RX: {nAP}')
	# plt.ylabel('Density')
	# plt.xlabel('Small Scale Fading (dB)')
	# plt.grid()
	# plt.show()
	return Xdata, Ydata

def kl_div(pVec1, pVec2):
	'''
	Kullback-Leibler divergence.
	'''
	KL = np.sum(pVec1*(np.log2(pVec1)-np.log2(pVec2)))
	return KL

def res_avg_distance(kldiv1, kldiv2):
	'''
	Resistor-Average Distance.
	'''
	RAD = 1/(1/kldiv1+1/kldiv2)
	return RAD

def compute_AIC(data, dist, n, K, nls_au, ga_mse, ga_rad):
	'''
	Compute Akaike Information Criteria of the estimates for GA-MSE and GA-RAD.
	'''
	if dist == 'a-u':	
	    # AIC NLS
	    sum = np.sum(np.log(alpha_mu_pdf(data, *nls_au).astype(float)))
	    AIC_nls = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: MSE
	    sum = np.sum(np.log(alpha_mu_pdf(data, *ga_mse)))
	    AIC_ga_mse = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: RAD
	    sum = np.sum(np.log(alpha_mu_pdf(data, *ga_rad)))
	    AIC_ga_rad = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	elif dist == 'k-u':
	    # AIC NLS
	    sum = np.sum(np.log(kappa_mu_pdf(data, *nls_au).astype(float)))
	    AIC_nls = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: MSE
	    sum = np.sum(np.log(kappa_mu_pdf(data, *ga_mse)))
	    AIC_ga_mse = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: RAD
	    sum = np.sum(np.log(kappa_mu_pdf(data, *ga_rad)))
	    AIC_ga_rad = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	elif dist == 'sk-u_t1':
	    # AIC NLS
	    sum = np.sum(np.log(shad_ku_T1_pdf(data, *nls_au).astype(float)))
	    AIC_nls = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: MSE
	    sum = np.sum(np.log(shad_ku_T1_pdf(data, *ga_mse)))
	    AIC_ga_mse = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: RAD
	    sum = np.sum(np.log(shad_ku_T1_pdf(data, *ga_rad)))
	    AIC_ga_rad = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	elif dist == 'sk-u_t2':
	    # AIC NLS
	    sum = np.sum(np.log(shad_ku_T2_pdf(data, *nls_au).astype(float)))
	    AIC_nls = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: MSE
	    sum = np.sum(np.log(shad_ku_T2_pdf(data, *ga_mse)))
	    AIC_ga_mse = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: RAD
	    sum = np.sum(np.log(shad_ku_T2_pdf(data, *ga_rad)))
	    AIC_ga_rad = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	elif dist == 'sk-u_t3':
	    # AIC NLS
	    sum = np.sum(np.log(shad_ku_T3_pdf(data, *nls_au).astype(float)))
	    AIC_nls = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: MSE
	    sum = np.sum(np.log(shad_ku_T3_pdf(data, *ga_mse)))
	    AIC_ga_mse = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)

	    # AIC GENETIC ALGORITHM - FITNESS: RAD
	    sum = np.sum(np.log(shad_ku_T3_pdf(data, *ga_rad)))
	    AIC_ga_rad = -2*sum + 2*K + (2*K*(K+1))/(n-K-1)


	rads = {'Methods':[f'NLS: {dist}', f'GA: {dist} - MSE', f'GA: {dist} - RAD'],
	    'Akaike Information Criteria':[AIC_nls[0], AIC_ga_mse[0], AIC_ga_rad[0]]}
    
	df = pd.DataFrame(rads)
	#     display(df.sort_values(by='Akaike Information Criteria').reset_index(drop=True))

	df = df.sort_values(by='Akaike Information Criteria').reset_index(drop=True)
	df['Delta_i'] =  df.loc[1:, 'Akaike Information Criteria'] - df.at[0, 'Akaike Information Criteria']
	df['Probability_i'] = np.exp(-df['Delta_i'])/2
	print(df.to_string()+'\n')
	return df

def fun_nls(params, xdata, ydata, dist):
	# params[0] = alpha, params[1] = mu, params[2] = omega, params[3] = md
	# r = Xdata
	if dist == 'a-u':
		res = alpha_mu_pdf(xdata, *params) - ydata
	elif dist == 'k-u':
		res = kappa_mu_pdf(xdata, *params) - ydata
	elif dist == 'sk-u_t1':
		res = shad_ku_T1_pdf(xdata, *params) - ydata
	elif dist == 'sk-u_t2':
		res = shad_ku_T2_pdf(xdata, *params) - ydata
	elif dist == 'sk-u_t3':
		res = shad_ku_T3_pdf(xdata, *params) - ydata
	return res.reshape(-1)

def get_nls(initial_conditions, xdata, ydata, dist, lb, ub):
	res = least_squares(fun_nls, initial_conditions, bounds=(lb, ub), args=(xdata, ydata, dist))
	return res

def get_mle(data, dist, lb):
	if dist == 'a-u':
		mle_est = nakagami.fit(data, floc=0)
		parameters = [2, mle_est[0], mle_est[2]]
	elif dist == 'k-u':
		mle_est = rice.fit(data, floc=0)
		b = mle_est[0]*mle_est[2]
		scale = mle_est[2]
		rice_K = ( b ** 2 ) / ( 2 * ( scale ** 2 ) )
		rice_omega = ( b ** 2 ) + ( 2 * ( scale ** 2 ) )
		parameters = [rice_K, 0.9, rice_omega]
	elif dist == 'sk-u_t1' or dist == 'sk-u_t2' or dist == 'sk-u_t3':
		mle_est = rice.fit(data, floc=0)
		b = mle_est[0]*mle_est[2]
		scale = mle_est[2]
		rice_K = ( b ** 2 ) / ( 2 * ( scale ** 2 ) )
		if rice_K < lb[0]:
			rice_K = lb[0]
		rice_omega = ( b ** 2 ) + ( 2 * ( scale ** 2 ) )
		if rice_omega < lb[2]:
			rice_omega = lb[2]
		parameters = [rice_K, 0.9, rice_omega, 1.1]
	return parameters

def fitness(x, dist, fitness_type, X, Y):
	'''
	Fitness function (MSE/RAD) to determine how good a given solution is.
	'''
	if fitness_type == 'MSE':
		if dist == 'a-u':
			y_pred = alpha_mu_pdf(X, *x).astype(float)
		elif dist == 'k-u':
			y_pred = kappa_mu_pdf(X, *x).astype(float)
		elif dist == 'sk-u_t1':
			y_pred = shad_ku_T1_pdf(X, *x).astype(float)
		elif dist == 'sk-u_t2':
			y_pred = shad_ku_T2_pdf(X, *x).astype(float)
		elif dist == 'sk-u_t3':
			y_pred = shad_ku_T3_pdf(X, *x).astype(float)

		n = len(Y)
		sum = []
		for val in (Y-y_pred):
			sum.append(val**2)
		sum = np.sum(sum)
		cost = 1/n * sum
		return cost

	elif fitness_type == 'RAD':
		pVec1 = Y/np.sum(Y)
		if dist == 'a-u':
			pVec2 = alpha_mu_pdf(X, *x)/np.sum(alpha_mu_pdf(X, *x))
		elif dist == 'k-u':
			pVec2 = kappa_mu_pdf(X, *x)/np.sum(kappa_mu_pdf(X, *x))
		elif dist == 'sk-u_t1':
			pVec2 = shad_ku_T1_pdf(X, *x)/np.sum(shad_ku_T1_pdf(X, *x))
		elif dist == 'sk-u_t2':
			pVec2 = shad_ku_T2_pdf(X, *x)/np.sum(shad_ku_T2_pdf(X, *x))
		elif dist == 'sk-u_t3':
			pVec2 = shad_ku_T3_pdf(X, *x)/np.sum(shad_ku_T3_pdf(X, *x))
		KLD_1 = kl_div(pVec1, pVec2)
		KLD_2 = kl_div(pVec2, pVec1)
		cost = res_avg_distance(KLD_1, KLD_2)
		return cost

def plot_results(dist, Xdata, Ydata, nls, ga_mse, ga_rad, path, nAP, path_folder):
	'''
	Plot the estimates for GA-MSE and GA-RAD.
	'''
	if not os.path.exists(path_folder):
	    os.makedirs(path_folder)

	if dist == 'a-u':
		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), Ydata, c='black')
		plt.plot(20*np.log10(Xdata), alpha_mu_pdf(Xdata, *nls), c='blue', label=r'$\alpha$-$\mu$ NLS')
		plt.plot(20*np.log10(Xdata), alpha_mu_pdf(Xdata, *ga_mse), label=r'$\alpha$-$\mu$ GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), alpha_mu_pdf(Xdata, *ga_rad), label=r'$\alpha$-$\mu$ GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{}.pdf'.format(path, nAP), dpi=150)

		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), 20*np.log10(Ydata), c='black')
		plt.plot(20*np.log10(Xdata), 20*np.log10(alpha_mu_pdf(Xdata, *nls)), c='blue', label=r'$\alpha$-$\mu$ NLS')
		plt.plot(20*np.log10(Xdata), 20*np.log10(alpha_mu_pdf(Xdata, *ga_mse)), label=r'$\alpha$-$\mu$ GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), 20*np.log10(alpha_mu_pdf(Xdata, *ga_rad)), label=r'$\alpha$-$\mu$ GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Log Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{} - log.pdf'.format(path, nAP), dpi=150)

	elif dist == 'k-u':	
		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), Ydata, c='black')
		plt.plot(20*np.log10(Xdata), kappa_mu_pdf(Xdata, *nls), c='blue', label=r'$\kappa$-$\mu$ NLS')
		plt.plot(20*np.log10(Xdata), kappa_mu_pdf(Xdata, *ga_mse), label=r'$\kappa$-$\mu$ GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), kappa_mu_pdf(Xdata, *ga_rad), label=r'$\kappa$-$\mu$ GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{}.pdf'.format(path, nAP), dpi=150)

		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), 20*np.log10(Ydata), c='black')
		plt.plot(20*np.log10(Xdata), 20*np.log10(kappa_mu_pdf(Xdata, *nls)), c='blue', label=r'$\kappa$-$\mu$ NLS')
		plt.plot(20*np.log10(Xdata), 20*np.log10(kappa_mu_pdf(Xdata, *ga_mse)), label=r'$\kappa$-$\mu$ GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), 20*np.log10(kappa_mu_pdf(Xdata, *ga_rad)), label=r'$\kappa$-$\mu$ GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Log Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{} - log.pdf'.format(path, nAP), dpi=150)

	elif dist == 'sk-u_t1':   	
		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), Ydata, c='black')
		plt.plot(20*np.log10(Xdata), shad_ku_T1_pdf(Xdata, *nls), c='blue', label=r'Shad. $\kappa$-$\mu$ T1 NLS')
		plt.plot(20*np.log10(Xdata), shad_ku_T1_pdf(Xdata, *ga_mse), label=r'Shad. $\kappa$-$\mu$ T1 GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), shad_ku_T1_pdf(Xdata, *ga_rad), label=r'Shad. $\kappa$-$\mu$ T1 GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{}.pdf'.format(path, nAP), dpi=150)

		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), 20*np.log10(Ydata), c='black')
		plt.plot(20*np.log10(Xdata), 20*np.log10(shad_ku_T1_pdf(Xdata, *nls)), c='blue', label=r'Shad. $\kappa$-$\mu$ T1 NLS')
		plt.plot(20*np.log10(Xdata), 20*np.log10(shad_ku_T1_pdf(Xdata, *ga_mse)), label=r'Shad. $\kappa$-$\mu$ T1 GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), 20*np.log10(shad_ku_T1_pdf(Xdata, *ga_rad)), label=r'Shad. $\kappa$-$\mu$ T1 GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Log Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{} - log.pdf'.format(path, nAP), dpi=150)

	elif dist == 'sk-u_t2':   	
		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), Ydata, c='black')
		plt.plot(20*np.log10(Xdata), shad_ku_T2_pdf(Xdata, *nls), c='blue', label=r'Shad. $\kappa$-$\mu$ T2 NLS')
		plt.plot(20*np.log10(Xdata), shad_ku_T2_pdf(Xdata, *ga_mse), label=r'Shad. $\kappa$-$\mu$ T2 GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), shad_ku_T2_pdf(Xdata, *ga_rad), label=r'Shad. $\kappa$-$\mu$ T2 GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{}.pdf'.format(path, nAP), dpi=150)

		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), 20*np.log10(Ydata), c='black')
		plt.plot(20*np.log10(Xdata), 20*np.log10(shad_ku_T2_pdf(Xdata, *nls).astype(float)), c='blue', label=r'Shad. $\kappa$-$\mu$ T2 NLS')
		plt.plot(20*np.log10(Xdata), 20*np.log10(shad_ku_T2_pdf(Xdata, *ga_mse).astype(float)), label=r'Shad. $\kappa$-$\mu$ T2 GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), 20*np.log10(shad_ku_T2_pdf(Xdata, *ga_rad).astype(float)), label=r'Shad. $\kappa$-$\mu$ T2 GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Log Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{} - log.pdf'.format(path, nAP), dpi=150)

	elif dist == 'sk-u_t3':   	
		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), Ydata, c='black')
		plt.plot(20*np.log10(Xdata), shad_ku_T3_pdf(Xdata, *nls), c='blue', label=r'Shad. $\kappa$-$\mu$ T3 NLS')
		plt.plot(20*np.log10(Xdata), shad_ku_T3_pdf(Xdata, *ga_mse), label=r'Shad. $\kappa$-$\mu$ T3 GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), shad_ku_T3_pdf(Xdata, *ga_rad), label=r'Shad. $\kappa$-$\mu$ T3 GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{}.pdf'.format(path, nAP), dpi=150)

		plt.figure(figsize=((10,7)))
		plt.scatter(20*np.log10(Xdata), 20*np.log10(Ydata), c='black')
		plt.plot(20*np.log10(Xdata), 20*np.log10(shad_ku_T3_pdf(Xdata, *nls)), c='blue', label=r'Shad. $\kappa$-$\mu$ T3 NLS')
		plt.plot(20*np.log10(Xdata), 20*np.log10(shad_ku_T3_pdf(Xdata, *ga_mse)), label=r'Shad. $\kappa$-$\mu$ T3 GA (MSE)', c='red')
		plt.plot(20*np.log10(Xdata), 20*np.log10(shad_ku_T3_pdf(Xdata, *ga_rad)), label=r'Shad. $\kappa$-$\mu$ T3 GA (RAD)', c='green')
		plt.title(f'Path {path} | RX: {nAP}')
		plt.legend()
		plt.grid(True)
		plt.xlabel(r'Small Scale Fading (dB)')
		plt.ylabel('Log Density')
		plt.savefig(path_folder+'/'+dist+'_path{}_{} - log.pdf'.format(path, nAP), dpi=150)