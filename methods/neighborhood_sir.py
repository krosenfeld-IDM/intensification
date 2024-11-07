""" neighborhood_sir.py

The posterior class for the regularized SIR model adjusted for 
the incorporation of a spatial neighborhood to influence seasonality."""
import sys

## Standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For matrix constructions
from scipy.sparse import diags

class NeighborhoodPosterior(object):

	""" Limited version of the strong prior model to estimate SIA stuff only, which we need to
	create regularizing inputs for the state level """

	def __init__(self,df,sia_effects,S0,S0_var,beta_corr=3.,tau=26,mu_guess=0.1):

		## Store some information about the model
		self.T = len(df)-1
		self.beta_corr = beta_corr
		self.C_t = df["cases"].values
		self.B_t = df["adj_births"].values
		self.sias = sia_effects.values
		self.num_sias = self.sias.shape[1]

		## Regularize S0?
		self.logS0_prior = np.log(S0)
		self.logS0_prior_var = S0_var/(S0**2)
		self.logS0 = np.log(S0)
		self.logS0_var = S0_var/(S0**2)

		## Set the initial guess for mu
		self.mu = mu_guess*np.ones((self.num_sias,))
		self.mu_var = np.zeros((self.num_sias,))
		
		## Initialize key pieces, like the periodic smoothing
		## matrix.
		self.tau = tau
		D2 = np.diag(tau*[-2])+np.diag((tau-1)*[1],k=1)+np.diag((tau-1)*[1],k=-1)
		D2[0,-1] = 1 ## Periodic BCs
		D2[-1,0] = 1
		self.pRW2 = np.dot(D2.T,D2)*((beta_corr**4)/4.)

		## The design matrices for the linear portion of the 
		## transmission regression problem.
		self.X = np.vstack((int(len(df)-1/len(self.pRW2))+1)*[np.eye(len(self.pRW2))])[:len(df)-1]
		self.C = np.linalg.inv(np.dot(self.X.T,self.X)+self.pRW2)
		self.H = np.dot(self.X,np.dot(self.C,self.X.T))

		## And the adjusted case trace, which is the sum of all
		## the underlying state level traces.
		self.adj_cases = df["adj_cases_p"].values

	def compartment_df(self):

		## Compute key quantities
		adj_sias = (self.mu*self.sias[:-1]).sum(axis=1)
		E_t = self.adj_cases[1:]
		I_t = self.adj_cases[:-1]
		S_t = np.exp(self.logS0) + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		
		## Pack them up
		time = np.arange(0,len(I_t)+1)
		E_t = pd.Series(E_t,index=time[1:],name="E_t")
		I_t = pd.Series(I_t,index=time[:-1],name="I_t")
		S_t = pd.Series(S_t,index=time[:-1],name="S_t")
		out = pd.concat([E_t,I_t,S_t],axis=1)
		return out
		
	def fixed_rt(self,theta):

		## Unpack 
		S0 = np.exp(theta[0])
		mu = theta[1:]

		## Compute key quantities
		adj_sias = (mu*self.sias[:-1]).sum(axis=1)
		E_t = self.adj_cases[1:]
		I_t = self.adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Solve the linear regression problem
		Y_hat = np.dot(self.H,Y_t)
		RSS = np.sum((Y_t-Y_hat)**2)
		lnp_beta = 0.5*self.T*np.log(RSS/self.T)

		## If you added S0 var
		lnp_S0 = 0.5*((theta[0]-self.logS0_prior)**2)/(self.logS0_prior_var)
		
		return lnp_beta + lnp_S0

	def fixed_rt_grad(self,theta):

		## Unpack 
		S0 = np.exp(theta[0])
		mu = theta[1:]

		## Compute key quantities
		adj_sias = (mu*self.sias[:-1]).sum(axis=1)
		E_t = self.adj_cases[1:]
		I_t = self.adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Solve the linear regression problem and compute
		## The implied variance
		Y_hat = np.dot(self.H,Y_t)
		resid = Y_t-Y_hat
		var = np.sum(resid**2)/self.T

		## Compute the contribution from S_t
		dYdt0 = -S0/S_t 
		grad_t = np.dot(dYdt0-np.dot(self.H,dYdt0),resid)/var
		grad_t += (theta[0]-self.logS0_prior)/self.logS0_prior_var

		## Compute the contribution for mu
		dYdmu = np.cumsum(self.sias[:-1],axis=0)/S_t[:,np.newaxis]
		grad_mu = np.dot((dYdmu-np.dot(self.H,dYdmu)).T,resid)/var

		## Combine and return
		jac = np.zeros((len(theta),))
		jac[0] = grad_t
		jac[1:] = grad_mu
		return jac

	def fixed_rt_hessian(self,theta):

		## Unpack the S0-mu point of interest 
		S0 = np.exp(theta[0])
		mu = theta[1:]

		## Compute key quantities
		adj_sias = (mu*self.sias[:-1]).sum(axis=1)
		E_t = self.adj_cases[1:]
		I_t = self.adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Solve the linear regression problem and compute
		## The implied variance
		Y_hat = np.dot(self.H,Y_t)
		resid = Y_t-Y_hat
		var = np.sum(resid**2)/self.T

		## Compute the S0-S0 term.
		dYdx0 = -S0/S_t 
		dYdx0dx0 = dYdx0**2 + dYdx0
		dS0dS0 = np.dot(dYdx0dx0-np.dot(self.H,dYdx0dx0),resid)/var+\
				 np.dot(dYdx0-np.dot(self.H,dYdx0),dYdx0-np.dot(self.H,dYdx0))/var+\
				 -(2./self.T)*((np.dot(dYdx0-np.dot(self.H,dYdx0),resid)/var)**2)
		dS0dS0 += 1./self.logS0_prior_var

		## Compute the mu-mu terms
		dYdmu = np.cumsum(self.sias[:-1],axis=0)/S_t[:,np.newaxis]
		dresid = dYdmu-np.dot(self.H,dYdmu)
		dmudmu = np.dot(dresid.T,dresid)/var+\
				 -(2./self.T)*np.dot(np.dot(dresid.T,resid),np.dot(dresid.T,resid).T)/var**2
		for i in range(self.num_sias):
			dYdmudi = dYdmu[:,i,np.newaxis]*dYdmu
			dmudmu[i,:] += np.dot((dYdmudi-np.dot(self.H,dYdmudi)).T,resid)/var
		
		## Compute the cross terms
		dYdx0dmu = dYdx0[:,np.newaxis]*dYdmu
		dS0dmu = np.dot((dYdx0dmu-np.dot(self.H,dYdx0dmu)).T,resid)/var+\
				 np.dot(dresid.T,dYdx0-np.dot(self.H,dYdx0))/var+\
				 -(2./self.T)*(np.dot(dresid.T,resid)/var)*(np.dot(dYdx0-np.dot(self.H,dYdx0),resid)/var)
		
		## Put it all together
		hess = np.zeros((self.num_sias+1,self.num_sias+1))
		hess[0,0] = dS0dS0
		hess[0,1:] = dS0dmu
		hess[1:,0] = dS0dmu
		hess[1:,1:] = dmudmu

		return hess

class HoodRegularizedModel(object):

	def __init__(self,df,sia_effects,S0,S0_var,hood_t,beta_corr=3.,tau=26,mu_guess=0.1):

		## Store some information about the model
		self.T = len(df)-1
		self.beta_corr = beta_corr
		self.C_t = df["cases"].values
		self.B_t = df["adj_births"].values
		self.sias = sia_effects.values
		self.num_sias = self.sias.shape[1]

		## Regularize S0?
		self.logS0_prior = np.log(S0)
		self.logS0_prior_var = S0_var/(S0**2)
		self.logS0 = np.log(S0)
		self.logS0_var = S0_var/(S0**2)

		## Set the initial guess for mu
		self.mu = mu_guess*np.ones((self.num_sias,))
		self.mu_var = np.zeros((self.num_sias,))

		## And the reporting rate prior pieces
		self.r_prior = df["rr_p"].values
		self.r_prior_precision = np.diag(1./(df["rr_p_var"].values))
		self.r_floor = (df["rr_p"]-4.*np.sqrt(df["rr_p_var"])).min()
		self.r_hat = df["rr_p"].values
		
		## Initialize key pieces, like the periodic smoothing
		## matrix.
		self.tau = tau
		D2 = np.diag(tau*[-2])+np.diag((tau-1)*[1],k=1)+np.diag((tau-1)*[1],k=-1)
		D2[0,-1] = 1 ## Periodic BCs
		D2[-1,0] = 1
		self.pRW2 = np.dot(D2.T,D2)*((beta_corr**4)/4.)

		## Which fits into a regularization matrix sized to account
		## for the scale factor between seasonalities (frequency dependent 
		## transmission, lol)
		self.lam = np.zeros((self.tau+1,self.tau+1))
		self.lam[1:,1:] = self.pRW2

		## And additionally the covariate associated with the neighborhood
		## dynamics.
		self.Y_N = np.log(hood_t["E_t"].values[1:])-\
					np.log(hood_t["I_t"].values[:-1])-\
					np.log(hood_t["S_t"].values[:-1])
		assert len(self.Y_N) == self.T,\
			"The state and hood dfs have to be on the same time-scale."

		## Construct the full design matrix, first by making one for a single
		## time series, then duplicating it and adding the scale factor column.
		X_t = np.vstack((int(len(df)-1/self.tau)+1)*[np.eye(self.tau)])[:len(df)-1]
		self.X = np.vstack([X_t,X_t])
		self.X = np.hstack([np.zeros((len(self.X),1)),
							self.X])
		self.X[:self.T,0] = 1.

		## The design matrices for the linear portion of the 
		## transmission regression problem.
		self.C = np.linalg.inv(np.dot(self.X.T,self.X)+self.lam)
		self.H = np.dot(self.X,np.dot(self.C,self.X.T))

	def fixed_mu(self,theta):

		## Unpack the input
		S0 = np.exp(theta[0])
		r_t = theta[1:]
	
		## Compute the implied model compartments
		adj_cases = (self.C_t+1.)/r_t - 1.
		adj_sias = (self.mu*self.sias[:-1]).sum(axis=1)
		E_t = adj_cases[1:]
		I_t = adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Augment Y_t with the neighboorhood information
		Y_tot = np.hstack([Y_t,self.Y_N])

		## Solve the linear regression problem
		Y_hat = np.dot(self.H,Y_tot)
		RSS = np.sum((Y_tot-Y_hat)**2)
		lnp_beta = 0.5*(2.*self.T)*np.log(RSS/(2.*self.T))

		## If you added S0 var
		lnp_S0 = 0.5*((theta[0]-self.logS0_prior)**2)/self.logS0_prior_var
		
		## Compute the r_t component
		lnp_rt = 0.5*np.sum(((r_t-self.r_prior)**2)*np.diag(self.r_prior_precision))

		return lnp_beta + lnp_rt + lnp_S0

	def fixed_mu_grad(self,theta):

		## Unpack the input
		S0 = np.exp(theta[0])
		r_t = theta[1:]

		## Compute the implied model compartments
		adj_cases = (self.C_t+1.)/r_t - 1.
		adj_sias = (self.mu*self.sias[:-1]).sum(axis=1)
		E_t = adj_cases[1:]
		I_t = adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Augment Y_t with the neighboorhood information
		Y_tot = np.hstack([Y_t,self.Y_N])

		## Solve the linear regression problem and compute
		## The implied variance
		Y_hat = np.dot(self.H,Y_tot)
		resid = Y_tot-Y_hat
		var = np.sum(resid**2)/(2.*self.T)

		## Compute the contribution from S_t
		dYdt0 = -S0/S_t 
		dYdt0 = np.hstack([dYdt0,np.zeros(dYdt0.shape)])
		grad_t = np.dot(dYdt0-np.dot(self.H,dYdt0),resid)/var
		grad_t += (theta[0]-self.logS0_prior)/self.logS0_prior_var

		## Compute the contribution for r_t 
		dYtdrt = (I_t+1.)/(I_t*r_t[:-1])
		dYtdrt_plus1 = -(E_t+1.)/(E_t*r_t[1:])
		dYdr = diags([dYtdrt,dYtdrt_plus1],[0,1],shape=(self.T,self.T+1)).todense()
		dYdr[:,1:] += -np.tril(np.outer(1./S_t,(E_t+1.)/r_t[1:]))
		dYdr = np.vstack([dYdr,np.zeros(dYdr.shape)])
		grad_r = np.dot((dYdr-np.dot(self.H,dYdr)).T,resid)/var
		grad_r += np.dot(self.r_prior_precision,r_t-self.r_prior)
		
		## Combine and return
		jac = np.zeros((len(theta),))
		jac[0] = grad_t
		jac[1:] = grad_r
		return jac

	def fixed_rt(self,theta):

		## Unpack 
		S0 = np.exp(theta[0])
		mu = theta[1:]

		## Compute key quantities
		adj_cases = (self.C_t+1.)/self.r_hat - 1.
		adj_sias = (mu*self.sias[:-1]).sum(axis=1)
		E_t = adj_cases[1:]
		I_t = adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Augment Y_t with the neighboorhood information
		Y_tot = np.hstack([Y_t,self.Y_N])
		
		## Solve the linear regression problem
		Y_hat = np.dot(self.H,Y_tot)
		RSS = np.sum((Y_tot-Y_hat)**2)
		lnp_beta = 0.5*(2.*self.T)*np.log(RSS/(2.*self.T))

		## If you added S0 var
		lnp_S0 = 0.5*((theta[0]-self.logS0_prior)**2)/(self.logS0_prior_var)
		
		return lnp_beta + lnp_S0

	def fixed_rt_grad(self,theta):

		## Unpack 
		S0 = np.exp(theta[0])
		mu = theta[1:]

		## Compute key quantities
		adj_cases = (self.C_t+1.)/self.r_hat - 1.
		adj_sias = (mu*self.sias[:-1]).sum(axis=1)
		E_t = adj_cases[1:]
		I_t = adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Augment Y_t with the neighboorhood information
		Y_tot = np.hstack([Y_t,self.Y_N])
		
		## Solve the linear regression problem and compute
		## The implied variance
		Y_hat = np.dot(self.H,Y_tot)
		resid = Y_tot-Y_hat
		var = np.sum(resid**2)/(2.*self.T)

		## Compute the contribution from S_t
		dYdt0 = -S0/S_t
		dYdt0 = np.hstack([dYdt0,np.zeros(dYdt0.shape)])
		grad_t = np.dot(dYdt0-np.dot(self.H,dYdt0),resid)/var
		grad_t += (theta[0]-self.logS0_prior)/self.logS0_prior_var

		## Compute the contribution for mu
		dYdmu = np.cumsum(self.sias[:-1],axis=0)/S_t[:,np.newaxis]
		dYdmu = np.vstack([dYdmu,np.zeros(dYdmu.shape)])
		grad_mu = np.dot((dYdmu-np.dot(self.H,dYdmu)).T,resid)/var

		## Combine and return
		jac = np.zeros((len(theta),))
		jac[0] = grad_t
		jac[1:] = grad_mu
		return jac

	def fixed_rt_hessian(self,theta):

		## Unpack the S0-mu point of interest 
		S0 = np.exp(theta[0])
		mu = theta[1:]

		## Compute key quantities
		adj_cases = (self.C_t+1.)/self.r_hat - 1.
		adj_sias = (mu*self.sias[:-1]).sum(axis=1)
		E_t = adj_cases[1:]
		I_t = adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Augment Y_t with the neighboorhood information
		Y_tot = np.hstack([Y_t,self.Y_N])

		## Solve the linear regression problem and compute
		## The implied variance
		Y_hat = np.dot(self.H,Y_tot)
		resid = Y_tot-Y_hat
		var = np.sum(resid**2)/(2.*self.T)

		## Compute the S0-S0 term.
		dYdx0 = -S0/S_t 
		dYdx0dx0 = dYdx0**2 + dYdx0
		dYdx0 = np.hstack([dYdx0,np.zeros(dYdx0.shape)])
		dYdx0dx0 = np.hstack([dYdx0dx0,np.zeros(dYdx0dx0.shape)])
		dS0dS0 = np.dot(dYdx0dx0-np.dot(self.H,dYdx0dx0),resid)/var+\
				 np.dot(dYdx0-np.dot(self.H,dYdx0),dYdx0-np.dot(self.H,dYdx0))/var+\
				 -(2./(2.*self.T))*((np.dot(dYdx0-np.dot(self.H,dYdx0),resid)/var)**2)
		dS0dS0 += 1./self.logS0_prior_var

		## Compute the mu-mu terms
		dYdmu = np.cumsum(self.sias[:-1],axis=0)/S_t[:,np.newaxis]
		dYdmu = np.vstack([dYdmu,np.zeros(dYdmu.shape)])
		dresid = dYdmu-np.dot(self.H,dYdmu)
		dmudmu = np.dot(dresid.T,dresid)/var+\
				 -(2./(2.*self.T))*np.dot(np.dot(dresid.T,resid),np.dot(dresid.T,resid).T)/var**2
		for i in range(self.num_sias):
			dYdmudi = dYdmu[:,i,np.newaxis]*dYdmu
			dmudmu[i,:] += np.dot((dYdmudi-np.dot(self.H,dYdmudi)).T,resid)/var
		
		## Compute the cross terms
		dYdx0dmu = dYdx0[:,np.newaxis]*dYdmu
		dS0dmu = np.dot((dYdx0dmu-np.dot(self.H,dYdx0dmu)).T,resid)/var+\
				 np.dot(dresid.T,dYdx0-np.dot(self.H,dYdx0))/var+\
				 -(2./(2.*self.T))*(np.dot(dresid.T,resid)/var)*(np.dot(dYdx0-np.dot(self.H,dYdx0),resid)/var)
		
		## Put it all together
		hess = np.zeros((self.num_sias+1,self.num_sias+1))
		hess[0,0] = dS0dS0
		hess[0,1:] = dS0dmu
		hess[1:,0] = dS0dmu
		hess[1:,1:] = dmudmu

		return hess

	def __call__(self,theta):

		## Unpack 
		S0 = np.exp(theta[0])
		mu = theta[1:self.num_sias+1]
		r_t = theta[1+self.num_sias:]

		## Compute key quantities
		adj_cases = (self.C_t+1.)/r_t - 1.
		adj_sias = (mu*self.sias[:-1]).sum(axis=1)
		E_t = adj_cases[1:]
		I_t = adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Augment Y_t with the neighboorhood information
		Y_tot = np.hstack([Y_t,self.Y_N])
		
		## Solve the linear regression problem
		Y_hat = np.dot(self.H,Y_tot)
		RSS = np.sum((Y_tot-Y_hat)**2)
		lnp_beta = 0.5*(2.*self.T)*np.log(RSS/(2.*self.T))

		## If you added S0 var
		lnp_S0 = 0.5*((theta[0]-self.logS0_prior)**2)/(self.logS0_prior_var)
		
		## Compute the r_t component
		lnp_rt = 0.5*np.sum(((r_t-self.r_prior)**2)*np.diag(self.r_prior_precision))

		return lnp_beta + lnp_rt + lnp_S0

