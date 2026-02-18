""" neighborhood_sir.py

The posterior class for the regularized SIR model adjusted for 
the incorporation of a spatial neighborhood to influence seasonality."""
import os

## Standard stuff
import numpy as np
import pandas as pd

## For matrix constructions
from scipy.sparse import diags

## For optimization
from scipy.optimize import minimize

class NeighborhoodPosterior(object):

    """ Limited version of the full model to estimate SIA impact only, which we need to
    create regularizing inputs for the state level. See the discussion around Appendix 2,
    equations 3 and 4. 

    This object encapsulates methods to evaluate the heirachical posterior distribution in
    different ways and its gradient to facilitate model fitting. """

    def __init__(self,df,sia_effects,S0,S0_var,beta_corr=3.,tau=26,mu_guess=0.1):

        ## Store some information about the model, i.e.
        ## inputs and their structure.
        self.T = len(df)-1
        self.beta_corr = beta_corr
        self.C_t = df["cases"].values
        self.B_t = df["adj_births"].values
        self.sias = sia_effects.values
        self.num_sias = self.sias.shape[1]

        ## Information needed to regularize 
        ## S0 from the annualize-time prior, which we choose to
        ## implement in log-space for numerical stability
        ## reasons.
        self.logS0_prior = np.log(S0)
        self.logS0_prior_var = S0_var/(S0**2)
        self.logS0 = np.log(S0)
        self.logS0_var = S0_var/(S0**2)

        ## Set the initial guess for mu, the fraction
        ## of sia doses that seroconvert a susceptible.
        self.mu = mu_guess*np.ones((self.num_sias,))
        self.mu_var = np.zeros((self.num_sias,))
        
        ## Initialize key pieces, like the periodic smoothing
        ## matrix associated with the seasonality prior..
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

        ## Compute key quantities related to the populations
        ## in the different model compartments. In this formulation
        ## E_t (exposed) is simply I_t (infectious) shifted by one
        ## time step.
        adj_sias = (self.mu*self.sias[:-1]).sum(axis=1)
        E_t = self.adj_cases[1:]
        I_t = self.adj_cases[:-1]
        S_t = np.exp(self.logS0) + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
        
        ## Pack them up into a single dataframe.
        time = np.arange(0,len(I_t)+1)
        E_t = pd.Series(E_t,index=time[1:],name="E_t")
        I_t = pd.Series(I_t,index=time[:-1],name="I_t")
        S_t = pd.Series(S_t,index=time[:-1],name="S_t")
        out = pd.concat([E_t,I_t,S_t],axis=1).sort_index()
        return out
        
    def fixed_rt(self,theta):

        ''' Evaluate the posterior on iterations where adj_cases is fixed. '''

        ## Unpack the input array.
        S0 = np.exp(theta[0])
        mu = theta[1:]

        ## Compute key quantities associated with
        ## the model populations.
        adj_sias = (mu*self.sias[:-1]).sum(axis=1)
        E_t = self.adj_cases[1:]
        I_t = self.adj_cases[:-1]
        S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
        Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

        ## Solve the linear regression problem associated with
        ## the transmission rate.
        Y_hat = np.dot(self.H,Y_t)
        RSS = np.sum((Y_t-Y_hat)**2)
        lnp_beta = 0.5*self.T*np.log(RSS/self.T)

        ## Add the prior for S0.
        lnp_S0 = 0.5*((theta[0]-self.logS0_prior)**2)/(self.logS0_prior_var)
        
        return lnp_beta + lnp_S0

    def fixed_rt_grad(self,theta):

        ## Unpack the input array.
        S0 = np.exp(theta[0])
        mu = theta[1:]

        ## Compute key quantities associated with
        ## the model populations.
        adj_sias = (mu*self.sias[:-1]).sum(axis=1)
        E_t = self.adj_cases[1:]
        I_t = self.adj_cases[:-1]
        S_t = S0 + np.cumsum(self.B_t[:-1]-E_t-adj_sias)
        Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)
        
        ## Solve the linear regression problem associated with
        ## the transmission rate.
        Y_hat = np.dot(self.H,Y_t)
        resid = Y_t-Y_hat
        var = np.sum(resid**2)/self.T

        ## Compute the contribution from the S_t prior.
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

    """ This is the main model class, to be applied at the level-of-interest (state level in the 
    paper), which takes the regularizing model output above as input. See the discussion around 
    appendix 2, equations 1 and 2 and equation 6. 

    This object encapsulates methods to evaluate the heirachical posterior distribution in
    different ways and its gradient to facilitate model fitting. """

    def __init__(self,df,sia_effects,S0,S0_var,hood_t,beta_corr=3.,tau=26,mu_guess=0.1):

        ## Store some information about the model, including
        ## the inputs and problem size.
        self.T = len(df)-1
        self.beta_corr = beta_corr
        self.C_t = df["cases"].values
        self.B_t = df["adj_births"].values
        self.sias = sia_effects.values
        self.num_sias = self.sias.shape[1]

        ## Information needed to regularize 
        ## S0 from the annualize-time prior, which we choose to
        ## implement in log-space for numerical stability
        ## reasons.
        self.logS0_prior = np.log(S0)
        self.logS0_prior_var = S0_var/(S0**2)
        self.logS0 = np.log(S0)
        self.logS0_var = S0_var/(S0**2)

        ## Set the initial guess for mu, the SIA efficacy.
        self.mu = mu_guess*np.ones((self.num_sias,))
        self.mu_var = np.zeros((self.num_sias,))

        ## And set up the reporting rate prior pieces
        self.r_prior = df["rr_p"].values
        self.r_prior_precision = np.diag(1./(df["rr_p_var"].values))
        self.r_floor = (df["rr_p"]-4.*np.sqrt(df["rr_p_var"])).min()
        self.r_hat = df["rr_p"].values
        
        ## Initialize key regression pieces, like the periodic smoothing
        ## matrix to be applied to the seasonality profile.
        self.tau = tau
        D2 = np.diag(tau*[-2])+np.diag((tau-1)*[1],k=1)+np.diag((tau-1)*[1],k=-1)
        D2[0,-1] = 1 ## Periodic BCs
        D2[-1,0] = 1
        self.pRW2 = np.dot(D2.T,D2)*((beta_corr**4)/4.)

        ## The above fits into a regularization matrix sized to account
        ## for the scale factor between seasonalities, gamma in the
        ## manuscript's appendix 2 equation 2.
        self.lam = np.zeros((self.tau+1,self.tau+1))
        self.lam[1:,1:] = self.pRW2

        ## And additionally the covariate associated with the neighborhood
        ## dynamics.
        self.Y_N = np.log(hood_t["E_t"].values[1:])-\
                    np.log(hood_t["I_t"].values[:-1])-\
                    np.log(hood_t["S_t"].values[:-1])
        assert len(self.Y_N) == self.T,\
            "The state and hood dfs have to be on the same time-index."

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

        ''' Evaluate the posterior on iterations where SIA impact, mu, is fixed. '''

        ## Unpack the input
        S0 = np.exp(theta[0])
        r_t = theta[1:]
    
        ## Compute the implied model compartments populations.
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

        ## Compute the S0 prior term
        lnp_S0 = 0.5*((theta[0]-self.logS0_prior)**2)/self.logS0_prior_var
        
        ## Compute the r_t prior term.
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

        ''' Evaluate the posterior on iterations where adj_cases is fixed. '''

        ## Unpack 
        S0 = np.exp(theta[0])
        mu = theta[1:]

        ## Compute key quantities, i.e. compartment
        ## populations in the state level model.
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

        ## Add the prior contribution from S0.
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

        ''' Evaluate the full, heirachical posterior across all parameters. This is mainly to facilitate
        visualization, posterior profiling, etc. as opposed to model fitting (which is more stable in stages
        as above). '''

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

######################################################################################################
## Some helper functions used throughout the model scripts.
######################################################################################################
def get_age_pyramid(state,fname=os.path.join("_data","grid3_population_by_state.csv")):

    ## Get the output from grid3
    df = pd.read_csv(fname,index_col=0)\
            .set_index(["state","age_bin"])
    df = df.loc[state].reset_index()
    population = int(np.round(df["total"].sum()))
    
    ## Make an age column, representing the start 
    ## of the age bins
    df["age"] = df["age_bin"].apply(lambda s: int(s.split()[0]))
    df = df.sort_values("age")

    ## Interpolate to yearly bins.
    pyramid = df[["age","total"]].set_index("age")["total"]
    pyramid = pyramid.reindex(np.arange(pyramid.index[-1]+5)).fillna(method="ffill")
    pyramid.loc[1:4] = pyramid.loc[1:4]/4
    pyramid.loc[5:] = pyramid.loc[5:]/5

    ## Turn it into a distribution
    pyramid = pyramid/population

    return pyramid, population

def prep_model_inputs(state,time_index,epi,cr,dists,mcv1_effic=0.825,mcv2_effic=0.95):

    '''This functions prepares the dataframe that has all the key inputs (cases, births, etc.) on the
    same time scale and in terms of the necessary inputs to the model (RI adjusted births, for example). '''

    ## Start by aggregating the epi data to the semi-monthly timescale.
    ## SMS: SemiMonthly-Start, i.e. first and 15th of every month.
    df = epi.resample("SMS").agg({"cases":"sum",
                                  "rejected":"sum",
                                  "births":"sum",
                                  "births_var":"sum",
                                  "mcv1":"mean",
                                  "mcv1_var":"mean",
                                  "mcv2":"mean",
                                  "mcv2_var":"mean"})
    df["births"] = df["births"].rolling(2).mean()
    df["births_var"] = df["births_var"].rolling(2).mean()
    df = df.loc[time_index] 

    ## Add a population column, which is constant over time.
    _, population = get_age_pyramid(state)
    df["population"] = len(df)*[population]

    ## Unpack the coarse regression outputs 
    ## and interpolate to the fine time scale.
    initial_S0 = cr.loc[2009,"S0"]
    initial_S0_var = cr.loc[2009,"S0_var"]
    rr_prior = cr[["rr","rr_var"]].copy().reset_index()
    rr_prior.columns = ["time","rr_p","rr_p_var"]
    rr_prior["time"] = pd.to_datetime({"year":rr_prior["time"],
                                       "month":1,
                                       "day":15})
    rr_prior = rr_prior.set_index("time")
    rr_prior = rr_prior.resample("d").interpolate().reindex(df.index)
    rr_prior = rr_prior.fillna(method="bfill").fillna(method="ffill")
    
    ## Add the reporting rate prior information to the 
    ## overall dataframe.
    df = pd.concat([df,rr_prior],axis=1)

    ## And the initial condition information
    df["initial_S0"] = len(df)*[initial_S0]
    df["initial_S0_var"] = len(df)*[initial_S0_var]

    ## Experimentally add some survival corrections to account for infection
    ## before RI. In sensitivity testing, this is a small effect, and we set it to 0
    ## for the manuscript, but retain the functionality here.
    survival = 1.-np.cumsum(0*dists,axis=1)
    pr_sus_at_mcv1 = survival[0]
    pr_sus_at_mcv1.index = pd.to_datetime({"year":pr_sus_at_mcv1.index,
                                           "month":6,"day":15})
    pr_sus_at_mcv1 = pr_sus_at_mcv1.resample("d").interpolate().reindex(time_index)
    pr_sus_at_mcv1 = pr_sus_at_mcv1.fillna(method="ffill")
    pr_sus_at_mcv2 = survival[1]
    pr_sus_at_mcv2.index = pd.to_datetime({"year":pr_sus_at_mcv2.index,
                                           "month":6,"day":15})
    pr_sus_at_mcv2 = pr_sus_at_mcv2.resample("d").interpolate().reindex(time_index)
    pr_sus_at_mcv2 = pr_sus_at_mcv2.fillna(method="ffill")

    ## Set up vaccination corrections (i.e. sink terms).
    df["v1"] = (df["births"]*mcv1_effic*pr_sus_at_mcv1*df["mcv1"]).shift(18).fillna(method="bfill")
    df["v1_var"] = mcv1_effic*pr_sus_at_mcv1*(df["births"]*df["mcv1"]*(1.-df["mcv1"])+\
                    df["mcv1_var"]*(df["births"]**2)+\
                    df["births_var"]*(df["mcv1"]**2)).shift(18).fillna(method="bfill")

    ## Compute immunizations from MCV2 as well.
    mcv1_failures = df["v1"]*(1.-mcv1_effic)/mcv1_effic
    mcv1_failures_var = df["v1_var"]*(1.-mcv1_effic)/mcv1_effic
    df["v2"] = (mcv2_effic*df["mcv2"]*pr_sus_at_mcv2*mcv1_failures).shift(30-18).fillna(method="bfill")
    df["v2_var"] = mcv2_effic*pr_sus_at_mcv2*(mcv1_failures*df["mcv2"]*(1.-df["mcv2"])+\
                    df["mcv2_var"]*(mcv1_failures**2)+\
                    mcv1_failures_var*(df["mcv2"]**2)).shift(30-18).fillna(method="bfill")

    ## Construct adjusted births, i.e. the estimate of the number of births
    ## missed by RI that become susceptible.
    df["adj_births"] = df["births"]-df["v1"]-df["v2"]
    df["adj_births_var"] = df["births_var"]+df["v1_var"]+df["v2_var"]

    ## Collect effects besides SIA and initial susceptibility, i.e. the growth in
    ## susceptibility in the absence of SIAs and infection. This is just for convenience, to
    ## not have to repeatedly accumulate adj_births.
    df = df.loc["2009-01-01":]
    df["S_t_tilde"] = np.cumsum(df["adj_births"])

    ## And finally compute prior adjusted cases
    df["adj_cases_p"] = (df["cases"]+1.)/df["rr_p"] - 1.

    return df

def prep_sia_effects(cal,time_index,md=False):

    ## Get the SIA calendar to collect SIA effects, looping over campaigns
    ## and aligning to the time steps 
    cal = cal.loc[(cal["start_date"] >= time_index[0]) &\
                  (cal["start_date"] <= time_index[-1])]
    cal = cal.sort_values("start_date").reset_index(drop=True)
    cal["time"] = cal["start_date"]+0.5*(cal["end_date"].fillna(cal["start_date"])-cal["start_date"])
    cal["time"] = cal["time"].dt.round("d")

    ## Capture the metadata
    sia_metadata = cal.reset_index().rename(columns={"index":"sia_num"})
    
    ## And make a time series version
    sia_effects = cal[["time","doses"]].copy()
    sia_effects["time"] = sia_effects["time"].apply(lambda t: np.argmin(np.abs(t-time_index)))
    sia_effects["time"] = time_index[sia_effects["time"].values]
    
    ## Consolidate any overlapping dates, and reshape into one
    ## timeseries per SIA, with the doses at the approporate dates
    sia_effects = sia_effects.groupby("time").sum().reset_index()
    sia_effects = sia_effects.reset_index().rename(columns={"index":"sia_num"})
    sia_effects = sia_effects.pivot(index="time",columns="sia_num",values="doses")
    sia_effects = sia_effects.reindex(time_index).fillna(0)

    ## Return with meta data in some instances, otherwise just send
    ## back the time series.
    if md:
        return sia_effects, sia_metadata
    else:
        return sia_effects

def fit_the_neighborhood_model(region,hood_df,hood_sias,initial_mu_guess=0.1):

    '''This function fits the model at the neighborhood level, leveraging the class above and 
    encapsulated methods therein. '''

    ## Create a model object.
    hoodP = NeighborhoodPosterior(
                hood_df,
                hood_sias,
                hood_df["initial_S0"].values[0],
                hood_df["initial_S0_var"].values[0],
                beta_corr=3.,
                tau=24,
                mu_guess=initial_mu_guess,
                )

    ## Fit this auxillary model by finding good SIAS given the
    ## coarse regression approximation to r_t
    x0 = np.ones((hoodP.num_sias+1,))
    x0[0] = hoodP.logS0_prior
    x0[1:] = hoodP.mu
    sia_op = minimize(hoodP.fixed_rt,
                      x0=x0,
                      jac=hoodP.fixed_rt_grad,
                      method="L-BFGS-B",
                      bounds=[(None,None)]+(len(x0)-1)*[(0,1)],
                      options={"ftol":1e-13,
                               "maxcor":100,
                               },
                      )
    print("\nResult from SIA optimization for the {}"
          " region... ".format(region.title()))
    print("Success = {}".format(sia_op.success))
    hoodP.logS0 = sia_op["x"][0]
    hoodP.mu = sia_op["x"][1:]

    return hoodP

def fit_the_regularized_model(state,state_df,state_sias,hood_t,initial_mu_guess=0.5):

    '''This function fits the model at the state level, leveraging the class above, i.e. the 
    encapsulated methods therein, as well as the fitted neighborhood model output (hood_t). '''

    ## Then use that estimate of the compartments to
    ## inform seasonality in the state level model.
    neglp = HoodRegularizedModel(
                state_df,
                state_sias,
                state_df["initial_S0"].values[0],
                state_df["initial_S0_var"].values[0],
                hood_t,
                beta_corr=3.,
                tau=24,
                mu_guess=initial_mu_guess)

    ## Fit the model by first finding good SIAS given the
    ## coarse regression approximation to r_t
    x0 = np.ones((neglp.num_sias+1,))
    x0[0] = neglp.logS0_prior
    x0[1:] = neglp.mu
    sia_op = minimize(neglp.fixed_rt,
                      x0=x0,
                      jac=neglp.fixed_rt_grad,
                      method="L-BFGS-B",
                      bounds=[(None,None)]+(len(x0)-1)*[(0,1)],
                      options={"ftol":1e-13,
                               "maxcor":100,
                               },
                      )
    print("\nResult from fixed r_t SIA optimization for just {}:".format(state))
    print("Success = {}".format(sia_op.success))
    neglp.logS0 = sia_op["x"][0]
    neglp.mu = sia_op["x"][1:]

    ## Then adjust the reporting rate given the SIAs
    x0 = np.ones((1+neglp.T+1,))
    x0[0] = neglp.logS0
    x0[1:] = neglp.r_hat
    rep_op = minimize(neglp.fixed_mu,
                      x0=x0,
                      jac=neglp.fixed_mu_grad,
                      method="L-BFGS-B",
                      bounds=[(None,None)]+(len(x0)-1)*[(5.e-4,1)],
                      options={"ftol":1e-13,
                               "maxcor":100,
                               },
                      )
    print("\n...And from fixed SIA r_t optimization")
    print("Success = {}".format(rep_op.success))
    neglp.logS0 = rep_op["x"][0]
    neglp.r_hat = rep_op["x"][1:]

    ## Compute the covariance matrix conditional on the
    ## reporting adjusted estimates
    x0 = np.ones((neglp.num_sias+1,))
    x0[0] = neglp.logS0
    x0[1:] = neglp.mu
    hessian = neglp.fixed_rt_hessian(x0)
    cov = np.linalg.inv(hessian)
    
    ## Finalize the uncertainty estimates
    overall_var = np.diag(cov)
    neglp.logS0_var = overall_var[0]
    neglp.mu_var = overall_var[1:]  

    ## Finally, fully specify the transmission parameters using the
    ## SIA and reporting rate estimates via the optimization above. In this case
    ## we relax the alpha = 1 constraint leveraged to make the optimization faster
    ## in the previous section. See the discussion under appendix 2, equation 7.
    adj_cases = ((state_df["cases"].values+1.)/neglp.r_hat)-1.
    adj_births = state_df["adj_births"].values
    adj_sias = (neglp.mu*neglp.sias[:-1]).sum(axis=1)
    E_t = adj_cases[1:]
    I_t = adj_cases[:-1]
    S_t = np.exp(neglp.logS0)+np.cumsum(state_df["adj_births"].values[:-1]-E_t-adj_sias)

    ## Which let's us estimate via a single log-linear regression 
    ## the alpha != 1 model parameters...
    print("\nSpecifying the final transmission term...")
    Y_t = np.log(E_t)-np.log(S_t)
    X = np.hstack([neglp.X[:neglp.T,1:],np.log(I_t)[:,np.newaxis]])
    pRW2 = np.zeros((X.shape[1],X.shape[1]))
    pRW2[:-1,:-1] = neglp.pRW2
    C = np.linalg.inv(np.dot(X.T,X)+pRW2)
    beta_hat = np.dot(C,np.dot(X.T,Y_t))
    beta_t = np.dot(X,beta_hat)
    RSS = np.sum((Y_t-beta_t)**2)
    sig_eps = np.sqrt(RSS/(neglp.T))#-X.shape[1]))
    print("sig_eps = {}".format(sig_eps))
    beta_cov = sig_eps*sig_eps*C
    beta_var = np.diag(beta_cov)
    beta_std = np.sqrt(beta_var)
    beta_t_std = np.sqrt(np.diag(np.dot(X,np.dot(beta_cov,X.T))))
    inf_seasonality = np.exp(beta_hat[:-1])
    inf_seasonality_std = np.exp(beta_hat[:-1])*beta_std[:-1]
    alpha = beta_hat[-1]
    alpha_std = beta_std[-1]
    print("alpha = {} +/- {}".format(alpha,2.*alpha_std))

    return neglp, inf_seasonality, inf_seasonality_std, alpha, sig_eps

def fit_the_regularized_model_smooth_rt(state,state_df,state_sias,hood_t,initial_mu_guess=0.5):

    '''This is just like the function above, but the final specification of the transmission term
    forces a smooth r_t over time to illustrate the effect of conventional time-correlation in reporting
    rate estimation. This is NOT used at all in the manuscript, but is a valuable illustration of how different
    correlation structures can affect outputs and model performance on forecast tests. The function is written to
    directly replace fit_the_regularized_model(...) in any script. '''

    ## Then use that estimate of the compartments to
    ## inform seasonality in the state level model.
    neglp = HoodRegularizedModel(
                state_df,
                state_sias,
                state_df["initial_S0"].values[0],
                state_df["initial_S0_var"].values[0],
                hood_t,
                beta_corr=3.,
                tau=24,
                mu_guess=initial_mu_guess)

    ## Fit the model by first finding good SIAS given the
    ## coarse regression approximation to r_t
    x0 = np.ones((neglp.num_sias+1,))
    x0[0] = neglp.logS0_prior
    x0[1:] = neglp.mu
    sia_op = minimize(neglp.fixed_rt,
                      x0=x0,
                      jac=neglp.fixed_rt_grad,
                      method="L-BFGS-B",
                      bounds=[(None,None)]+(len(x0)-1)*[(0,1)],
                      options={"ftol":1e-13,
                               "maxcor":100,
                               },
                      )
    print("\nResult from fixed r_t SIA optimization for just {}:".format(state))
    print("Success = {}".format(sia_op.success))
    neglp.logS0 = sia_op["x"][0]
    neglp.mu = sia_op["x"][1:]

    ## Then adjust the reporting rate given the SIAs
    x0 = np.ones((1+neglp.T+1,))
    x0[0] = neglp.logS0
    x0[1:] = neglp.r_hat
    rep_op = minimize(neglp.fixed_mu,
                      x0=x0,
                      jac=neglp.fixed_mu_grad,
                      method="L-BFGS-B",
                      bounds=[(None,None)]+(len(x0)-1)*[(5.e-4,1)],
                      options={"ftol":1e-13,
                               "maxcor":100,
                               },
                      )
    print("\n...And from fixed SIA r_t optimization")
    print("Success = {}".format(rep_op.success))
    neglp.logS0 = rep_op["x"][0]

    ## Smooth the estimate
    from scipy.signal import savgol_filter
    rt_smooth = savgol_filter(rep_op["x"][1:],49,3)
    neglp.r_hat = rt_smooth #rep_op["x"][1:]

    ## Compute the covariance matrix conditional on the
    ## reporting adjusted estimates
    x0 = np.ones((neglp.num_sias+1,))
    x0[0] = neglp.logS0
    x0[1:] = neglp.mu
    hessian = neglp.fixed_rt_hessian(x0)
    cov = np.linalg.inv(hessian)
    
    ## Finalize the uncertainty estimates
    overall_var = np.diag(cov)
    neglp.logS0_var = overall_var[0]
    neglp.mu_var = overall_var[1:]  

    ## Finally, fully specify the transmission parameters using the
    ## SIA and reporting rate estimates via the optimization above
    adj_cases = ((state_df["cases"].values+1.)/neglp.r_hat)-1.
    adj_births = state_df["adj_births"].values
    adj_sias = (neglp.mu*neglp.sias[:-1]).sum(axis=1)
    E_t = adj_cases[1:]
    I_t = adj_cases[:-1]
    S_t = np.exp(neglp.logS0)+np.cumsum(state_df["adj_births"].values[:-1]-E_t-adj_sias)

    ## Which let's us estimate via a single log-linear regression 
    ## the alpha != 1 model parameters...
    print("\nSpecifying the final transmission term...")
    Y_t = np.log(E_t)-np.log(S_t)
    X = np.hstack([neglp.X[:neglp.T,1:],np.log(I_t)[:,np.newaxis]])
    pRW2 = np.zeros((X.shape[1],X.shape[1]))
    pRW2[:-1,:-1] = neglp.pRW2
    C = np.linalg.inv(np.dot(X.T,X)+pRW2)
    beta_hat = np.dot(C,np.dot(X.T,Y_t))
    beta_t = np.dot(X,beta_hat)
    RSS = np.sum((Y_t-beta_t)**2)
    sig_eps = np.sqrt(RSS/(neglp.T))#-X.shape[1]))
    print("sig_eps = {}".format(sig_eps))
    beta_cov = sig_eps*sig_eps*C
    beta_var = np.diag(beta_cov)
    beta_std = np.sqrt(beta_var)
    beta_t_std = np.sqrt(np.diag(np.dot(X,np.dot(beta_cov,X.T))))
    inf_seasonality = np.exp(beta_hat[:-1])
    inf_seasonality_std = np.exp(beta_hat[:-1])*beta_std[:-1]
    alpha = beta_hat[-1]
    alpha_std = beta_std[-1]
    print("alpha = {} +/- {}".format(alpha,2.*alpha_std))

    return neglp, inf_seasonality, inf_seasonality_std, alpha, sig_eps