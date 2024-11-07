""" TransmissionModel.py

This script creates and visualizes transmission model-based estimates for a 
given state. """
import os
import sys

## Standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## The main model class
import methods.neighborhood_sir as nTSIR

## For model fitting
from scipy.optimize import minimize

## For measuring model performance
from sklearn.metrics import r2_score

## For reference throughout
colors = ["#375E97","#FB6542","#FFBB00","#5ca904","xkcd:saffron"]

def axes_setup(axes):
    axes.spines["left"].set_position(("axes",-0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return

def low_mid_high(samples):
    l0 = np.percentile(samples,2.5,axis=0)
    h0 = np.percentile(samples,97.5,axis=0)
    l1 = np.percentile(samples,25.,axis=0)
    h1 = np.percentile(samples,75.,axis=0)
    #m = samples.mean(axis=0)
    m = np.percentile(samples,50.,axis=0) 
    return l0, l1, m, h1, h0

def fit_quality(data,samples,verbose=True):

    ## Compute a summary
    l0,l1,m,h1,h0 = low_mid_high(samples)

    ## Compute scores
    score = r2_score(data,m)
    score50 = len(data[np.where((data >= l1) & (data <= h1))])/len(m)
    score95 = len(data[np.where((data >= l0) & (data <= h0))])/len(m)
    if verbose:
        print("R2 score = {}".format(score))
        print("With 50 interval: {}".format(score50))
        print("With 95 interval: {}".format(score95))
    
    return score, score50, score95

def get_age_pyramid(state,fname=os.path.join("_data","grid3_population_by_state.csv")):

    ## Get the output from geopode
    df = pd.read_csv(fname,index_col=0)\
            .set_index(["state","age_bin"])
    df = df.loc[state].reset_index()
    population = int(np.round(df["total"].sum()))
    
    ## Make an age column, representing the start 
    ## of the age bins
    df["age"] = df["age_bin"].apply(lambda s: int(s.split()[0]))
    df = df.sort_values("age")

    ## Interpolate to a 
    pyramid = df[["age","total"]].set_index("age")["total"]
    pyramid = pyramid.reindex(np.arange(pyramid.index[-1]+5)).fillna(method="ffill")
    pyramid.loc[1:4] = pyramid.loc[1:4]/4
    pyramid.loc[5:] = pyramid.loc[5:]/5

    ## Turn it into a distribution
    pyramid = pyramid/population

    return pyramid, population

def prep_model_inputs(state,time_index,epi,cr,dists):

    ## Start by aggregating the epi data
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

    ## Add a population column
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

    ## Set up vaccination
    df["v1"] = (df["births"]*mcv1_effic*pr_sus_at_mcv1*df["mcv1"]).shift(18).fillna(method="bfill")
    df["v1_var"] = mcv1_effic*pr_sus_at_mcv1*(df["births"]*df["mcv1"]*(1.-df["mcv1"])+\
                    df["mcv1_var"]*(df["births"]**2)+\
                    df["births_var"]*(df["mcv1"]**2)).shift(18).fillna(method="bfill")

    ## Compute immunizations from MCV2
    mcv1_failures = df["v1"]*(1.-mcv1_effic)/mcv1_effic
    mcv1_failures_var = df["v1_var"]*(1.-mcv1_effic)/mcv1_effic
    df["v2"] = (mcv2_effic*df["mcv2"]*pr_sus_at_mcv2*mcv1_failures).shift(30-18).fillna(method="bfill")
    df["v2_var"] = mcv2_effic*pr_sus_at_mcv2*(mcv1_failures*df["mcv2"]*(1.-df["mcv2"])+\
                    df["mcv2_var"]*(mcv1_failures**2)+\
                    mcv1_failures_var*(df["mcv2"]**2)).shift(30-18).fillna(method="bfill")

    ## Construct adjusted births
    df["adj_births"] = df["births"]-df["v1"]-df["v2"]
    df["adj_births_var"] = df["births_var"]+df["v1_var"]+df["v2_var"]

    ## Collect effects besides SIA and initial susceptibility
    df = df.loc["2009-01-01":]
    df["S_t_tilde"] = np.cumsum(df["adj_births"])

    ## And finally compute prior adjusted cases
    df["adj_cases_p"] = (df["cases"]+1.)/df["rr_p"] - 1.

    return df

def prep_sia_effects(cal,time_index):

    ## Get the SIA calendar to collect SIA effects, looping over campaigns
    ## and aligning to the time steps 
    cal = cal.loc[(cal["start_date"] >= time_index[0]) &\
                  (cal["start_date"] <= time_index[-1])]
    cal = cal.sort_values("start_date").reset_index(drop=True)
    cal["time"] = cal["start_date"]+0.5*(cal["end_date"].fillna(cal["start_date"])-cal["start_date"])
    cal["time"] = cal["time"].dt.round("d")
    sia_effects = cal[["time","doses"]].copy()
    sia_effects["time"] = sia_effects["time"].apply(lambda t: np.argmin(np.abs(t-time_index)))
    sia_effects["time"] = time_index[sia_effects["time"].values]
    
    ## Consolidate any overlapping dates, and reshape into one
    ## timeseries per SIA, with the doses at the approporate dates
    sia_effects = sia_effects.groupby("time").sum().reset_index()
    sia_effects = sia_effects.reset_index().rename(columns={"index":"sia_num"})
    sia_effects = sia_effects.pivot(index="time",columns="sia_num",values="doses")
    sia_effects = sia_effects.reindex(time_index).fillna(0)

    return sia_effects

if __name__ == "__main__":

    ## Get the state name
    state = " ".join(sys.argv[1:])
    if state == "":
        state = "lagos"

    ## Process the flags, used by generator scripts
    ## that call this script for every state.
    state = state.split("-")
    if len(state) > 1:
        _serialize = True
        import pickle
    else:
        _serialize = False
    state = state[0].rstrip()

    ## Set the state of interest and
    ## get the associated region, for reference
    s_and_r = pd.read_csv(os.path.join("_data",
                          "states_and_regions.csv"),
                          index_col=0)
    region = s_and_r.loc[s_and_r["state"] == state,
                         "region"].values[0]
    hood = s_and_r.loc[s_and_r["region"] == region,
                       "state"]

    ## Get the epi data from CSV, with some
    ## data type parsing and reshaping.
    epi = pd.read_csv(os.path.join("_data",
                      "southern_states_epi_timeseries.csv"),
                      index_col=0,
                      parse_dates=["time"])\
                    .set_index(["state","time"])
    epi = epi.loc[hood]
    time_index = pd.date_range("2008-01-01","2024-01-31",
                                freq="SMS")

    ## Get the remaining raw data
    ## The SIA calendar
    cal = pd.read_csv(os.path.join("_data",
                      "imputed_sia_calendar_by_state.csv"),
                    index_col=0,
                    parse_dates=["start_date","end_date"]) 
    cal = cal.loc[cal["state"].isin(hood)]

    ## And get the age at infection inferences
    age_dists = pd.read_csv(os.path.join("_data",
                            "southern_age_at_infection.csv"),
                        index_col=0)\
                        .set_index(["region","birth_year","age"])
    dists = age_dists.loc[region,"avg"].unstack()
    dists_var = age_dists.loc[region,"var"].unstack()

    ## Set up the vaccination related parameters, which include
    ## estimates of the probability of survival till MCV1
    mcv1_effic = 0.825
    mcv2_effic = 0.95
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

    ## Then, get the coarse regression results prepared
    ## in the SurvivalRegrGenerator.py
    cr = pd.read_pickle(os.path.join("pickle_jar",
                        "coarse_outputs_by_state.pkl")).loc[hood]

    ## Now, loop over states in the neighborhood to construct
    ## the key modeling inputs state-by-state.
    dfs = {}
    for this_state in hood:
        this_state_df = prep_model_inputs(this_state,
                                          time_index,
                                          epi.loc[this_state],
                                          cr.loc[this_state],
                                          dists)
        dfs[this_state] = this_state_df
    dfs = pd.concat(dfs.values(),keys=dfs.keys())

    ## Create a state df and a hood df
    columns = ["cases","rejected","population",
               "adj_births","adj_births_var",
               "initial_S0","initial_S0_var","S_t_tilde",
               "adj_cases_p"]
    state_df = dfs.loc[state,columns+["rr_p","rr_p_var"]]
    hood_df = dfs.loc[hood,#[s for s in hood if s != state],
                      columns].groupby(level=1).sum()

    ## Make a comparison plot
    fig, axes = plt.subplots(figsize=(12,5))
    axes_setup(axes)
    x = hood_df["cases"].resample("MS").sum()
    axes.fill_between(x.index,0,x.values,
                      facecolor="#000000",edgecolor="None",
                      label="Cases across the {} region".format(region.title()),
                      alpha=0.25)
    axes.plot([],color="#D43790",lw=4,label="Cases in just {} state".format(state.title()))
    axes2 = axes.twinx()
    axes2.spines["right"].set_position(("axes",1.025))
    axes2.spines["top"].set_visible(False)
    axes2.spines["left"].set_visible(False)
    axes2.spines["bottom"].set_visible(False)
    axes2.spines["right"].set_color("#D43790")
    axes2.plot(state_df["cases"].resample("MS").sum(),lw=4,color="#D43790",
              label="Cases in just {} state".format(state.title()))
    axes.set_ylim((0,None))
    axes2.set_ylim((0,None))
    axes.set_ylabel("Regional cases (monthly)")
    axes2.set_ylabel("State-level cases (monthly)",color="#D43790")
    axes2.tick_params(axis="y",colors="#D43790")
    axes.legend(frameon=False,fontsize=22)
    fig.tight_layout()

    ## And associated SIA effects
    state_sias = prep_sia_effects(cal.loc[cal["state"] == state].copy(),
                                  state_df.index)
    hood_sias = prep_sia_effects(cal.copy(),#cal.loc[cal["state"] != state].copy(),
                                 hood_df.index)

    ## Start by solving the neighborhood problem, to create regularizing
    ## inputs for the state level model.
    initial_guess = {"south west":0.1,"south east":0.1,
                     "south south":0.1}
    hoodP = nTSIR.NeighborhoodPosterior(hood_df,
                                        hood_sias,
                                        hood_df["initial_S0"].values[0],
                                        hood_df["initial_S0_var"].values[0],
                                        beta_corr=3.,
                                        tau=24,
                                        mu_guess=initial_guess.get(region,.15),
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
          " region = ".format(region.title()))
    print(set(hood))
    print("Success = {}".format(sia_op.success))
    hoodP.logS0 = sia_op["x"][0]
    hoodP.mu = sia_op["x"][1:]
    hood_t = hoodP.compartment_df()

    ## Then use that estimate of the compartments to
    ## inform seasonality in the state level model.
    initial_guess = {"abia":0.6,"ebonyi":0.6,
                     "enugu":0.6,"imo":0.6,"akwa ibom":0.4,
                     "bayelsa":0.4,"cross river":0.2,"delta":0.4,
                     "rivers":0.4,"edo":0.2,"ekiti":0.2,"ogun":0.9,
                     }
    neglp = nTSIR.HoodRegularizedModel(state_df,
                                    state_sias,
                                    state_df["initial_S0"].values[0],
                                    state_df["initial_S0_var"].values[0],
                                    hood_t,
                                    beta_corr=3.,
                                    tau=24,
                                    mu_guess=initial_guess.get(state,0.8)*hoodP.mu.mean())

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
    sia_op_cov = sia_op["hess_inv"].todense()
    sia_op_std = np.sqrt(np.diag(sia_op_cov))

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
    rep_op_cov = rep_op["hess_inv"].todense()
    rep_op_std = np.sqrt(np.diag(rep_op_cov))

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

    ## Summarize the results
    print("\nSo we find...")
    print("Initial log S0 = {} +/- {}".format(neglp.logS0_prior,np.sqrt(neglp.logS0_prior_var)))
    print("After SIA op log S0 = {} +/- {}".format(sia_op["x"][0],sia_op_std[0]))
    print("After reporting op log S0 = {} +/- {}".format(rep_op["x"][0],rep_op_std[0]))
    print("Overall log S0 = {} +/- {}".format(neglp.logS0,np.sqrt(neglp.logS0_var)))
    print("\nFor SIAs:")
    for i in state_sias.columns:
        d = state_sias[i].loc[state_sias[i] != 0].index[0]
        print("{}: {} SIA: {} +/- {} ({} immunized, {} doses)".format(i,d,neglp.mu[i],np.sqrt(neglp.mu_var[i]),
                                                            int(neglp.mu[i]*state_sias[i].sum()),int(state_sias[i].sum())))
    inf_sia_tot = (neglp.mu*neglp.sias).sum()
    reached_tot = state_sias.sum().sum()
    implied_effic = 100*inf_sia_tot/reached_tot
    print("Total immunized in SIAs "+\
          "= {0:.0f} (of {1:.0f}, aka {2:.2f} percent overall)".format(inf_sia_tot,reached_tot,implied_effic))

    ## Use the result to finish model specification
    adj_cases = ((state_df["cases"].values+1.)/neglp.r_hat)-1.
    adj_births = state_df["adj_births"].values
    adj_sias = (neglp.mu*neglp.sias[:-1]).sum(axis=1)
    E_t = adj_cases[1:]
    I_t = adj_cases[:-1]
    S_t = np.exp(neglp.logS0)+np.cumsum(state_df["adj_births"].values[:-1]-E_t-adj_sias)

    ## Fit the transmission rate model
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

    ## Sample the fitted model
    num_samples = 10000
    eps_t = np.exp(sig_eps*np.random.normal(size=(num_samples,len(state_df))))
    S0_samples = np.random.normal(np.exp(neglp.logS0),
                                   np.exp(neglp.logS0)*np.sqrt(neglp.logS0_var),
                                   size=(num_samples,))
    adj_births_samples = np.random.multivariate_normal(adj_births,
                                                       np.diag(state_df["adj_births_var"].values),
                                                       size=(num_samples,))
    traj_long = np.zeros((num_samples,2,len(state_df)))
    traj_long[:,0,0] = S0_samples
    traj_long[:,1,0] = I_t[0]   
    traj_short = np.zeros((num_samples,2,len(state_df)))
    traj_short[:,0,0] = S0_samples
    traj_short[:,1,0] = I_t[0]
    for t in range(1,len(state_df)):

        ## Get the transmission rate
        beta = inf_seasonality[(t-1)%neglp.tau]

        ## Compute the force of infection in each case
        lam_long = beta*traj_long[:,0,t-1]*(traj_long[:,1,t-1]**alpha)
        lam_short = beta*traj_short[:,0,t-1]*(I_t[t-1]**alpha)

        ## Incorporate uncertainty across samples
        traj_long[:,1,t] = lam_long*eps_t[:,t-1]
        traj_long[:,0,t] = traj_long[:,0,t-1]+adj_births_samples[:,t-1]-adj_sias[t-1]-traj_long[:,1,t]
        traj_short[:,1,t] = lam_short*eps_t[:,t-1]
        traj_short[:,0,t] = traj_short[:,0,t-1]+adj_births_samples[:,t-1]-adj_sias[t-1]-traj_short[:,1,t]

        ## Regularize for the 0 boundary 
        traj_long[:,:,t] = np.clip(traj_long[:,:,t],0.,None)
        traj_short[:,:,t] = np.clip(traj_short[:,:,t],0.,None)

    ## Sample to get estimates of observed cases
    cases_short = np.random.binomial(np.round(traj_short[:,1,:]).astype(int),
                                     p=rep_op["x"][1:])
    cases_long = np.random.binomial(np.round(traj_long[:,1,:]).astype(int),
                                     p=rep_op["x"][1:])

    ## Test the goodness of fit
    print("\nGoodness of fit to cases (short term):")
    fit_quality(state_df["cases"].values,cases_short)
    print("\nGoodness of fit to cases (long term):")
    fit_quality(state_df["cases"].values,cases_long)

    ## To make everything per pop, get the necessary information
    _, population = get_age_pyramid(state)

    ## Make it all per pop
    traj_long = 100*traj_long/population
    traj_short = 100*traj_short/population

    ## Summarize the result
    long_low, _, long_mid, _, long_high = low_mid_high(traj_long)
    short_low, _, short_mid, _, short_high = low_mid_high(traj_short)
    short_c_low, _, short_c_mid, _, short_c_high = low_mid_high(cases_short)
    long_c_low, _, long_c_mid, _, long_c_high = low_mid_high(cases_long)

    ## Plot the results
    fig, axes = plt.subplots(3,1,sharex=False,figsize=(14,10))
    for ax in axes:
        axes_setup(ax)
    axes[0].spines["left"].set_color("grey")
    axes[0].fill_between(state_df.index,short_c_low,short_c_high,
                         alpha=0.4,facecolor="grey",edgecolor="None")
    axes[0].plot(state_df.index,short_c_mid,color="grey",lw=3,label="Model fit")
    axes[0].plot(state_df.index,state_df["cases"],color="k",ls="None",
                 marker=".",markerfacecolor="k",markeredgecolor="k",markersize=10,
                 label="Reported cases in {}".format(state.title()))
    axes[0].set_ylim((0,None))
    axes[0].set_ylabel("Measles cases",color="grey",labelpad=15)
    axes[0].tick_params(axis="y",colors="grey")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1],labels[::-1],
                   frameon=False,fontsize=18)

    ## Twin the axes for prevalence
    p_ax = axes[0].twinx()
    p_ax.spines["right"].set_position(("axes",1.025))
    p_ax.spines["top"].set_visible(False)
    p_ax.spines["left"].set_visible(False)
    p_ax.spines["bottom"].set_visible(False)
    p_ax.spines["right"].set_color(colors[1])
    p_ax.fill_between(state_df.index,short_low[1],short_high[1],
                        alpha=0.3,facecolor=colors[1],edgecolor="None",
                        label=r"X$_t$|X$_{t-1}$")
    p_ax.plot(state_df.index,short_mid[1],color=colors[1],lw=3)
    p_ax.set_ylim((0,None))
    p_ax.set_ylabel("Prevalence (%)",color=colors[1],labelpad=15)
    p_ax.tick_params(axis="y",colors=colors[1])

    ## Plot the susceptibility panel
    axes[1].spines["left"].set_color(colors[0])
    axes[1].fill_between(state_df.index,short_low[0],short_high[0],
                         alpha=0.3,facecolor=colors[0],edgecolor="None")
    axes[1].plot(state_df.index,short_mid[0],color=colors[0],lw=3)
    y0,y1 = axes[1].get_ylim()
    N = state_sias.max().max()
    for i in state_sias.columns:
        d = state_sias[i].loc[state_sias[i] != 0].index[0]
        n = int(neglp.mu[i]*state_sias[i].sum())
        height = 0.15*(1)# + n/N)
        axes[1].axvline(d,ymax=height,color=colors[4],lw=3)
        if n >= 1e6:
            label = "{0:0.1f}M".format(n/1e6)
        elif n >= 1e3:
            label = "{0:0.0f}k".format(n/1e3)
        else:
            label = str(n)
        axes[1].text(d,1.025*height*(y1-y0) + y0,label,
                     horizontalalignment="right",verticalalignment="bottom",
                     color=colors[4],fontsize=18)
    axes[1].plot([],lw=3,color=colors[4],label="SIA (number immunized)")
    axes[1].legend(loc=2,frameon=False,fontsize=18)
    axes[1].set_ylabel("Susceptibility (%)",color=colors[0])
    axes[1].tick_params(axis="y",colors=colors[0])

    ## Make the reporting panel
    axes[2].spines["left"].set_color(colors[3])
    std = np.sqrt(state_df["rr_p_var"])
    axes[2].fill_between(state_df.index,
                         100.*(state_df["rr_p"]-2.*std),
                         100.*(state_df["rr_p"]+2.*std),
                         facecolor="grey",edgecolor="None",alpha=0.1)
    axes[2].fill_between(state_df.index,
                         100.*(state_df["rr_p"]-std),
                         100.*(state_df["rr_p"]+std),
                         facecolor="grey",edgecolor="None",alpha=0.2,
                         label="Our expectation based on\nthe age distribution of cases")
    axes[2].plot(state_df.index,100.*state_df["rr_p"],color="grey",ls="dashed",lw=3)
    axes[2].plot(state_df.index,100*neglp.r_hat,color=colors[3],lw=2)
    axes[2].set_ylabel("Reporting rate (%)",color=colors[3])
    axes[2].tick_params(axis="y",colors=colors[3])
    axes[2].legend(frameon=False,fontsize=18,loc=2)
    axes[2].set_ylim((0,None))
    fig.tight_layout()
    fig.savefig(os.path.join("_plots","model_overview.png"))
    if _serialize:
        pickle.dump(fig, 
                    open(os.path.join("pickle_jar",
                         "model_overview.fig.pickle"),
                         "wb"))

    Sbar = population*(short_mid[0].mean()/100.)
    inf_seasonality_std = inf_seasonality_std/(inf_seasonality.mean())
    inf_seasonality = inf_seasonality/(inf_seasonality.mean())
    fig, axes = plt.subplots(figsize=(9,5))
    axes_setup(axes)
    axes.grid(color="grey",alpha=0.2)
    axes.fill_between(np.arange(len(inf_seasonality)),
                      (inf_seasonality-inf_seasonality_std),
                      (inf_seasonality+inf_seasonality_std),
                      facecolor="#87235E",edgecolor="None",alpha=0.3,zorder=2)
    axes.plot(inf_seasonality,color="#87235E",lw=5,zorder=3,label="Transmission")
    axes.set_ylabel(r"Rel. transmission rate, $\beta_t/\langle\beta_t\rangle$")
    axes.set_xticks(np.arange(0,12*2,2))
    axes.set_xticklabels([dt.strftime("%b") for dt in state_df.index[0:24:2]],
                         fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join("_plots","model_seasonality.png"))

    ## Done
    if not _serialize:
        plt.show()