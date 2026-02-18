""" TransmissionModel.py

This script creates and visualizes transmission model-based estimates for a 
given state. """
import os
import sys

## Standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## The main model class and fitting functions.
import methods.neighborhood_sir as nSIR

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

if __name__ == "__main__":

    ## Get the state name from command line input
    ## otherwise stick with a default.
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
    ## data type parsing and reshaping into a multiindex
    ## dataframe with state and time as levels.
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

    ## And get the age at infection inferences, which are 
    ## based on observed ages of cases - see the discussion in the
    ## manuscript's appendix 2.
    age_dists = pd.read_csv(os.path.join("_data",
                            "southern_age_at_infection.csv"),
                        index_col=0)\
                        .set_index(["region","birth_year","age"])
    dists = age_dists.loc[region,"avg"].unstack()
    dists_var = age_dists.loc[region,"var"].unstack()

    ## Then, get the coarse regression results prepared
    ## in the SurvivalRegrGenerator.py
    cr = pd.read_pickle(os.path.join("pickle_jar",
                        "coarse_outputs_by_state.pkl")).loc[hood]

    ## Now, loop over states in the neighborhood to construct
    ## the key modeling inputs state-by-state.
    dfs = {}
    for this_state in hood:
        this_state_df = nSIR.prep_model_inputs(
                            this_state,
                            time_index,
                            epi.loc[this_state],
                            cr.loc[this_state],
                            dists)
        dfs[this_state] = this_state_df
    dfs = pd.concat(dfs.values(),keys=dfs.keys())

    ## Create a state df and a neighborhood df by subsetting
    ## and aggregating as appropriate.
    columns = ["cases","rejected","population",
               "adj_births","adj_births_var",
               "initial_S0","initial_S0_var","S_t_tilde",
               "adj_cases_p"]
    state_df = dfs.loc[state,columns+["rr_p","rr_p_var"]]
    hood_df = dfs.loc[hood,#[s for s in hood if s != state],
                      columns].groupby(level=1).sum()

    ## Make a comparison plot of the data from each.
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

    ## Reshape the state and neighborhood campaign effects from a table
    ## of metadata to a set of timeseries with non-zero entries at the
    ## time of the campaign.
    state_sias = nSIR.prep_sia_effects(cal.loc[cal["state"] == state].copy(),
                                  state_df.index)
    hood_sias = nSIR.prep_sia_effects(cal.copy(),hood_df.index)

    ## Start by solving the neighborhood problem, to create regularizing
    ## inputs for the state level model. See the discussion around 
    ## Appendix 2's equations 3 and 4 for details.
    hoodP = nSIR.fit_the_neighborhood_model(region,hood_df,hood_sias)

    ## Then use that estimate of the compartment populations to
    ## inform seasonality in the state level model. Note that we need
    ## some care in the initial-guess on campaign effects to prevent the
    ## BFGS line search from selecting infeasible values and crashing.
    initial_guess = {"abia":0.6,"anambra":0.9,
                     "enugu":0.5,"imo":0.6,
                     "bayelsa":0.4,"cross river":0.2,"delta":0.4,
                     "rivers":0.4,"edo":0.2,"ekiti":0.2,"ogun":0.9,
                     }
    neglp, inf_seasonality, inf_seasonality_std, alpha, sig_eps = \
        nSIR.fit_the_regularized_model(
                    state,state_df,state_sias,
                    hoodP.compartment_df(),
                    initial_guess.get(state,0.8)*hoodP.mu.mean())

    ## Summarize some of the inferences...
    print("\nFor the SIAs we have:")
    for i in state_sias.columns:
        d = state_sias[i].loc[state_sias[i] != 0].index[0]
        print("{}: {} SIA: {} +/- {} ({} immunized, {} doses)".format(i,d,neglp.mu[i],np.sqrt(neglp.mu_var[i]),
                                                            int(neglp.mu[i]*state_sias[i].sum()),int(state_sias[i].sum())))
    inf_sia_tot = (neglp.mu*neglp.sias).sum()
    reached_tot = state_sias.sum().sum()
    implied_effic = 100*inf_sia_tot/reached_tot
    print("Total immunized in SIAs "+\
          "= {0:.0f} (of {1:.0f}, aka {2:.2f} percent overall)".format(inf_sia_tot,reached_tot,implied_effic))

    ## Sample the fitted model by presampling the transmission volatility
    ## effect (eps_t), the initial conditions, and the source term, then
    ## looping over time. traj_long samples without reference to the data,
    ## traj_short samples in one step ahead increments.
    num_samples = 10000
    eps_t = np.exp(sig_eps*np.random.normal(size=(num_samples,len(state_df))))
    S0_samples = np.random.normal(np.exp(neglp.logS0),
                                   np.exp(neglp.logS0)*np.sqrt(neglp.logS0_var),
                                   size=(num_samples,))
    adj_births_samples = np.random.multivariate_normal(state_df["adj_births"].values,
                                                       np.diag(state_df["adj_births_var"].values),
                                                       size=(num_samples,))
    adj_cases = ((state_df["cases"].values+1.)/neglp.r_hat)-1.
    adj_sias = (neglp.mu*neglp.sias[:-1]).sum(axis=1)
    I_t = adj_cases[:-1]
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

    ## Sample to get estimates of observed cases from the
    ## incidence trajectories.
    cases_short = np.random.binomial(np.round(traj_short[:,1,:]).astype(int),
                                     p=neglp.r_hat)
    cases_long = np.random.binomial(np.round(traj_long[:,1,:]).astype(int),
                                     p=neglp.r_hat)

    ## Test the goodness of fit relative to observed
    ## cases, in terms of R^2 score.
    print("\nGoodness of fit to cases (short term):")
    fit_quality(state_df["cases"].values,cases_short)
    print("\nGoodness of fit to cases (long term):")
    fit_quality(state_df["cases"].values,cases_long)

    ## Save the estimates of S_t and I_t if we're 
    ## serializing output.
    if _serialize:
        st_df = pd.DataFrame(np.array([traj_short[:,0,:].mean(axis=0),
                                       traj_short[:,0,:].var(axis=0),
                                       traj_short[:,1,:].mean(axis=0),
                                       traj_short[:,1,:].var(axis=0)]).T,
                             columns=["Savg","Svar","Iavg","Ivar"],
                             index=state_df.index)
        st_df.to_pickle(os.path.join("pickle_jar",
                        "{}_traj.pkl".format(state.replace(" ","_"))))

    ## Make it all per population for plotting
    ## purposes.
    traj_long = 100*traj_long/(state_df["population"].values[None,None,:])
    traj_short = 100*traj_short/(state_df["population"].values[None,None,:])

    ## Summarize the result in terms of percentiles.
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

    Sbar = (state_df["population"].mean())*(short_mid[0].mean()/100.)
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