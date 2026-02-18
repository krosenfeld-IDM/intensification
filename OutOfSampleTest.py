""" OutofSampleTest.py

This script fits the model to a portion of the dataset and then tests performance against the
withheld portion. """
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

## Specify the cutoff date for the data
out_of_sample_date = "2020-12-31"

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
    
    ## Reshape the state and neighborhood campaign effects from a table
    ## of metadata to a set of timeseries with non-zero entries at the
    ## time of the campaign.
    state_sias = nSIR.prep_sia_effects(cal.loc[cal["state"] == state].copy(),
                                  state_df.index)
    hood_sias = nSIR.prep_sia_effects(cal.copy(),hood_df.index)

    ## Split the data into within (_tr for training) and out of 
    ## sample components
    state_tr = state_df.loc[:out_of_sample_date]
    hood_tr = hood_df.loc[:out_of_sample_date]
    state_sias_tr = state_sias.loc[:out_of_sample_date]
    state_sias_tr = state_sias_tr.loc[:,(state_sias_tr != 0).any(axis=0)]
    hood_sias_tr = hood_sias.loc[:out_of_sample_date]
    hood_sias_tr = hood_sias_tr.loc[:,(hood_sias_tr != 0).any(axis=0)]

    ## Start by solving the neighborhood problem, to create regularizing
    ## inputs for the state level model. See the discussion around 
    ## Appendix 2's equations 3 and 4 for details.
    hoodP = nSIR.fit_the_neighborhood_model(region,hood_tr,hood_sias_tr)

    ## Then use that estimate of the compartments to
    ## inform seasonality in the state level model. Note that we need
    ## some care in the initial-guess on campaign effects to prevent the
    ## BFGS line search from selecting infeasible values and crashing.
    initial_guess = {
                     "enugu":0.6,"imo":0.7,"akwa ibom":0.9,
                     "oyo":0.9,"lagos":0.9,"ogun":1.,"kano":0.9,
                     "ekiti":0.25,"anambra":0.9,
                     }
    neglp, inf_seasonality, inf_seasonality_std, alpha, sig_eps = \
        nSIR.fit_the_regularized_model(
                    state,state_tr,state_sias_tr,
                    hoodP.compartment_df(),
                    initial_guess.get(state,0.8)*hoodP.mu.mean())

    ## Now prepare to extrapolate, first by preparing future campaign
    ## effects based on past performance.
    print("\nFull SIA calendar:")
    state_cal = cal.loc[cal["state"] == state].copy()
    print(state_cal)
    fitted_sias = state_cal.loc[(state_cal["start_date"] >= state_tr.index[0]) &
                                (state_cal["end_date"] <= state_tr.index[-1])]\
                         .reset_index(drop=True)

    ## Then the SIAs, which takes a little care, specifically to
    ## remove the targeted 2019 IRI effect so that averages are just
    ## based on the appropriate events.
    extrap_mu = np.ones((len(state_sias.columns),))
    extrap_mu[:len(state_sias_tr.columns)] = neglp.mu
    adj_sias = extrap_mu*state_sias
    avg_impact = adj_sias[fitted_sias.loc[fitted_sias["age_group"] == "9-59M"].index]
    avg_impact = avg_impact.sum(axis=1).sum()/(len(avg_impact.columns))
    for c in state_sias.columns[len(state_sias_tr.columns):]:
        adj_sias.loc[adj_sias[c] != 0,c] = avg_impact
    full_adj_sias = adj_sias.values[:-1].sum(axis=1)

    ## Also specify the full time source term, by periodically forward filling
    ## based on the adj. births you have, thus preserving the seasonality in births
    ## as you extrapolate.
    full_adj_births = state_tr["adj_births"].reindex(state_df.index)
    ab_profile = state_tr.loc[state_tr.index[-24:],"adj_births"].values
    full_adj_births = full_adj_births.fillna(
            pd.Series(ab_profile[np.arange(len(state_df)-len(state_tr))%24],
                      index=state_df.index[len(state_tr):])).values

    ## Set up the full reporting rate as well, extrapolating
    ## at the prior level.
    full_rr = pd.Series(neglp.r_hat,
                   index=state_tr.index,name="rr")
    full_rr = full_rr.reindex(state_df.index).fillna(
                state_tr["rr_p"].values[-1]
                )

    ## Sample the fitted model on the full timescale, set by state
    ## df, not by the training data frame.
    num_samples = 10000
    adj_cases = ((state_tr["cases"].values+1.)/neglp.r_hat)-1.
    I_t = adj_cases[:-1]
    eps_t = np.exp(sig_eps*np.random.normal(size=(num_samples,len(state_df))))
    S0_samples = np.random.normal(np.exp(neglp.logS0),
                                   np.exp(neglp.logS0)*np.sqrt(neglp.logS0_var),
                                   size=(num_samples,))
    adj_births_samples = np.random.multivariate_normal(full_adj_births,
                                                       np.diag(state_df["adj_births_var"].values),
                                                       size=(num_samples,))
    trajectories = np.zeros((num_samples,2,len(state_df)))
    trajectories[:,0,0] = S0_samples
    trajectories[:,1,0] = I_t[0]
    for t in range(1,len(state_df)):

        ## Get the transmission rate
        beta = inf_seasonality[(t-1)%neglp.tau]

        ## Compute the force of infection in each case, using the data when
        ## you have it, but otherwise being self-referential
        if t < len(state_tr):
            lam = beta*trajectories[:,0,t-1]*(I_t[t-1]**alpha)
        else:
            lam = beta*trajectories[:,0,t-1]*(trajectories[:,1,t-1]**alpha)

        ## Incorporate uncertainty across samples
        trajectories[:,1,t] = lam*eps_t[:,t-1]
        trajectories[:,0,t] = trajectories[:,0,t-1]+\
                                adj_births_samples[:,t-1]-\
                                full_adj_sias[t-1]-\
                                trajectories[:,1,t]

        ## Regularize for the 0 boundary 
        trajectories[:,:,t] = np.clip(trajectories[:,:,t],0.,None)

    ## Sample to get estimates of observed cases
    model_cases = np.random.binomial(np.round(trajectories[:,1,:]).astype(int),
                                     p=full_rr.values)

    ## Test the goodness of fit
    t1 = len(state_tr)
    print("\nGoodness of fit to cases (training):")
    fit_quality(state_df["cases"].values[:t1],model_cases[:,:t1])
    print("\nGoodness of fit to cases (testing):")
    fit_quality(state_df["cases"].values[t1:],model_cases[:,t1:])

    ## Make it all per pop for plotting.
    trajectories = 100*trajectories/(state_df["population"].values[None,None,:])

    ## Summarize the results
    traj_low, _, traj_mid, _, traj_high = low_mid_high(trajectories)
    cases_low, _, cases_mid, _, cases_high = low_mid_high(model_cases)

    ## Plot the results
    fig, axes = plt.subplots(2,1,sharex=False,figsize=(14,7))
    for ax in axes:
        axes_setup(ax)
    axes[0].spines["left"].set_color("k")
    axes[0].fill_between(state_df.index[:t1],
                         cases_low[:t1],
                         cases_high[:t1],
                         alpha=0.4,facecolor="grey",edgecolor="None")
    axes[0].plot(state_df.index[:t1],cases_mid[:t1],color="grey",lw=3,
                 label="Model fit to data from {}".format(state.title()),
                 )
    axes[0].fill_between(state_df.index[t1:],
                         cases_low[t1:],
                         cases_high[t1:],
                         alpha=0.4,facecolor=colors[1],edgecolor="None")
    axes[0].plot(state_df.index[t1:],cases_mid[t1:],color=colors[1],lw=3,
                 label="Model forecast",
                 )
    axes[0].plot(state_df.index[:t1],state_df["cases"].values[:t1],
                 color="grey",ls="None",
                 marker=".",markerfacecolor="grey",markeredgecolor="grey",markersize=10,
                 label="Measles cases used for fitting")
    axes[0].plot(state_df.index[t1:],state_df["cases"].values[t1:],
                 color="k",ls="None",
                 marker=".",markerfacecolor="k",markeredgecolor="k",markersize=12,
                 label="Measles cases witheld for testing")
    axes[0].set_ylim((0,None))
    axes[0].set_ylabel("Reported cases",color="k",labelpad=15)
    axes[0].tick_params(axis="y",colors="k")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles,labels,
                   frameon=False,fontsize=18,
                   loc=2)

    ## Plot the susceptibility panel
    axes[1].spines["left"].set_color("k")
    axes[1].fill_between(state_df.index[:t1],
                         traj_low[0][:t1],
                         traj_high[0][:t1],
                         alpha=0.3,facecolor="grey",edgecolor="None",
                         label="Model fit to data from {}".format(state.title())
                         )
    axes[1].plot(state_df.index[:t1],traj_mid[0][:t1],color="grey",lw=3)
    axes[1].fill_between(state_df.index[t1:],
                         traj_low[0][t1:],
                         traj_high[0][t1:],
                         alpha=0.3,facecolor=colors[0],edgecolor="None",
                         label="Model forecast",
                         )
    axes[1].plot(state_df.index[t1:],traj_mid[0][t1:],color=colors[0],lw=3)
    axes[1].set_ylabel("Susceptibility (%)",color="k")
    axes[1].tick_params(axis="y",colors="k")
    #axes[1].legend(frameon=False,fontsize=18)

    ## Finish up
    #axes[1].set_xlim((pd.to_datetime("2016-01-01"),None))
    fig.tight_layout()
    fig.savefig(os.path.join("_plots","out_of_sample_test.png"))
    if _serialize:
        pickle.dump(fig, 
                    open(os.path.join("pickle_jar",
                         "out_of_sample.fig.pickle"),
                         "wb"))
    if not _serialize:
        plt.show()

    