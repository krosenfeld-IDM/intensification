""" SIAImpactPosteriors.py

A script that fits the model (just as in TransmissionModel.py) but computes the posterior
profiles of the catch-up vaccination parameters as opposed to time series from the model. """
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

## Get some colors related to SIA age range
colors = {"9-23M":"#5ca904", 
          "12-23M":"#EB116A",
          "9-59M":"#116AEB",
          "6M-10Y":"#6AEB11",
          "9M-15Y":"grey",
          "6M-9Y":"#11EB92"}

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
    state_sias, sia_metadata = nSIR.prep_sia_effects(cal.loc[cal["state"] == state].copy(),
                                		state_df.index,md=True)
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

	## Construct the full parameter vector and evaluate
    ## the negative log posterior at that value.
    x0 = np.zeros((1+neglp.num_sias+neglp.T+1,))
    x0[0] = neglp.logS0
    x0[1:neglp.num_sias+1] = neglp.mu
    x0[neglp.num_sias+1:] = neglp.r_hat
    map_NLP = neglp(x0)

    ## Set up the grid of possible campaign efficacy values, 
    ## representing the fraction of delivered doses that immunized
    ## a susceptible person in the model, and set aside some storage
    ## for posterior profiles (X). 
    possible_mu = np.linspace(0.,1.,800)
    X = np.vstack(len(possible_mu)*[x0])

    ## Set up a big figure
    num_rows = 1+int((neglp.num_sias-1)/3)
    fig, axes = plt.subplots(num_rows,3,sharex=True,figsize=(12,num_rows*3))
    axes = axes.reshape(-1)
    for ax in axes:
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="grey",alpha=0.2)

	## Loop over SIAs, plotting the posterior profile
    ## for each.
    xlim = (-0.025,0.3333)
    xlabel = "Per dose efficacy"
    dists = np.zeros((len(sia_metadata),len(possible_mu)))
    for i in range(neglp.num_sias):

        ## Set up the inputs
        this_X = X.copy()
        this_X[:,1+i] = possible_mu

        ## Compute the distribution, ignoring underflow errors
        ## at very unlikely outcomes.
        with np.errstate(invalid="ignore"):
            this_dist = np.array([np.exp(-neglp(x)+map_NLP) for x in this_X])
        this_dist = np.nan_to_num(this_dist,0)
        this_dist *= 1./np.sum(this_dist)

        ## Store it
        dists[i] = this_dist

        ## Then gather the SIA metadata, including an 
        ## annotation for plotting.
        d = sia_metadata.loc[i,"time"]
        adj_reach = sia_metadata.loc[i,"doses"]
        if adj_reach < 1e6:
            reach = str(int(np.around(adj_reach,-3)/1000))
            unit = "k"
        else:
            reach = str(np.around(adj_reach,-4)/1e6)
            unit = "M"
        age_range = sia_metadata.loc[i,"age_group"]

        ## Plot the result
        axes[i].fill_between(possible_mu,0,this_dist,
                             facecolor=colors[age_range],edgecolor="None",alpha=0.3)
        axes[i].plot(possible_mu,this_dist,
                     lw=4,color=colors[age_range])
        
        ## Add the annotation.
        axes[i].text(0.99,0.01,"{}\n{}{} doses\n{}".format(
                        d.strftime("%b, %Y").replace("Jun","June"),reach,unit,age_range.replace("-"," to ")
                        ),
                     fontsize=22,color="k",#"#bf209f",
                     horizontalalignment="right",verticalalignment="bottom",
                     transform=axes[i].transAxes)

        ## Details
        axes[i].set_yticks([])
        axes[i].set_ylim((0,None))
        axes[i].set_xlim(xlim)
        axes[i].set_xticks(np.linspace(0,0.3,4))
        if i > 3*(num_rows-1)-1:
            axes[i].set_xlabel(xlabel)
        if i % 3 == 0:
            axes[i].set_ylabel("Probability")
    
    ## Clean up
    for ax in axes[i+1:]:
        ax.axis("off")

    ## Finish up
    fig.tight_layout()
    fig.savefig(os.path.join("_plots","sia_conditional_dists.png"))

    ## If you're saving estimates...
    if _serialize:
        sia_metadata["region"] = neglp.num_sias*[region]
        sia_metadata["avg"] = neglp.mu
        sia_metadata["std"] = np.sqrt(neglp.mu_var)
        sia_metadata["dist"] = [d for d in dists]
        sia_metadata.to_pickle(os.path.join("pickle_jar",
            "{}_sias.pkl"\
            .format(state.replace(" ","_"))))
        pickle.dump(fig, 
                    open(os.path.join("pickle_jar",
                         "sia_conditional_dists.fig.pickle"),
                         "wb"))

    ## Otherwise you're done.
    if not _serialize:
        plt.show()