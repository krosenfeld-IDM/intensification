""" SIAImpactAnalysis.py

This script takes output from GenerateStateSummary.py, specifically the dataframes of
posterior distributions and hidden states, and uses them to create some estimates of the
2019 IRIs impact relative to the campaigns. """
import os
import sys
import methods

## Standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For custom log-scale ticks
import matplotlib.ticker

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

if __name__ == "__main__":

    ## Get the data from pickle_jar
    ## created via the posterior generator script
    ## GenerateStateSummary.py. 
    dists = pd.read_pickle(
        os.path.join("pickle_jar","sia_dists_by_state.pkl"),
        )
    mu = np.linspace(0,1,
        len(dists["dist"].loc[0])
        )
    
    ## Get the susceptibility esimates as well
    st = pd.read_pickle(
        os.path.join("pickle_jar","hidden_states_by_state.pkl"),
        )

    ## Get the states and regions for 
    ## reference
    s_and_r = pd.read_csv(os.path.join("_data",
                          "states_and_regions.csv"),
                          index_col=0).sort_values(["region","state"])
    states = s_and_r.loc[s_and_r["region"].str.startswith("south"),
                        "state"].reset_index(drop=True)
    
    ## Subset to southern campaigns
    dists = dists.loc[dists["state"].isin(states)]
    st = st.loc[states.values]

    ## Estimate some summary stats, which overwrite the
    ## those estimated in the optimization - in this case, they
    ## account for the 0 to 1 limit, and so are slightly modified.
    dists["avg"] = dists["dist"].apply(lambda a: np.sum(a*mu))
    dists["std"] = dists["dist"].apply(lambda a: np.sum(a*mu*mu))
    dists["std"] = np.sqrt(dists["std"] - dists["avg"]**2)
    
    ## Set up an overall summary plot to make the manuscript's
    ## Figure 5.
    fig = plt.figure(figsize=(11.5,8))
    gs = fig.add_gridspec(3,3)
    scat = fig.add_subplot(gs[:-1,:-1])
    ax1 = fig.add_subplot(gs[0,-1])
    ax2 = fig.add_subplot(gs[1,-1])
    s_ax = fig.add_subplot(gs[-1,:])

    ## Start with the scatter of state-intervention pairs, with
    ## colors by age target.
    axes_setup(scat)
    scat.grid(color="grey",alpha=0.2)

    ## Wide age first
    df = dists.loc[dists["age_group"] == "9-59M"]
    scat.errorbar(df["doses"],
                  df["avg"],yerr=2.*df["std"],
                  color=colors["9-59M"],alpha=0.9,
                  marker="o",lw=1,markersize=9,
                  ls="None",
                  label="9 to 59M campaigns")

    ## Then narrow age
    df = dists.loc[dists["age_group"] == "9-23M"]
    scat.errorbar(df["doses"],
                  df["avg"],yerr=2.*df["std"],
                  color=colors["9-23M"],alpha=0.9,
                  marker="o",lw=1,markersize=9,
                  ls="None",
                  label="9 to 23M campaigns")

    ## Finish up with custom ticks.
    scat.set_xscale("log")
    scat.set_xticks([2e5,4e5,6e5,1e6,2e6,3e6])
    scat.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    scat.set_xticklabels([r"$0.2$",r"$0.4$",r"$0.6$",r"$1$",r"$2$",r"$3$"])
    scat.set(xlabel=r"Doses delivered $(\times 10^6)$",
             ylabel="Per dose efficacy",
             )
    
    ## Add the first dist
    ## First compute the mixture distribution
    df = dists.loc[dists["age_group"] == "9-59M"]
    w = df["doses"]/(df["doses"].sum())
    dist = (df["dist"]*w.values).sum()
    avg = (mu*dist).sum()

    ## Then plot it
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.grid(color="grey",alpha=0.2)
    ax1.plot(mu,dist,
             lw=4,color=colors["9-59M"])
    ax1.fill_between(mu,0,dist,
                     facecolor=colors["9-59M"],edgecolor="None",
                     alpha=0.3)
    scat.plot([df["doses"].min(),df["doses"].max()],
            [avg,avg],
            ls="dashed",lw=4,color=colors["9-59M"],alpha=0.6,
            )
    ax1.set_yticks([])
    ax1.set_xlim((-0.025,0.4025))
    ax1.set_xticks([0,0.1,0.2,0.3,0.4])
    ax1.set_ylim((0,None))
    ax1.set_xlabel("Per dose efficacy")
    ax1.text(0.97,0.97,"9 to 59M\ncampaigns",
             ha="right",va="top",
             color=colors["9-59M"],fontsize=22,
             transform=ax1.transAxes)

    ## Add the 2019 IRI dist
    ## Compute the mixture...
    df = dists.loc[dists["age_group"] == "9-23M"]
    w = df["doses"]/(df["doses"].sum())
    dist = (df["dist"]*w.values).sum()
    avg2019 = (mu*dist).sum()

    ## And plot it
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.grid(color="grey",alpha=0.2)
    ax2.plot(mu,dist,
             lw=4,color=colors["9-23M"])
    ax2.fill_between(mu,0,dist,
                     facecolor=colors["9-23M"],edgecolor="None",
                     alpha=0.3)
    scat.plot([df["doses"].min(),df["doses"].max()],
            [avg2019,avg2019],
            ls="dashed",lw=4,color=colors["9-23M"],alpha=0.6,
            )
    ax2.set_yticks([])
    ax2.set_xlim((-0.025,0.4025))
    ax2.set_xticks([0,0.1,0.2,0.3,0.4])
    ax2.set_ylim((0,None))
    ax2.set_xlabel("Per dose efficacy")
    ax2.text(0.97,0.97,"9 to 23M\nIRIs",
             ha="right",va="top",
             color=colors["9-23M"],fontsize=22,
             transform=ax2.transAxes)

    ## Add overall susceptibility
    st = st.groupby(level=1).sum()
    st["Serr"] = np.sqrt(st["Svar"])
    st["Ierr"] = np.sqrt(st["Ivar"])
    s_ax.spines["top"].set_visible(False)
    s_ax.spines["right"].set_visible(False)
    s_ax.spines["left"].set_visible(False)
    s_ax.fill_between(st.index,
                      (st["Savg"]-2*st["Serr"]).values,
                      (st["Savg"]+2*st["Serr"]).values,
                      facecolor="k",edgecolor="None",alpha=0.3)
    s_ax.plot(st["Savg"],
              color="k",lw=3)

    ## Add campaign timeline as little bars.
    ylim = s_ax.get_ylim()
    sias = set(zip(dists["age_group"],
                   dists["time"]))
    for ag, t in sias:
        s_ax.axvline(t,
                     ymax=1./3.,
                     color=colors[ag],lw=3)
    s_ax.set_yticks([])
    s_ax.set_ylabel("Susceptibility")
    s_ax.set_ylim((0.8*ylim[0],ylim[1]))

    ## Done
    fig.tight_layout()
    fig.savefig(os.path.join("_plots","sia_analysis.png"))

    ## Print some stats of performance averaged over states.
    print("\nThe avg across campaigns is {}".format(avg))
    print("The avg for the IRI is {}".format(avg2019))
    print("So the 2019 doses were {} times as effective".format(avg2019/avg))

    ## Done.
    plt.show()