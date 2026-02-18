""" VisualizeInputs.py

Script to make a single state plot of the model inputs. """
import sys
import os
import methods

## Standard tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For reference
colors = ["#FF420E","#0078ff","#BF00BA","xkcd:goldenrod","#00ff07","k","#00BF05"]

def axes_setup(axes):
    axes.spines["left"].set_position(("axes",-0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(color="grey",alpha=0.2)
    return

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

    ## Get the epi data from CSV, with some
    ## data type parsing and reshaping into a multi-index
    ## dataframe with state and time at levels 0 and 1 
    ## respectively.
    epi = pd.read_csv(os.path.join("_data",
                      "southern_states_epi_timeseries.csv"),
                      index_col=0,
                      parse_dates=["time"])\
                    .set_index(["state","time"])

    ## And the SIA calendar, via output from SIACalendar.py
    cal = pd.read_csv(os.path.join("_data",
                      "imputed_sia_calendar_by_state.csv"),
                    index_col=0,
                    parse_dates=["start_date","end_date"])

    ## Get the raw age distribution of cases as well
    hist = pd.read_csv(os.path.join("_data",
                       "binned_age_distributions.csv"))\
                    .set_index("state")
    hist.columns = [int(float(c)) for c in hist.columns]

    ## And then get the survey summary statistics
    full_survey = pd.read_csv(os.path.join("_data",
                              "survey_mcv1_summary_stats.csv"),
                              index_col=0,
                              parse_dates=["year"])


    ## Subset to the state of interest
    sf = epi.loc[state].copy()
    sias = cal.loc[cal["state"] == state].copy()
    survey = full_survey.loc[full_survey["state"] == state]
    hist = hist.loc[state]

    ## Set up a plot with multiple, distinctly
    ## sized panels.
    fig = plt.figure(figsize=(12,10))
    gs = fig.add_gridspec(3,3)
    case_ax = fig.add_subplot(gs[0,:-1])
    demo_ax = fig.add_subplot(gs[1,:])
    vacc_ax = fig.add_subplot(gs[2,:])
    age_ax = fig.add_subplot(gs[0,-1])
    axes = [case_ax,demo_ax,vacc_ax,age_ax]

    ## Make the cases panel
    dots = sf.loc[sf["cases"] != 0.,"cases"]
    axes_setup(axes[0])
    axes[0].plot(sf["cases"],color=colors[0])
    axes[0].plot(dots,color=colors[0],
                 marker=".",ls="None")
    axes[0].set_ylim((0,None))
    axes[0].set_ylabel("Daily reports")

    ## Make the births panel
    births = sf[["births","births_var"]].resample("MS").sum()
    births["err"] = np.sqrt(births["births_var"])
    births = births.iloc[:-1]
    axes_setup(axes[1])
    axes[1].fill_between(births.index,
                         (births["births"]-2.*births["err"]).values,
                         (births["births"]+2.*births["err"]).values,
                         facecolor=colors[1],edgecolor="None",alpha=0.3)
    axes[1].plot(births["births"],lw=3,color=colors[1])
    axes[1].set_ylabel("Monthly births")
    
    ## And a vaccination panel
    mcv1 = sf[["mcv1","mcv1_var"]].resample("MS").mean()
    mcv1["err"] = np.sqrt(mcv1["mcv1_var"])
    axes_setup(axes[2])
    axes[2].fill_between(mcv1.index,
                         (mcv1["mcv1"]-2.*mcv1["err"]).values,
                         (mcv1["mcv1"]+2.*mcv1["err"]).values,
                         facecolor=colors[2],edgecolor="None",alpha=0.2)
    axes[2].plot(mcv1["mcv1"],lw=3,color=colors[2])

    ## Add the survey data
    axes[2].errorbar(survey["year"],survey["mcv_est"].values,yerr=survey["mcv_se"],
                     ls="None",color="k",marker="o",markersize=8)
    axes[2].set_ylim((0,1.1))
    axes[2].set_ylabel("Coverage")

    ## Survey type annotations.
    for i, row in survey.iterrows():
        if row.loc["mcv_est"] < mcv1.loc[row.loc["year"],"mcv1"]:
            v_align = "top"
        else:
            v_align = "bottom"
        label = "  "+row.loc["svy_type"]
        axes[2].text(row.loc["year"],row.loc["mcv_est"],label,
                     horizontalalignment="left",verticalalignment=v_align,
                     fontsize=16,color="k",alpha=0.8)

    ## Add MCV2 introduction
    y0, y1 = axes[2].get_ylim()
    mcv2_intro = sf.loc[sf["mcv2"] != 0].index[0]+pd.to_timedelta(15*30.437,unit="d")
    axes[2].axvline(mcv2_intro,
                    ymax=0.4,color="grey",lw=2,ls="dashed")
    axes[2].text(mcv2_intro,0.4*(y1-y0),"MCV2 is \nintroduced ",
                 horizontalalignment="right",verticalalignment="bottom",
                 fontsize=18,color="grey")

    ## Add the SIAs
    sias["dose_frac"] = sias["doses"]/(sias["doses"].max())
    for i, r in sias.iterrows():
        height = 0.35*r.loc["dose_frac"]
        if r.loc["start_date"].year < 2008:
            continue
        axes[2].axvline(r.loc["start_date"],
                        ymax=height,
                        color=colors[3],lw=2)
        if r.loc["doses"] >= 1e6:
            label = "{0:0.1f}M".format(r.loc["doses"]/1e6)
        elif r.loc["doses"] >= 1e3:
            label = "{0:0.0f}k".format(r.loc["doses"]/1e3)
        else:
            label = str(r.loc["doses"])
        #label += " SIA\ndoses"
        axes[2].text(r.loc["start_date"],1.05*height*(y1-y0),label,
                     horizontalalignment="center",verticalalignment="bottom",
                     color=colors[3],fontsize=16)
    axes[2].plot([],color=colors[3],lw=2,label="SIA doses")
    
    ## Add p(age|infection) inset
    axes_setup(axes[3])
    axes[3].grid(color="grey",alpha=0.2)
    axes[3].plot(hist.index,
                hist.values*100,
                color=colors[0],lw=4,drawstyle="steps-post")
    axes[3].set_xticks(hist.index[::2])
    axes[3].set_ylim((0,None))
    axes[3].set_xlabel("Age")
    axes[3].set_ylabel("% of cases")


    ## Finish up, with an optional title.
    #fig.suptitle("Measles data in {}".format(state.title()))
    fig.tight_layout(h_pad=0.,w_pad=0)#rect=[0.0, 0.0, 1, 0.95])

    ## Add the panel labels
    axes[0].text(0,0.95,"Incidence",
                 horizontalalignment="left",verticalalignment="top",rotation=0,
                 fontsize=28,color=colors[0],
                 transform=axes[0].transAxes)
    axes[1].text(0,0.95,"Demography",
                 horizontalalignment="left",verticalalignment="top",rotation=0,
                 fontsize=28,color=colors[1],
                 transform=axes[1].transAxes)
    axes[2].text(0,0.95,"Vaccination",
                 horizontalalignment="left",verticalalignment="top",rotation=0,
                 fontsize=28,color=colors[2],
                 transform=axes[2].transAxes)

    ## Save it, as a png but also if this script is being run
    ## as a subprocess in a generator script, the object itself
    ## is pickled for use elsewhere.
    fig.savefig(os.path.join("_plots","model_inputs.png"))
    if _serialize:
        pickle.dump(fig, 
                    open(os.path.join("pickle_jar",
                         "model_inputs.fig.pickle"),
                         "wb"))
    if not _serialize:
        plt.show()