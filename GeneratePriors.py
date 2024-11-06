""" PriorGenerator.py

This script calls SurvivalPrior.py as a subprocess for every state, stores the outputs for
each states, and compiles them into a single serialized object for use in the fast time
scale models. """
import os
import sys
import methods

## Standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For making a PDF of plots
from matplotlib.backends.backend_pdf import PdfPages

## For launching the SurvivalPrior.py script.
import pickle
import subprocess

if __name__ == "__main__":

    ## Set how output is displayed
    _verbose = False

    ## Get the list of southern states
    s_and_r = pd.read_csv(os.path.join("_data",
                          "states_and_regions.csv"),
                          index_col=0).sort_values(["region","state"])
    states = s_and_r.loc[s_and_r["region"].str.startswith("south"),
                        "state"].reset_index(drop=True)

    ## Loop over states, fit prior models, compile results,
    ## and create a book of plots.
    prior_dfs = {}
    with PdfPages(os.path.join("_plots","prior_summary.pdf")) as book:
        
        ## Loop over states
        print("\nLooping over states...")
        for i, state in states.iteritems():

            print("Prior construction for {} ({}/{})...".format(state.upper(),i+1,len(states)))
            subprocess.run("python SurvivalPrior.py {} -s".format(state),
                    capture_output=~_verbose)

            ## Get and store the output dataframe
            prior_dfs[state] = pd.read_pickle(os.path.join("pickle_jar","survival_prior.pkl"))

            ## Get and store the output plot, modifying to include
            ## the state's name.
            fig = pickle.load(open(
                        os.path.join("pickle_jar","survival_prior.fig.pickle"), 
                        'rb'))
            fig.text(0.02,0.02,state.title(),
                     ha="left",va="bottom",fontsize=48)

            ## Save it to the pdf book.
            book.savefig(fig)
            plt.close(fig)

        ## Set some PDF metadata
        d = book.infodict()
        d["Title"] = "Cohort-scale measles immunity in S. Nigeria"
        d["CreationDate"] = pd.to_datetime("today")
        
    ## Put the outputs together
    output = pd.concat(prior_dfs.values(),keys=prior_dfs.keys())
    print("\nCoarse regression estimates (to pickle):")
    print(output)
    output.to_pickle(os.path.join("pickle_jar","coarse_outputs_by_state.pkl"))