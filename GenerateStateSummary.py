""" GenerateStateSummary.py

This script takes about 20 min to run on a laptop, but generates a PDF of model 
outputs by state, calling VisualizeInputs.py, TranssmissionModel.py, OutOfSampleTest.py, 
and SIAImpactPosteriors.py as subprocesses, then compiling the serialized results. 

It also compiles SIA posterior distributions into a serialized dataset for analysis."""
import os
import sys
import methods

## Standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For making a PDF of plots
from matplotlib.backends.backend_pdf import PdfPages

## For launching the modeling scripts for a given state.
import pickle
import subprocess

if __name__ == "__main__":

    ## Get the list of southern states
    s_and_r = pd.read_csv(os.path.join("_data",
                          "states_and_regions.csv"),
                          index_col=0).sort_values(["region","state"])
    states = s_and_r.loc[s_and_r["region"].str.startswith("south"),
                        "state"].reset_index(drop=True)

    ## Loop over states, fit models, compile results,
    ## and create a book of plots.
    sia_dists = {}
    with PdfPages(os.path.join("_plots","state_models_summary.pdf")) as book:
        
        ## Loop over states
        print("\nLooping over states...")
        for i, state in states.iteritems():

            print("Model construction for {} ({}/{})...".format(state.upper(),i+1,len(states)))
            subprocess.run("python VisualizeInputs.py {} -s".format(state))
            model_out =subprocess.run("python TransmissionModel.py {} -s".format(state),capture_output=True,text=True)
            test_out = subprocess.run("python OutOfSampleTest.py {} -s".format(state),capture_output=True,text=True)
            post_out = subprocess.run("python SIAImpactPosteriors.py {} -s".format(state),capture_output=True,text=True)

            ## Check for convergence across the three model scripts (fit to different time
            ## windows in some cases)
            if "Success = False" in model_out.stdout:
                raise RuntimeError("{} full time model didn't converge!".format(state))
            if "Success = False" in test_out.stdout:
                raise RuntimeError("{} out of sample testing model didn't converge!".format(state))
            if "Success = False" in post_out.stdout:
                raise RuntimeError("{} efficacy posterior model didn't converge!".format(state))

            ## Retrieve the figures
            input_fig = pickle.load(open(os.path.join("pickle_jar","model_inputs.fig.pickle"),"rb"))
            model_fig = pickle.load(open(os.path.join("pickle_jar","model_overview.fig.pickle"),"rb"))
            test_fig = pickle.load(open(os.path.join("pickle_jar","out_of_sample.fig.pickle"),"rb"))
            dists_fig = pickle.load(open(os.path.join("pickle_jar","sia_conditional_dists.fig.pickle"),"rb"))
            
            ## Make pages
            ## The input page
            input_fig.text(0.01,0.01,state.title(),
                     ha="left",va="bottom",fontsize=36)
            book.savefig(input_fig)
            plt.close(input_fig)

            ## Model overview page
            model_fig.text(0.01,0.01,state.title(),
                     ha="left",va="bottom",fontsize=36)
            book.savefig(model_fig)
            plt.close(model_fig)

            ## Out of sample page
            test_fig.text(0.01,0.01,state.title(),
                     ha="left",va="bottom",fontsize=36)
            book.savefig(test_fig)
            plt.close(test_fig)

            ## Risk assessment page
            dists_fig.text(0.01,0.01,state.title(),
                     ha="left",va="bottom",fontsize=36)
            book.savefig(dists_fig)
            plt.close(dists_fig)

            ## Finally get the SIA efficacy posterior, which you read in
            ## and compile with the other estimates, then delete the temporary
            ## file associated with this state.
            file = os.path.join("pickle_jar",
                "{}_sias.pkl".format(state.replace(" ","_")))
            this_state_dists = pd.read_pickle(file)
            sia_dists[state] = this_state_dists
            os.remove(file)

        ## Set some PDF metadata
        d = book.infodict()
        d["Title"] = "Measles transmission models for S. Nigeria"
        d["CreationDate"] = pd.to_datetime("today")
    
    ## Put it all together and save
    output = pd.concat(sia_dists.values(),axis=0)\
             .reset_index(drop=True)
    print("\nSaving the output:")
    print(output)
    output.to_pickle(
        os.path.join("pickle_jar","sia_dists_by_state.pkl")
        )