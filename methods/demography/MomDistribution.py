""" MomDistribution.py

Using the DHS weights to coarsely estimate the fraction of individuals in each
demographic cell (i.e. mom-characteristic combination). """
import sys
import survey

## For filepaths
import os

## I/O functionality is built on top
## of pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_a_dhs(path,columns,convert_categoricals=True):
	df = pd.read_stata(path,
					   columns=columns,
					   convert_categoricals=convert_categoricals)
	df["caseid"] = df["caseid"].str.strip()
	df["survey"] = path.split(os.path.sep)[1].lower()
	df["v005"] *= 1e-6
	df["v005"] *= 1./(df["v005"].sum())
	return df

if __name__ == "__main__":

	## Set the paths to the relevant survey data,
	## the associated individual recodes
	irs = [
		   os.path.join("_surveys","DHS5_2008","NGIR53DT","NGIR53FL.DTA"),
		   os.path.join("_surveys","DHS6_2013","NGIR6ADT","NGIR6AFL.DTA"),
		   os.path.join("_surveys","DHS7_2018","NGIR7ADT","NGIR7AFL.DTA"),
		   ]

	## What columns do you need from each?
	ir_columns = [
				  "caseid", ## mom id
				  "v011", ## mom's DoB in CMC
				  "v008", ## interview data in CMC
				  "v013", ## age in 5 year bins
				  "v023", ## sample strata (state, or region+state+U/R)
				  "v024", ## region
				  "v025", ## urban/rural
				  "v106", ## mom's educational attainment (v106 is low res, 149 is high)
				  "v224", ## number of entries in br recode
				  "v005", ## mom's sample weight
				  ]

	## Get the data via pandas
	irs = [get_a_dhs(path,ir_columns,True) for path in irs]

	## Put them all together
	irs = pd.concat(irs,axis=0).reset_index(drop=True)
	print("\nFull dataset:")
	print(irs)

	## Create a year covariate from the interview date
	irs["year"] =irs["survey"].str.slice(start=5).astype(np.int64)

	## Create a column for the state
	state_regex = r"^[nsNS][ewcsEWCS]\s(.*)\s\b(?:urban|rural|Urban|Rural)\b"
	irs["state"] = irs["v023"].str.extract(state_regex)[0].str.lower()
	irs["state"] = irs["state"].fillna(irs["v023"]).str.replace("fct abuja","abuja")

	## Summarize the possibilities
	print("\nVariable values...")
	for c in ["v013","v024","state","v025","v106","year"]:
		values = sorted(irs[c].unique())
		print("{} values = {}".format(c,values))

	## Create a table of demographic cells
	df = irs[["state","v025","v106","v013","v005","year"]].copy()
	df = df.groupby([c for c in df.columns if c != "v005"]).sum().fillna(0)["v005"]

	## Interpolate over time
	years = np.arange(2008,2022,dtype=np.int64)
	df = df.groupby(["state","v025","v106","v013"]).apply(lambda s: s.loc[s.name].reindex(years).interpolate())
	df = df.reset_index()
	
	## Print and save
	print("\nTotal weight per cell:")
	print(df)
	df.to_pickle("pickle_jar\\mom_distribution.pkl")

	## Check the normalization
	print("\nNormalization check:")
	print(df[["year","v005"]].groupby("year").sum())

