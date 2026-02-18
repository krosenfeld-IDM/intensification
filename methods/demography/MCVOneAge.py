""" MCVOneAge.py 

A KDE approach to estimate the age at which mcv is administered in practice in 
Nigeria. """
import sys
import survey

## For filepaths
import os

## I/O functionality is built on top
## of pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For making PDFs
from matplotlib.backends.backend_pdf import PdfPages

## For density estimates
from scipy.stats import gaussian_kde as kde

## For reference
colors = ["#375E97","#FB6542","#FFBB00","#3F681C"]

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def cms_to_datetime(a,d=15):
	assert (a >= 1).all(), "CMS format must be >= 1"
	years = 1900+((a-1)/12).astype(int)
	months = a - 12*(years-1900)
	dates = pd.to_datetime({"year":years,
							"month":months,
							"day":d})
	return dates

def get_a_dhs(path,columns,convert_categoricals=True,add_survey=False):
	df = pd.read_stata(path,
					   columns=columns,
					   convert_categoricals=convert_categoricals)
	df["caseid"] = df["caseid"].str.strip()
	if add_survey:
		df["survey"] = path.split(os.path.sep)[1].lower()
	return df

if __name__ == "__main__":

	## Set the paths to the relevant survey data,
	## Starting with the DHS kid recodes
	krs = [
		   os.path.join("_surveys","DHS5_2008","NGKR53DT","NGKR53FL.DTA"),
		   os.path.join("_surveys","DHS6_2013","NGKR6ADT","NGKR6AFL.DTA"),
		   os.path.join("_surveys","DHS7_2018","NGKR7ADT","NGKR7AFL.DTA"),
		   ]

	## And the associated individual recodes
	irs = [
		   os.path.join("_surveys","DHS5_2008","NGIR53DT","NGIR53FL.DTA"),
		   os.path.join("_surveys","DHS6_2013","NGIR6ADT","NGIR6AFL.DTA"),
		   os.path.join("_surveys","DHS7_2018","NGIR7ADT","NGIR7AFL.DTA"),
		   ]

	## What columns do you need from each?
	kr_columns = [
				  "caseid", ## mom id
				  "bord", ## child birth order
				  "b3", #CMC DoB of child
				  "b5", # is the childe alive?
				  "v011", #CMC DoB of mom
				  "v008", #Interview CMC date
				  "h9", # measles vaccine 1
				  "h1", # has a vaccine card
				  "h9d", ## Recieved Measles 1
				  "h9m", ## Recieved Measles 1
				  "h9y", ## Recieved Measles 1
				  ]
	ir_columns = [
				  "caseid", ## mom id
				  "v149", ## mom's educational attainment
				  "v023", ## sample strata (state, or region+state+U/R)
				  "v024", ## region
				  "v025", ## urban/rural
				  "v224", ## number of entries in br recode
				  "v013", ## age in 5 year bins
				  "v005", #Mom's sample weight
				  ]

	## Get the data via pandas, krs then irs
	krs = [get_a_dhs(path,kr_columns,True,True) for path in krs]
	irs = [get_a_dhs(path,ir_columns,True,False) for path in irs]

	## Merge them and put it all together
	for i, ir in enumerate(irs):
		krs[i] = krs[i].merge(ir,
					on="caseid",
					how="left",
					validate="m:1",
					)
	krs = pd.concat(krs,axis=0).reset_index(drop=True)
	print("\nFull dataset:")
	print(krs)

	## Subset to children alive and within the 12 to 35 month
	## age range
	krs["b5"] = krs["b5"].str.lower()
	krs["age"] = krs["v008"] - krs["b3"]
	krs = krs.loc[(krs["b5"] == "yes") &\
				  (krs["age"] >= 12) &\
				  (krs["age"] < 36)]


	## Process the vaccination history into a simpler column
	## for use in modeling.
	krs["h9"] = krs["h9"].str.lower()
	interpretation = { # based on krs["h9"].value_counts(dropna=False)
					  "no":0,
					  "reported by mother":1,
					  "vaccination date on card":2,
					  "vacc. date on card":2,
					  "vaccination marked on card":2,
					  "dk":0,
					  np.nan:0,
					  "vacc. marked on card":2,
					  "don't know":0
					  }
	krs["mcv"] = krs["h9"].apply(interpretation.get)

	## Take a look at country wide vaccination age where
	## you have the data
	krs["mcv_date"] = pd.to_datetime({"year":krs["h9y"],
									  "month":krs["h9m"],
									  "day":krs["h9d"]},
									  errors="coerce")
	krs["bday"] = cms_to_datetime(krs["b3"])
	krs["mcv_age"] = (krs["mcv_date"]-krs["bday"]).dt.days/30.4167

	## Plot the distribution
	obs_ages = krs.loc[krs["mcv_date"].notnull(),"mcv_age"].values
	pdf = kde(obs_ages)
	age_range = np.linspace(2,24,1000)
	pdf_est = pdf(age_range)
	mode_age_i = np.argmax(pdf_est)
	mode_pdf, mode_age = pdf_est[mode_age_i], age_range[mode_age_i]
	print("\nWe find (from {0:d} entries) that "
		  "MCV age = {1:0.5f} months".format(len(obs_ages),mode_age))
	fig, axes = plt.subplots(figsize=(8,6))
	axes.spines["left"].set_visible(False)
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	axes.fill_between(age_range,0,pdf_est,
					  facecolor="grey",edgecolor="None",alpha=0.8)
	axes.plot([mode_age],[mode_pdf],
			  color="k",lw=3,
			  marker="o",markeredgecolor="k",markerfacecolor="None",
			  markersize=15,markeredgewidth=2)
	axes.plot([mode_age*1.05,mode_age*1.4],[mode_pdf,mode_pdf],
			  color="k",lw=3,ls="dashed")
	axes.text(mode_age*1.4,mode_pdf," {0:0.2f} months".format(mode_age),
			  color="k",fontsize=22,
			  horizontalalignment="left",verticalalignment="center")
	axes.set_xlabel("Age at measles vaccination (months)")
	#axes.set_ylabel("Probability")
	axes.set_yticks([])
	axes.set_ylim((0,None))
	fig.tight_layout()
	fig.savefig("_plots\\mcv1_age.png")
	plt.show()

