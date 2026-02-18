""" BirthSeasonality.py

State-by-state seasonality estimates. """
import sys
import survey

## For filepaths
import os

## I/O functionality is built on top
## of pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For evaluating the posterior and
## fitting the model.
import survey.buckets as sb

## For making PDFs
from matplotlib.backends.backend_pdf import PdfPages

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

## Survey I/O
def get_a_dhs(path,columns,convert_categoricals=True,add_survey=False):
	df = pd.read_stata(path,
					   columns=columns.keys(),
					   convert_categoricals=convert_categoricals)
	df["caseid"] = df["caseid"].str.strip()

	## Rename according to the schema
	df.columns = [columns[c] for c in df.columns]

	if add_survey:
		df["survey"] = path.split(os.path.sep)[1].lower()
	return df

def get_a_mics(path,columns,convert_categoricals=True,add_survey=False):
	df = pd.read_spss(path,
					  usecols=columns.keys(),
					  convert_categoricals=convert_categoricals)

	## Rename according to the schema
	df.columns = [columns[c] for c in df.columns]

	if add_survey:
		df["survey"] = path.split(os.path.sep)[1].lower()
	return df

## DHS related schemas, which map a file path (recode) to the columns
## needed from the recode.
## Birth recode
br_schema = {
	os.path.join("_surveys","DHS5_2008","NGBR53DT","NGBR53FL.DTA"):
		{
		 "caseid":"caseid", ## mom id
		 "b3":"child_DoB", #CMC DoB of child
		 "v011":"mom_DoB", #CMC DoB of mom
		 "v008":"interview_date", #Interview CMC date
		 "v023":"strata", ## sample strata (state, or region+state+U/R)
		 "v005":"weight", #Mom's sample weight
		},
	os.path.join("_surveys","DHS6_2013","NGBR6ADT","NGBR6AFL.DTA"):
		{
		 "caseid":"caseid", ## mom id
		 "b3":"child_DoB", #CMC DoB of child
		 "v011":"mom_DoB", #CMC DoB of mom
		 "v008":"interview_date", #Interview CMC date
		 "v023":"strata", ## sample strata (state, or region+state+U/R)
		 "v005":"weight", #Mom's sample weight
		},
	os.path.join("_surveys","DHS7_2018","NGBR7ADT","NGBR7AFL.DTA"):
		{
		 "caseid":"caseid", ## mom id
		 "b3":"child_DoB", #CMC DoB of child
		 "v011":"mom_DoB", #CMC DoB of mom
		 "v008":"interview_date", #Interview CMC date
		 "v023":"strata", ## sample strata (state, or region+state+U/R)
		 "v005":"weight", #Mom's sample weight
		},
}

## MICS related schemas, which map a file path (recode) to the columns
## needed from the recode.
## Birth recode
bh_schema = {
	os.path.join("_surveys","MICS5_2016","bh.sav"):
		{
		 "HH1":"cluster", ## woman cluster
		 "HH2":"hh", ## woman hh
		 "LN":"line_num", ## line number
		 "BHLN":"birth_ln", ## birth history ln
		 "HH7":"state", ## state 
		 "BH4M":"child_birth_mon", ## child birth month
		 "BH4Y":"child_birth_year", ## child birth year
		},
	os.path.join("_surveys","MICS6_2021","bh.sav"):
		{
		 "HH1":"cluster", ## woman cluster
		 "HH2":"hh", ## woman hh
		 "LN":"line_num", ## line number
		 "BHLN":"birth_ln", ## birth history ln
		 "HH7":"state", ## state  
		 "BH4M":"child_birth_mon", ## child birth month
		 "BH4Y":"child_birth_year", ## child birth year
		},
}

## For reference
num_to_month = {1:"January",2:"February",3:"March",4:"April",
				5:"May",6:"June",7:"July",8:"August",9:"September",
				10:"October",11:"November",12:"December"}

if __name__ == "__main__":

	## Get the data via pandas, first the DHS
	brs = {path.split(os.path.sep)[1].lower()\
			:get_a_dhs(path,columns,True,True) 
			for path, columns in br_schema.items()}
	dhs = pd.concat(brs.values(),axis=0)\
			.reset_index(drop=True)
	
	## Now shift over to the MICS datasets
	bhs = {path.split(os.path.sep)[1].lower()\
			:get_a_mics(path,columns,True,True) 
			for path, columns in bh_schema.items()}
	mics = pd.concat(bhs.values(),axis=0)\
			.reset_index(drop=True)

	## Create a column for the state
	state_regex = r"^[nsNS][ewcsEWCS]\s(.*)\s\b(?:urban|rural|Urban|Rural)\b"
	dhs["state"] = dhs["strata"].str.extract(state_regex)[0].str.lower()
	dhs["state"] = dhs["state"].fillna(dhs["strata"]).str.replace("fct abuja","abuja")
	mics["state"] = mics["state"].str.lower().str.replace("fct abuja","abuja").str.replace("fct","abuja")

	## Get a birthdate column, and subset to recent
	## births within the survey period.
	dhs["birth_date"] = cms_to_datetime(dhs["child_DoB"])
	dhs = dhs.loc[(dhs["birth_date"] >= "2005-01-01")\
				& (dhs["birth_date"] <= "2019-12-31")].reset_index(drop=True)
	dhs["birth_year"] = dhs["birth_date"].dt.year
	dhs["birth_month"] = dhs["birth_date"].dt.month#strftime("%B").str.lower()

	## And same for the mics
	mics.loc[mics["child_birth_year"].apply(lambda x: isinstance(x, str)),
			 "child_birth_year"] = np.nan
	mics["child_birth_mon"] = mics["child_birth_mon"].str.lower()\
								.replace("no response",np.nan)\
								.replace("missing",np.nan)\
								.replace("inconsistent",np.nan)\
								.map({"january":1,"february":2,"march":3,"april":4,
									  "may":5,"june":6,"july":7,"august":8,
									  "september":9,"october":10,"november":11,"december":12})
	mics["birth_date"] = pd.to_datetime({"month":mics["child_birth_mon"],
									  "year":mics["child_birth_year"],
									  "day":1})
	mics = mics.loc[(mics["birth_date"] >= "2005-01-01")\
				& (mics["birth_date"] <= "2019-12-31")].reset_index(drop=True)
	mics["birth_year"] = mics["birth_date"].dt.year
	mics["birth_month"] = mics["birth_date"].dt.month#strftime("%B").str.lower()

	## Put the full dataset together
	columns = ["survey","state",
			   "birth_date","birth_year","birth_month"]
	df = pd.concat([dhs[columns],
					mics[columns]],axis=0).reset_index(drop=True)
	print("\nFull dataset:")
	print(df)

	## Start a pdf document, with one page per state.
	output = {}
	with PdfPages("_plots\\birth_seasonality_by_state.pdf") as pdf:

		## Starting the loop through states
		print("\nStarting the loop through states...")
		for state, sf in df.groupby("state"):

			## Compute the monthly fraction
			monthly = sf[["birth_year","birth_month"]].copy()
			monthly["frac"] = np.ones((len(sf),),dtype=np.int32)
			monthly = monthly.groupby(["birth_year","birth_month"]).sum()["frac"]
			monthly = monthly.unstack(level=1).fillna(0)
			total_by_year = monthly.sum(axis=1)
			monthly_frac = monthly.div(total_by_year,axis=0)

			## Set up the buckets posterior
			buckets = sb.BinomialPosterior(monthly,
										   correlation_time=5.,
										   g2g_correlation=3.)
			result = sb.FitModel(buckets)
			print("for {}, success = {}".format(state,result.success))

			## Unpack the result
			samples = sb.SampleBuckets(result,buckets)
			low = np.percentile(samples,2.5,axis=0)
			high = np.percentile(samples,97.5,axis=0)
			mid = samples.mean(axis=0)
			var = samples.var(axis=0)

			## Create outputs for use elsewhere
			mid_df = pd.DataFrame(mid,
								  columns=monthly_frac.columns,
								  index=monthly_frac.index).stack().rename("avg")
			var_df = pd.DataFrame(var,
								  columns=monthly_frac.columns,
								  index=monthly_frac.index).stack().rename("var")
			this_output = pd.concat([mid_df,var_df],axis=1)
			output[state] = this_output

			## Make a plot
			fig, axes = plt.subplots(3,4,sharex=True,sharey=True,figsize=(16,9))
			axes = axes.reshape(-1)
			for ax in axes:
				axes_setup(ax)
				ax.grid(color="grey",alpha=0.2)
			for i in monthly_frac.columns:

				## Plot the model
				axes[i-1].fill_between(monthly_frac.index,
									   low[:,i-1],high[:,i-1],
									   facecolor="grey",edgecolor="None",alpha=0.4,label="Model")
				axes[i-1].plot(monthly_frac.index,mid[:,i-1],
							   lw=2,color="grey")

				## Plot the data
				axes[i-1].plot(monthly_frac[i],
							   ls="dashed",lw=3,color="k",
							   markersize=10,marker="o",label="DHS survey")
				axes[i-1].text(0.01,0.99,num_to_month[i],
							   horizontalalignment="left",verticalalignment="top",
							   fontsize=22,color="xkcd:red wine",
							   transform=axes[i-1].transAxes)
				if (i-1)%4 == 0:
					axes[i-1].set_ylabel("Probability")
					#axes[i-1].set_ylim((0.055,0.125))
					#axes[i-1].set_yticks([0.06,0.08,0.1,0.12])
				if i == 4:
					axes[i-1].legend(frameon=False,loc=1,fontsize=16)

			## Finish up
			fig.suptitle("Seasonality in "+state.title())
			fig.tight_layout(rect=[0, 0.0, 1, 0.9])
			if state == "jigawa":
				print("..........saving this one!")
				fig.savefig("_plots\\birth_seasonality_ex.png")
			pdf.savefig(fig)
			plt.close(fig)

		## Set up metadata
		d = pdf.infodict()
		d['Title'] = "Birth seasonality in Nigeria"
		d['Author'] = "Niket"

	## Finish up
	print("...finished.")

	## Create the full output
	output = pd.concat(output.values(),keys=output.keys())
	print("\nFinal output:")
	print(output)
	output.to_pickle("pickle_jar\\birth_seasonality_by_state.pkl")


		