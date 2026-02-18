""" AgeAtKthKid.py

Log-normal regression to estimate a Mom's age at the time of their
child's birth. """
import sys
import survey

## For filepaths
import os

## I/O functionality is built on top
## of pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For model fitting
import survey.ridge as rr

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
		 "bord":"bord", ## child birth order
		 "b3":"child_DoB", #CMC DoB of child
		 "v011":"mom_DoB", #CMC DoB of mom
		 "v008":"interview_date", #Interview CMC date
		 "v005":"weight", #Mom's sample weight
		},
	os.path.join("_surveys","DHS6_2013","NGBR6ADT","NGBR6AFL.DTA"):
		{
		 "caseid":"caseid", ## mom id
		 "bord":"bord", ## child birth order
		 "b3":"child_DoB", #CMC DoB of child
		 "v011":"mom_DoB", #CMC DoB of mom
		 "v008":"interview_date", #Interview CMC date
		 "v005":"weight", #Mom's sample weight
		},
	os.path.join("_surveys","DHS7_2018","NGBR7ADT","NGBR7AFL.DTA"):
		{
		 "caseid":"caseid", ## mom id
		 "bord":"bord", ## child birth order
		 "b3":"child_DoB", #CMC DoB of child
		 "v011":"mom_DoB", #CMC DoB of mom
		 "v008":"interview_date", #Interview CMC date
		 "v005":"weight", #Mom's sample weight
		},
}

## Individual recode
ir_schema = {
	os.path.join("_surveys","DHS5_2008","NGIR53DT","NGIR53FL.DTA"):
		{
		 "caseid":"caseid", ## mom id
		 "v106":"mom_edu", ## mom's educational attainment (v106 is low res, 149 is high)
		 "v023":"strata", ## sample strata (state, or region+state+U/R)
		 "v024":"region", ## region
		 "v025":"area", ## urban/rural
		},
	os.path.join("_surveys","DHS6_2013","NGIR6ADT","NGIR6AFL.DTA"):
		{
		 "caseid":"caseid", ## mom id
		 "v106":"mom_edu", ## mom's educational attainment (v106 is low res, 149 is high)
		 "v023":"strata", ## sample strata (state, or region+state+U/R)
		 "v024":"region", ## region
		 "v025":"area", ## urban/rural
		},
	os.path.join("_surveys","DHS7_2018","NGIR7ADT","NGIR7AFL.DTA"):
		{
		 "caseid":"caseid", ## mom id
		 "v106":"mom_edu", ## mom's educational attainment (v106 is low res, 149 is high)
		 "v023":"strata", ## sample strata (state, or region+state+U/R)
		 "v024":"region", ## region
		 "v025":"area", ## urban/rural
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
		 "birthord":"bord", ## child birth order
		 "BH4M":"child_birth_mon", ## child birth month
		 "BH4Y":"child_birth_year", ## child birth year
		},
	os.path.join("_surveys","MICS6_2021","bh.sav"):
		{
		 "HH1":"cluster", ## woman cluster
		 "HH2":"hh", ## woman hh
		 "LN":"line_num", ## line number
		 "BHLN":"birth_ln", ## birth history ln
		 "brthord":"bord", ## child birth order
		 "BH4M":"child_birth_mon", ## child birth month
		 "BH4Y":"child_birth_year", ## child birth year
		},
}

## women's recode
wm_schema = {
	os.path.join("_surveys","MICS5_2016","wm.sav"):
		{
		 "HH1":"cluster", ## woman cluster
		 "HH2":"hh", ## woman hh
		 "LN":"line_num", ## line number
		 "WM6D":"interview_day", ## interview day
		 "WM6M":"interview_mon", ## interview month
		 "WM6Y":"interview_year", ## interview year
		 "WB1M":"mom_birth_mon", ## mom's birth month
		 "WB1Y":"mom_birth_year", ## mom's birth year
		 "WB4":"mom_edu", ## mom's education
		 "HH6":"area", ## urban/rural
		 "HH7":"state", ## state 
		 "Zone":"region", ## region
		 "wmweight":"weight", ## mom's sample weight
		},
	os.path.join("_surveys","MICS6_2021","wm.sav"):
		{
		 "HH1":"cluster", ## woman cluster
		 "HH2":"hh", ## woman hh
		 "LN":"line_num", ## line number
		 "WM6D":"interview_day", ## interview day
		 "WM6M":"interview_mon", ## interview month
		 "WM6Y":"interview_year", ## interview year
		 "WB3M":"mom_birth_mon", ## mom's birth month
		 "WB3Y":"mom_birth_year", ## mom's birth year
		 "WB6A":"mom_edu", ## mom's education
		 "HH6":"area", ## urban/rural
		 "HH7":"state", ## state 
		 "zone":"region", ## region
		 "wmweight":"weight", ## mom's sample weight
		},
}
	
if __name__ == "__main__":

	## Get the data via pandas, brs then irs
	brs = {path.split(os.path.sep)[1].lower()\
			:get_a_dhs(path,columns,False,True) 
			for path, columns in br_schema.items()}
	irs = {path.split(os.path.sep)[1].lower():\
			get_a_dhs(path,columns,True,False)
			for path, columns in ir_schema.items()}
	
	## Merge them and put it all together
	dhs = []
	for k in brs.keys():
		this_dhs = brs[k].merge(irs[k],
					on="caseid",
					how="left",
					validate="m:1",
					)
		dhs.append(this_dhs)
	dhs = pd.concat(dhs,axis=0)
	dhs = dhs.sort_values(["survey","caseid","bord"]).reset_index(drop=True)
	print("\nThe DHS data for this analysis:")
	print(dhs)
	
	## Now shift over to the MICS datasets, starting with the
	## birth recodes
	bhs = {path.split(os.path.sep)[1].lower()\
			:get_a_mics(path,columns,True,True) 
			for path, columns in bh_schema.items()}
	wms = {path.split(os.path.sep)[1].lower()\
			:get_a_mics(path,columns,True,False) 
			for path, columns in wm_schema.items()}

	## Merge them and put it all together
	mics = []
	for k in bhs.keys():
		this_mics = bhs[k].merge(wms[k],
					on=["cluster","hh","line_num"],
					how="left",
					validate="m:1",
					)
		mics.append(this_mics)
	mics = pd.concat(mics,axis=0).reset_index(drop=True)
	print("\nThe MICS data for this analysis:")
	print(mics)

	## Create a year covariate from the interview date for both
	## The dhs and the mics
	dhs["year"] = cms_to_datetime(dhs["interview_date"]).dt.year.astype(str)
	mics["year"] = mics["interview_year"].astype(int).astype(str)

	## Create a column for the state
	state_regex = r"^[nsNS][ewcsEWCS]\s(.*)\s\b(?:urban|rural|Urban|Rural)\b"
	dhs["state"] = dhs["strata"].str.extract(state_regex)[0].str.lower()
	dhs["state"] = dhs["state"].fillna(dhs["strata"]).str.replace("fct abuja","abuja")
	mics["state"] = mics["state"].str.lower().str.replace("fct abuja","abuja").str.replace("fct","abuja")

	## Clean up the area covariate
	mics["area"] = mics["area"].str.lower()
	mics["region"] = mics["region"].str.lower()

	## And the education covariate
	mics["mom_edu"] = mics["mom_edu"].str.lower()
	edu_cats = {"secondary / secondary-technical":"secondary",
				np.nan:"no education",
				"non-formal":"no education",
				"preschool":"primary",
				"higher/tertiary":"higher",
				"senior secondary":"secondary",
				"junior secondary":"secondary",
				"secondary technical":"secondary",
				"eccde":"no education",
				"vei/iei":"no education",
				"no response":"no education",
				}
	mics["mom_edu"] = mics["mom_edu"].apply(lambda c: edu_cats.get(c,c))

	## Correct the birth order columns
	mics.loc[mics["child_birth_year"].apply(lambda x: isinstance(x, str)),
			 "child_birth_year"] = np.nan
	bord = []
	for k, sf in mics.groupby(["survey","cluster","hh","line_num"]):
		if sf["child_birth_year"].isnull().any():
			bord += list(sf["birth_ln"].values)
			continue
		this_moms = np.argsort(sf["child_birth_year"].values)+1
		bord += list(this_moms)
	#mics["bord"] = bord
	mics["bord"] = mics["birth_ln"].astype(int)

	## Correct for twins, keeping the one with the lower birth order
	mics = mics.loc[~mics[
				["survey","cluster","hh","line_num",
				"child_birth_mon","child_birth_year"]].duplicated(keep="first")]
	dhs = dhs.loc[~dhs[["survey","caseid","child_DoB"]].duplicated(keep="first")]
	
	## Make a Mom's age covariate
	mics["mom_birth_mon"] = mics["mom_birth_mon"].str.lower()\
							.replace("dk",np.nan)\
							.replace("no response",np.nan)\
							.replace("missing",np.nan)\
							.map({"january":1,"february":2,"march":3,"april":4,
								  "may":5,"june":6,"july":7,"august":8,
								  "september":9,"october":10,"november":11,"december":12})
	mics.loc[mics["mom_birth_year"].apply(lambda x: isinstance(x, str)),
			 "mom_birth_year"] = np.nan
	mics["child_birth_mon"] = mics["child_birth_mon"].str.lower()\
								.replace("no response",np.nan)\
								.replace("missing",np.nan)\
								.replace("inconsistent",np.nan)\
								.map({"january":1,"february":2,"march":3,"april":4,
									  "may":5,"june":6,"july":7,"august":8,
									  "september":9,"october":10,"november":11,"december":12})
	mics["mom_DoB"] = pd.to_datetime({"month":mics["mom_birth_mon"],
									  "year":mics["mom_birth_year"],
									  "day":1})
	mics["child_DoB"] = pd.to_datetime({"month":mics["child_birth_mon"],
									  "year":mics["child_birth_year"],
									  "day":1})
	mics["mom_age"] = (mics["child_DoB"]-mics["mom_DoB"]).dt.days/365.
	dhs["mom_age"] = (dhs["child_DoB"]-dhs["mom_DoB"])/12.

	## Choose covariates
	variables = ["survey","area","region","state","year","mom_edu","bord","mom_age"]
	df = pd.concat([dhs[variables],
					mics[variables]],axis=0)
	df = df.loc[(df.notnull().all(axis=1)) &\
				(df["mom_age"] > 5)].reset_index(drop=True)
	#df = dhs[variables].copy()
	print("\nCleaned and compiled dataset...")
	print(df)

	## Standardize values across surveys:
	print("\nVariable values...")
	for c in variables[:-1]:
		values = sorted(df[c].unique())
		print("{} values ({} of them) = {}".format(c,len(values),values))

	## Add mom's age at birth, and compute the intended
	## response variables
	df["ln_mom_age"] = np.log(df["mom_age"])
	
	## Set up the specifics of the regression problem
	response = "ln_mom_age"
	features = ["area","mom_edu","year"]
	correlation_time = (3**4)/8.
	reference = ["urban","primary","year:2008"]

	## Loop over states and compile models
	print("\nStarting the loop over states...")
	output = {}
	for name, sf in df.groupby("state"):

		## Update the states
		print("...fitting in {}".format(name))
		
		## Set up the regression problem's design matrix and 
		## response vector
		Y = sf[response].astype(float)
		X = []
		missing_year_strings = sorted(list(set(np.arange(2008,2021).astype(str))\
								-set(sf["year"].unique())))
		for f in features:
			this_f = pd.get_dummies(sf[f],dtype=float)
			this_f.columns = [str(c) for c in this_f.columns]
			if f == "year":
				for c in missing_year_strings:
					this_f[c] = np.zeros((len(this_f),))
				this_f = this_f[sorted(this_f.columns)]
				this_f.columns = "year:"+this_f.columns
			X.append(this_f)
		X = pd.concat(X,axis=1)
		X["bord"] = sf["bord"].copy().astype(float)

		## Define the intercept
		X["intercept"] = np.ones((len(X),))
		X = X[["intercept","bord"] + X.columns[:-2].tolist()]
		X = X.drop(columns=reference)
		p = len(X.columns)

		## Create the regulatization matrix
		years = [c for c in X.columns if c.startswith("year:")]
		T = len(years)
		D2 = np.diag(T*[-2])+np.diag((T-1)*[1],k=1)+np.diag((T-1)*[1],k=-1)
		#D2[0,2] = 1
		D2[-1,-3] = 1
		RW2 = np.dot(D2.T,D2)*correlation_time
		lam = np.zeros((p,p))
		lam[-T:,-T:] = RW2

		## Fit the model
		lp = rr.RidgeRegression(X,Y,lam)
		beta_err = np.sqrt(np.diag(lp.beta_cov))
		beta = pd.DataFrame(np.array([lp.beta_hat,beta_err]).T,
							columns=["beta","std_err"],
							index=X.columns)
		
		## Add reference category placeholders and save the
		## result
		beta = pd.concat([beta,
						  pd.DataFrame(np.zeros((len(reference),2)),
						  			   index=reference,columns=beta.columns)],
						  axis=0)
		beta["var"] = lp.var*np.ones((len(beta),))
		output[name] = beta

	## Compile the output
	output = pd.concat(output.values(),keys=output.keys())
	print("\nFinal output:")
	print(output)
	output.to_pickle("pickle_jar\\lognormal_age_at_k_by_state.pkl")

	## Make a pdf of fits across states
	cmap = plt.get_cmap("magma")
	bord_values = np.arange(1,11)
	colors = [cmap(i) for i in np.linspace(0.4,0.95,len(bord_values))]
	with PdfPages("_plots\\age_at_kth_kid_by_state.pdf") as book:

		## Loop over states, making a page for each
		print("\nMaking a book of plots...")
		for state, sf in df.groupby("state"):

			## Get the relevant features
			beta = output.loc[state]

			## Compute histograms across bords
			by_ord = sf[["bord","mom_age"]].copy()
			by_ord["bord"] = np.clip(by_ord["bord"],0,10)
			by_ord["freq"] = np.ones((len(by_ord),),dtype=int)
			by_ord = by_ord.groupby(["bord","mom_age"]).count().sort_index()["freq"]

			## Make a big plot
			fig, axes = plt.subplots(2,5,sharex=True,sharey=True,figsize=(15,6))
			axes = axes.reshape(-1)
			for ax in axes:
				ax.spines["left"].set_visible(False)
				ax.spines["top"].set_visible(False)
				ax.spines["right"].set_visible(False)
				ax.grid(color="grey",alpha=0.2)

			## Loop over k, making a panel for each
			for i in bord_values:

				## Get the data
				hist = by_ord.loc[i]
				total = hist.sum()

				## Make the histogram monthly
				hist.index = (12*hist.index).astype(int)/12.
				hist = hist.groupby(level=0).sum()
				
				## Compute relevant averages, first by
				## subsetting to this view of the data
				kf = sf.loc[sf["bord"] == i]

				## Compute the average across other factors
				edu = kf["mom_edu"].value_counts()
				edu *= (1./(edu.sum()))
				edu = edu.loc[~edu.index.isin(reference)]
				edu = (edu*(beta.loc[edu.index,"beta"])).sum()

				## Compute the average across other factors
				time = kf["year"].value_counts()
				time.index = "year:"+time.index
				time *= (1./(time.sum()))
				time = time.loc[~time.index.isin(reference)]
				time = (time*(beta.loc[time.index,"beta"])).sum()

				## Compute the average across other factors
				ur = kf["area"].value_counts()
				ur *= (1./(ur.sum()))
				ur = ur.loc[~ur.index.isin(reference)]
				ur = (ur*(beta.loc[ur.index,"beta"])).sum()
				
				## Compute the model-based estimate
				mu = beta.loc["intercept","beta"]+i*beta.loc["bord","beta"]+ur+time+edu
				pdf = rr.log_normal_density(hist.index.to_numpy(),mu,lp.var)
				pdf *= total/(pdf.sum())
			
				## Plot it all
				axes[i-1].plot(hist,lw=2,color="k")
				axes[i-1].plot(hist.index,pdf,lw=4,color=colors[i-1])
				axes[i-1].text(0.01,0.99,"Child {}".format(i),
								fontsize=22,color="k",#"#bf209f",
								horizontalalignment="left",verticalalignment="top",
								transform=axes[i-1].transAxes)
				axes[i-1].set_ylim((0,None))
				axes[i-1].set_xlim((-1,51))
				axes[i-1].set_yticks([])
				if i >= 6:
					axes[i-1].set_xlabel("Mom's age at birth")
					axes[i-1].set_xticks(np.arange(0,6)*10)

			## Finish up
			fig.suptitle("Mom's age at kth childbirth in "+state.title())
			fig.tight_layout(rect=[0, 0.0, 1, 0.9])
			book.savefig(fig)
			plt.close(fig)

		## Set up metadata
		d = book.infodict()
		d['Title'] = "The age of Nigerian mom's at their kth childbirth"
		d['Author'] = "Niket"

	## Done
	print("...done!")