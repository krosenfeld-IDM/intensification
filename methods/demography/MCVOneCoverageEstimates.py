""" MCVOneCoverageEstimates.py

Using the distributions estimated in MCVOneProbability.py to estimate state
level coverage across birth cohorts.  """
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

## For regression estimates
import survey.logistic as lr

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

	## Get the demographic cell populations
	dist = pd.read_pickle("pickle_jar\\mom_distribution.pkl")

	## Interpolate to the monthly scale
	time = pd.date_range(start="2006-01-01",end="2023-12-01",freq="MS",name="time")
	dist["year"] = pd.to_datetime({"year":dist["year"],
								   "month":6,"day":1})
	dist = dist.set_index([c for c in dist.columns if c != "v005"]).sort_index()["v005"]
	dist = dist.groupby([n for n in dist.index.names if n != "year"]).apply(
						lambda s: s.loc[s.name].reindex(time).interpolate(limit_direction="both")
						)
	dist = dist.reset_index()

	## Set the problem up in terms of state-time pairs, since we're going to
	## marginalize across the rest of the pieces
	dist = dist.set_index(["state","time"]).sort_index()

	## Get the logistic regression results
	lr_results = pd.read_pickle("pickle_jar\\mcv1_logistic_regression_by_state.pkl")
	lr_results["beta_var"] = lr_results["beta_err"]**2

	## And the associated covariance matrices
	covariances = pd.read_pickle("pickle_jar\\mcv1_logistic_regression_covariances_by_state.pkl")

	## Extract some details of the regression problem
	## from the serialized outputs
	time_covs = covariances.loc["abia"].index.str.startswith("time:")
	time_covs = covariances.loc["abia"].loc[time_covs].index
	num_fe = covariances.loc["abia"].shape[0]-len(time_covs)
	fixed_ef = lr_results.loc["abia"].index[:num_fe]
	reference = lr_results.loc["abia"].index[len(time_covs)+num_fe:]

	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
	#	print(lr_results.loc["abia"])
	#sys.exit()
	
	## Loop over cells to compute distribution parameters
	## for all possible combinations.
	print("\nComputing estimates by state/time...")
	output = []
	for st, s in dist.groupby(["state","time"]):

		## Unpack this subset
		state, dt = st
		t = f"time:{dt.year}-{dt.month:02}"

		## Normalize the weights
		weights = s.copy()
		weights["v005"] *= 1./(weights["v005"].sum())

		## Subset to the right state
		beta = lr_results.loc[state]
		beta_cov = covariances.loc[state]

		## Then compute the mom types
		mom_x = pd.get_dummies(weights[["v025","v106"]],
							   prefix="",prefix_sep="").astype(float)
		mom_x["intercept"] = np.ones((len(mom_x),))
		mom_x = mom_x[fixed_ef]

		## Add the time component
		X_time = pd.DataFrame(np.zeros((len(mom_x),len(time_covs))),
							  index=mom_x.index,
							  columns=time_covs)
		if t not in reference:
			X_time.loc[:,t] = 1.
		
		## Put it all together and compute
		mom_x = pd.concat([mom_x,X_time],axis=1).values
		lno = np.dot(mom_x,beta["beta"].values[:num_fe+len(time_covs)])
		var = np.diag(np.dot(np.dot(mom_x,beta_cov.values),mom_x.T))

		## Finally, compute probabilites and variances
		p_vax = lr.logistic_function(lno)
		p_var = var*(p_vax**2)*((1.-p_vax)**2)

		## And weight and store them
		p_vax = (weights["v005"]*p_vax).sum()
		p_var = ((weights["v005"]**2)*p_var).sum()

		## Store the results
		output.append((state,
					   dt,
					   p_vax,
					   p_var))

	## Put it all together
	print("\nFinal output...")
	ri = pd.DataFrame(output,
					  columns=["state","time","mcv","var"])
	print(ri)	
	ri.to_pickle("pickle_jar\\mcv1_lr_estimate_by_state.pkl")

