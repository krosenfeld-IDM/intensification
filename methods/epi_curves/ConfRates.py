""" ConfRates.py

Models for the probability that a clinically compatible case is actually measles. 
Stratified either by region (with a state effect) or state. 

Note the input data required to run this script is not provided in the repo. This
script is meant to illustrate the workflow.
"""

## Standard tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For logisitc regression
import logistic as lr

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

if __name__ == "__main__":

	## Get the cleaned up linelist, prepared in
	## PrepareCaseDataset.py
	ll = pd.read_pickle("..\\pickle_jar\\clean_linelist.pkl")
	ll = ll.loc[ll["time"] >= "2009-01-01"].reset_index(drop=True)
	
	## For use throughout
	state_to_region = pd.read_pickle(
					  "..\\pickle_jar\\state_to_region.pkl"
					  ).set_index("state")["region"].astype(str)

	## Choose the strata to work at
	## "state" or "region"
	strata = "state" ## Region level plotting will fail, but conf rate estimates will work.
	assert (strata == "region" or strata == "state"),\
			"'{}' is an invalid strata".format(strata)

	## Construct a few additional features
	## Starting with the time smoothing
	ll["month"] = ll["time"].dt.to_period("M").dt.to_timestamp()
	time_steps = pd.date_range(ll["month"].min(),
							   ll["month"].max(),
							   freq="MS")

	## And then the age bins
	ll["under_5"] = (ll["age"] < 5.).astype(np.float64)
	ll["over_5"] = (ll["age"] >= 5.).astype(np.float64)
	ll["missing_age"] = (ll["age"].isnull()).astype(np.float64)
	
	## And then the dose covariate
	ll["missing_dose"] = (ll["doses"].isnull()).astype(np.float64)
	ll["zero_dose"] = (ll["doses"] == 0).astype(np.float64)
	ll["one_dose"] = (ll["doses"] == 1).astype(np.float64)
	ll["two_dose"] = (ll["doses"] == 2).astype(np.float64)

	## Create a dataset for model fitting
	df = ll.copy().sort_values("time")
	df["intercept"] = np.ones((len(df),))
	
	## Add state variables if strata == "region"
	if strata == "region":
		state_covs = pd.get_dummies(df["state"])
		df = pd.concat([df,
						state_covs[sorted(state_covs.columns)],
						],
						axis=1)
	elif strata == "state":
		state_covs = pd.DataFrame()
	state_ef = list(state_covs.columns)

	## Finally add the time covariates, with a little extra sauce to
	## make sure we have a column for every time step.
	time_covs = pd.get_dummies(df["month"]).T
	time_covs = time_covs.reindex(time_steps).T.fillna(0.)
	df = pd.concat([df,
					time_covs[sorted(time_covs.columns)],
					],
					axis=1)
	time_ef = list(time_covs.columns)[1:]

	## Set up the regression problem definitions
	response = "classification"
	fixed_ef = [
				"intercept",
				"one_dose",
				"two_dose",
				"missing_dose",
				#"under_5",
				"over_5",
				#"missing_age",
				]
	reference = ["under_5","zero_dose",time_steps[0]]

	## including the RW matrix
	T = len(time_steps)-1
	D2 = np.diag(T*[-2])+np.diag((T-1)*[1],k=1)+np.diag((T-1)*[1],k=-1)
	#D2[0,2] = 1
	D2[-1,-3] = 1
	RW2 = ((4**4)/8.)*np.dot(D2.T,D2)

	## Finally subset the data in fit and prediction pieces
	cols = fixed_ef+\
		   state_ef+\
		   time_ef+\
		   ["classification",strata]
	df = df[cols]
	fit_df = df.loc[df["classification"].notnull()].copy()

	## Loop over strata and fit models
	print("\nLooping over modeling strata = '{}'...".format(strata))
	output = {}
	covariances = {}
	predictions = []
	for name, sf in fit_df.groupby(strata):
		
		## Subset to what we need for this strata
		if strata == "region":
			states = state_to_region.loc[state_to_region == name].index
			this_reference = reference+[states[0]]
		elif strata == "state":
			states = []
			this_reference = reference.copy()

		## Put the full matrix together, leaving out a state
		## in the strata to avoid the dummy trap.
		sf = sf[fixed_ef+\
				list(states)[1:]+\
				time_ef+\
				[response]]

		## Set up the full regularization matrix
		this_strata_features = sf.columns[:-1]
		p = len(this_strata_features)
		lam = np.zeros((p,p))
		lam[-T:,-T:] = RW2

		## Fit the model
		model = lr.LogisticRegressionPosterior(sf[this_strata_features].values,
											   sf[response].values,
											   lam=lam)
		result = lr.FitModel(model)
		print("...for {}, success = {}".format(name,result.success))

		## Save the output, first by putting together a
		## dataframe of effects
		beta_hat = result["x"]
		beta_cov = result["hess_inv"]
		beta_err = np.sqrt(np.diag(beta_cov))
		beta = pd.DataFrame(np.array([beta_hat,beta_err]).T,
							columns=["beta","beta_err"],
							index=this_strata_features)

		## Add reference category placeholders and save the
		## result
		beta = pd.concat([beta,
						  pd.DataFrame(np.zeros((len(this_reference),2)),
									   index=this_reference,columns=beta.columns)],
						  axis=0)
		output[name] = beta

		## Store the covariance matrix
		covariances[name] = pd.DataFrame(beta_cov,
										 index=this_strata_features,
										 columns=this_strata_features)

		## Get the prediction dataset
		pred_df = df.loc[(df[strata] == name) &\
						 (df["classification"].isnull())].copy()
		pred_X = pred_df[this_strata_features].values

		## Compute a prediction and store it
		pred = lr.logistic_function(np.dot(pred_X,beta_hat))
		pred_var = np.diag(np.dot(np.dot(pred_X,beta_cov),pred_X.T))
		pred_var = pred_var*((pred*(1.-pred))**2)
		pred = pd.DataFrame(np.array([pred,pred_var]).T,
							index=pred_df.index,
							columns=["conf_prob","conf_var"])
		predictions.append(pred)

	## Put together the 3 outputs - one of them is the regresion results, which
	## you can use to visualize effect sizes, etc.
	output = pd.concat(output.values(),keys=output.keys())
	print("\nFinal output:")
	print(output)
	output.to_pickle("..\\pickle_jar\\conf_regression_parameters.pkl")
	
	## The other is the full covariance matrices, since sometimes you need
	## the whole (gaussian approx) to the regression posterior
	covariances = pd.concat(covariances.values(),keys=covariances.keys())
	print("\nAssociated covariance matricies:")
	print(covariances)
	covariances.to_pickle("..\\pickle_jar\\conf_regression_covariances.pkl")
	
	## The third is the prediction dataset
	predictions = pd.concat(predictions,
							axis=0)
	ll = pd.concat([ll,predictions],
					axis=1)
	print("\nAugmented line list...")
	print(ll)
	ll.to_pickle("..\\pickle_jar\\clean_linelist_regressed.pkl")