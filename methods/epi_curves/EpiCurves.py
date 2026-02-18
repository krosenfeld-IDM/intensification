"""EpiCurveGenerator.py

Script to compute and visualize cases over time based on the regression outputs in ConfRates.py """

## Standard tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For logisitc regression related
## business
import logistic as lr

## For making PDFs
from matplotlib.backends.backend_pdf import PdfPages

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

if __name__ == "__main__":

	## Get the cleaned up linelist, with the 
	## regression estimates from ConfRates.py
	ll = pd.read_pickle("..\\pickle_jar\\clean_linelist_regressed.pkl")
	
	## And the associated regression results
	output = pd.read_pickle("..\\pickle_jar\\conf_regression_parameters.pkl")
	output["beta_var"] = output["beta_err"]**2 
	convariances = pd.read_pickle("..\\pickle_jar\\conf_regression_covariances.pkl")

	## And for reference
	state_to_region = pd.read_pickle(
					  "..\\pickle_jar\\state_to_region.pkl"
					  )#.set_index("state")["region"].astype(str)
	#states = sorted(state_to_region.index)

	## And set the time discretization for the output
	t_index = pd.date_range(start="12/31/2008",end="1/31/2024",freq="SM")
	ll["time_bin"] = pd.cut(ll["time"],t_index,labels=t_index[1:])
	ll["time_bin"] = pd.to_datetime(ll["time_bin"])

	## Loop over states in an open book for plots
	epi_curves = {}
	print("\nLooping over states...")
	with PdfPages("..\\_plots\\epi_curves.pdf") as book:

		## Use the region order to organize the book
		for region, states in state_to_region.groupby("region"):

			fig, axes = plt.subplots(figsize=(16,9))
			axes.axis("off")
			axes.text(0.5,0.5,region.title()+" region",
					  horizontalalignment="center",verticalalignment="center",
					  transform=axes.transAxes,
					  color="k",fontsize=48)
			fig.tight_layout()
			book.savefig(fig)
			plt.close(fig)
			
			## Loop over states
			for state in ["lagos"]:#states["state"]:
	
				## Get the subset of the data
				print("...making estimates and a page for {}".format(state))
				sf = ll.loc[ll["state"] == state]
				beta = output.loc[state]
				beta_cov = convariances.loc[state]

				## Construct an epi-curve. Start by binning the lab confirmed
				## data into a SM time series.
				lab_confirmed = pd.to_datetime(sf.loc[sf["classification"] == 1,"time_bin"].copy())
				lab_confirmed = lab_confirmed.value_counts(dropna=False).sort_index().rename("lab_confirmed")
				lab_confirmed = lab_confirmed.reindex(t_index).fillna(0)

				## Similarly construct the lab rejected cases
				lab_rejected = pd.to_datetime(sf.loc[sf["classification"] == 0,"time_bin"].copy())
				lab_rejected = lab_rejected.value_counts(dropna=False).sort_index().rename("lab_rejected")
				lab_rejected = lab_rejected.reindex(t_index).fillna(0)

				## Do the same for compatible and our estimate of
				## confirmed cases.
				compatible = pd.to_datetime(sf.loc[sf["classification"].isnull(),"time_bin"].copy())
				compatible = compatible.value_counts(dropna=False).sort_index().rename("compatible")
				compatible = compatible.reindex(t_index).fillna(0)

				## Confirmed estimate
				confirmed_est = sf.loc[sf["classification"].isnull(),
									  ["time_bin","conf_prob"]].groupby("time_bin").sum()
				confirmed_est.index = pd.to_datetime(confirmed_est.index)
				confirmed_est = confirmed_est["conf_prob"].rename("confirmed_estimate")
				confirmed_est = confirmed_est.reindex(t_index).fillna(0)

				## And the associated variance
				confirmed_var = sf.loc[sf["classification"].isnull(),
									  ["time_bin","conf_var"]].groupby("time_bin").sum()
				confirmed_var.index = pd.to_datetime(confirmed_var.index)
				confirmed_var = confirmed_var["conf_var"].reindex(t_index).fillna(0)
				confirmed_std = np.sqrt(confirmed_var)

				## Put together an output to save
				time_series = pd.concat([lab_confirmed.astype(np.int32),
										 lab_rejected.astype(np.int32),
										 compatible.astype(np.int32),
										 confirmed_est,
										 confirmed_var],axis=1)
				epi_curves[state] = time_series

				## Construct the epi curve
				inf_epi_curve = lab_confirmed+confirmed_est

				## Finally, construct the inferred test positivity over time,
				## averaging appropriately over the fixed effects. Start by computing the
				## local distribution of fixed effects
				age_weights = sf[["under_5","over_5","missing_age"]].sum(axis=0)/(len(sf))
				dose_weights = sf[["zero_dose","one_dose","two_dose","missing_dose"]].sum(axis=0)/(len(sf))
				weights = pd.concat([age_weights,dose_weights],axis=0)
				
				## Construct the feature matrix, first by making the matrix to mark the
				## time step (taking into account the first step as reference)
				time_covs = sorted(beta.loc[~beta.index.isin(sf.columns)].index[1:])
				X_time = np.eye(len(time_covs))
				X_time[0,0] = 1
				X_time = X_time[:,1:]

				## Then add the fixed effect weights
				num_fe = len(beta_cov) - X_time.shape[1]
				X_fe = np.ones((num_fe,))
				X_fe[1:] = weights.loc[beta.index[1:num_fe]].values
				X_fe = np.array(len(time_covs)*[X_fe])
				
				## Finally, compute the estimates by putting X together and multiplying
				X = np.hstack([X_fe,X_time])
				inf_pt = lr.logistic_function(np.dot(X,beta["beta"].values[:X.shape[-1]]))
				inf_pt_var = np.diag(np.dot(np.dot(X,beta_cov.values),X.T))
				inf_pt_var = inf_pt_var*((inf_pt*(1.-inf_pt))**2)
				inf_pt_err = np.sqrt(inf_pt_var)

				## Plot the results
				fig, axes = plt.subplots(2,1,sharex=True,figsize=(16,9))
				for ax in axes:
					axes_setup(ax)

				## Specify the colors for each part of the case stack
				## (confirmed, epi_linked, and clinically compatible)
				colors = ["0.25","0.75"]
				alphas = [1,1]

				## Make the stacked plot
				floor = 0*lab_confirmed
				stack = [lab_confirmed,compatible]
				labels = ["Lab confirmed and epi linked","Clinically compatible\nbut untested"]
				for a,c,l,s in zip(alphas,colors,labels,stack):
					axes[0].fill_between(s.index,floor.values,(floor+s).values,
									  color=c,alpha=a,label=l)
					floor += s
				ylim = axes[0].get_ylim()

				## Add rejections
				#axes[0].plot(lab_rejected,
				#			 color="k",lw=1,ls="dashed",label="Lab rejected",alpha=0.3)

				## Plot the pending and rejected
				axes[0].fill_between(inf_epi_curve.index,
								  inf_epi_curve-2.*confirmed_std,inf_epi_curve+2.*confirmed_std,
								  facecolor="#FF420E",edgecolor="None",alpha=0.5)
				axes[0].plot(inf_epi_curve,color="#FF420E",lw=3,label="Inferred incidence curve for {}".format(state.title()))

				## Details
				axes[0].set(ylabel="Cases per semi-month",
						 ylim=(0,ylim[1]))
				h, l = axes[0].get_legend_handles_labels()
				#axes[0].legend(#[h[i] for i in [1,2,3,0]],
				#			   #[l[i] for i in [1,2,3,0]],
				#			   frameon=False,loc=2)

				## Add rejected case panel for reference
				test_positivity = (lab_confirmed/(lab_confirmed+lab_rejected)).dropna()
				total_tests = (lab_confirmed+lab_rejected).loc[test_positivity.index]
				err = np.sqrt(test_positivity*(1. - test_positivity)/total_tests)
				axes[1].errorbar(test_positivity.index,test_positivity.values,
								 yerr=2.*err.values,
								 color="k",
								 lw=1,ls="None",
								 #marker="o",markersize=8,
								 zorder=1,label="Observed positivity across labs")
				axes[1].fill_between(time_covs,
									 inf_pt-2.*inf_pt_err,inf_pt+2.*inf_pt_err,
									 facecolor="#88b4f5",edgecolor="None",alpha=0.9,zorder=2)
				axes[1].plot(time_covs,inf_pt,color="#116AEB",lw=4,zorder=3,
							 label="Overall, model-based estimate")
				axes[1].set_ylabel("Confirmation probability")
				axes[1].set_ylim((0,1))
				#axes[1].legend(frameon=False,loc=1)

				## Finish up
				#fig.suptitle("Incidence estimates in {}".format(state.title()))
				fig.tight_layout()#rect=[0, 0.0, 1, 0.97])

				## Save an image?
				if state == "lagos":
					print("..........saving this one!")
					fig.savefig("..\\_plots\\epi_curve_inference.png",
								transparent=False)
					plt.show()
					sys.exit()

				## Save a page
				book.savefig(fig)
				plt.close(fig)
			
	## Done
	print("...done!")

	## Put together the epi-curve outputs
	epi_curves = pd.concat(epi_curves.values(),keys=epi_curves.keys())
	print("\nFinal epi_curves:")
	print(epi_curves)
	epi_curves.to_pickle("..\\pickle_jar\\estimated_epi_curves.pkl")










