""" StratifiedDistribution.py

Age at infection estimates, via a soft-max smoothing (buckets) method, stratified
at region level.

Note that the input data for this workflow is not included in the repository, only
the outputs."""

## Standard tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## The regression model
import masked_buckets as mb

## For making PDFs
from matplotlib.backends.backend_pdf import PdfPages

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return axes

if __name__ == "__main__":

	## Get the processed line list from the pickle jar,
	## made originally in the epi_curves directory.
	## NOTE: Data at this level of disaggregtion is not included in
	## the repository. 
	root = " "## Path to the directory where you have a classified line list format as below
	ll = pd.read_pickle(root+"\\pickle_jar\\clean_linelist_regressed.pkl")

	## Subset to confirmed cases, including those that have a 
	## more than 90% chance of confirmation according to the
	## regression model, that have age information. Pick the main
	## columns of interest for this study.
	ll = ll.loc[((ll["classification"] == 1.) |\
				(ll["conf_prob"] >= 0.9)) &\
				(ll["age"].notnull()),
					[
					"time",
					"region",
					"state",
					"dob",
					"age",
					"doses"
					]
				]
	
	## Set up the birth year
	ll["birth_year"] = (ll["time"] - \
					   pd.to_timedelta(365*ll["age"],unit="d")).dt.year

	## And the age at infection in months
	ll["age_at_inf"] = ll["time"].dt.year - ll["birth_year"]
	age_max = 25 #29
	ll["age_at_inf"] = np.clip(ll["age_at_inf"],None,age_max)

	## Subset to relevant years
	ll = ll.loc[(ll["birth_year"] >= 2002) & (ll["birth_year"] <= 2023)]

	## And prepare the mask associated mask
	max_year = ll["time"].max().year
	mask = np.arange(2002,max_year+1)[:,np.newaxis]+\
		   np.arange(0,age_max+1)[np.newaxis,:]
	mask = np.where((mask < ll["time"].dt.year.min()) | (mask > max_year),0,1)
	mask = pd.DataFrame(mask,
						index=np.arange(2002,max_year+1,dtype=int),
						columns=np.arange(0,age_max+1,dtype=int))

	## Summarize into a collection of histograms
	df = ll[["birth_year","age_at_inf"]].groupby("birth_year").apply(
			lambda s: s["age_at_inf"].value_counts().sort_index()
			).unstack().fillna(0)
	national_index = df.index
	
	## Groupby strata and fit models accordingly, all in a 
	## pdf book for plots.
	strata = "region"
	correlation_times = {"north central":10.,
						 "north east":10.,
						 "north west":10., #10.,
						 "south east":10., #8. ## previously
						 "south south":10.,
						 "south west":10.}
	with PdfPages("..\\_plots\\stratified_age_at_inf_by_birth_year.pdf") as book:
		print("\nLooping over {}...".format(strata))
		output = {}
		for name, sl in ll.groupby(strata):		

			## Summarize the data
			df = sl[["birth_year","age_at_inf"]].groupby("birth_year").apply(
			lambda s: s["age_at_inf"].value_counts().sort_index()
			).unstack().fillna(0)

			## Align the mask
			this_mask = mask.loc[df.index].values
			df = df.T.reindex(mask.columns).fillna(0).T
			
			## For plotting
			total_by_year = df.sum(axis=1)
			frac = df.div(total_by_year,axis=0)
			tot_samples = int(total_by_year.sum())

			## Get the correlation time
			correlation_time = correlation_times.get(name,5.)

			## Fit the model
			buckets = mb.BinomialPosterior(df,
									  correlation_time=correlation_time,
									  g2g_correlation=4.,
									  mask=this_mask,
									  )
			result = mb.FitModel(buckets)
			print("...for {}, success = {},"
				  " {} samples".format(name,result.success,tot_samples))

			## Unpack the result
			samples = mb.SampleBuckets(result,buckets)
			low = np.percentile(samples,2.5,axis=0)
			high = np.percentile(samples,97.5,axis=0)
			mid = samples.mean(axis=0)
			var = samples.var(axis=0)

			## Create outputs for use elsewhere, with aligned indices.
			## this only affects regions with 0 cases in the latest cohort.
			mid_df = pd.DataFrame(mid,
								  columns=frac.columns,
								  index=frac.index)#.stack().rename("avg")
			mid_df = mid_df.reindex(national_index)\
						.fillna(method="ffill")\
						.stack().rename("avg")
			var_df = pd.DataFrame(var,
								  columns=frac.columns,
								  index=frac.index)#.stack().rename("var")
			var_df = var_df.reindex(national_index)\
						.fillna(method="ffill")\
						.stack().rename("var")
			this_output = pd.concat([mid_df,var_df],axis=1)
			output[name] = this_output

			## make a plot
			fig, axes = plt.subplots(7,4,sharex=True,sharey=False,figsize=(15,14)) ## r X c >= age-max+1+2
			axes = axes.reshape(-1)
			for i,ax in enumerate(axes):
				if i in {2,3}:#,18,19}:
					ax.axis("off")
					continue
				axes_setup(ax)
				ax.grid(color="grey",alpha=0.2)
			for i in frac.columns:

				## Axis to panel
				if i > 1:
					j = i + 2
				else:
					j = i

				## Plot the model
				axes[j].fill_between(frac.index,
									   low[:,i],high[:,i],
									   facecolor="grey",edgecolor="None",alpha=0.4,label="Model")
				axes[j].plot(frac.index,mid[:,i],
							 lw=2,color="grey")

				## Plot the data
				axes[j].plot(frac[i],
							   ls="None",lw=1,color="k",
							   markersize=8,
							   marker="o",markeredgecolor="k",markerfacecolor="None",markeredgewidth=2,
							   label="Observed fraction")
				axes[j].text(0.01,0.99,"{} years old".format(i),
							   horizontalalignment="left",verticalalignment="top",
							   fontsize=22,color="k",
							   transform=axes[j].transAxes)
				if j%4 == 0:
					axes[j].set_ylabel("Probability")
					#axes[i-1].set_ylim((0.055,0.125))
					#axes[i-1].set_yticks([0.06,0.08,0.1,0.12])
				if j >= 8:
					axes[j].set_xticks(np.arange(2000,2021,5))

			## Set up a legend
			axes[2].plot([],
						ls="None",color="k",
						markersize=8,
						marker="o",markeredgecolor="k",markerfacecolor="None",markeredgewidth=2,
						label="Observed fraction")
			#axes[2].plot([],lw=3,color="xkcd:goldenrod",ls="dashed",label="Standard model")
			axes[2].plot([],lw=2,color="grey",label="Bias corrected")
			axes[2].legend(loc="center",frameon=False)

			## Finish up
			fig.suptitle("Infection age by birth cohort in "+name.title())
			fig.tight_layout(rect=[0, 0.0, 1, 0.9])
			book.savefig(fig)
			plt.close(fig)

		## Set up metadata
		d = book.infodict()
		d['Title'] = "Infection age in Nigeria"
		d['Author'] = "Niket"

	## Finish up
	print("...finished.")

	## Create the full output
	output = pd.concat(output.values(),keys=output.keys())
	print("\nFinal output:")
	print(output)
	output.to_pickle("..\\pickle_jar\\age_at_inf_by_{}.pkl".format(strata))

	## Make a plot of the mean
	with PdfPages("..\\_plots\\stratified_mean_age_at_inf.pdf") as book:
		print("\nLooping over {} for the trend in mean...".format(strata))
		for name, sl in ll.groupby(strata):

			## Compute the mean
			mu = sl[["birth_year","age_at_inf"]].copy()
			mu = mu.groupby("birth_year").agg(
							{"age_at_inf":["mean",
										   "var",
										   "count"]}
							).droplevel(0,axis=1)
			mu["err"] = np.sqrt(mu["var"])

			## For comparison, what's the same but by report
			## date
			rd = sl[["time","age_at_inf"]].copy()
			rd["time"] = rd["time"].dt.year
			rd = rd.groupby("time").agg(
							{"age_at_inf":["mean",
										   "var",
										   "count"]}
							).droplevel(0,axis=1)
			rd["err"] = np.sqrt(rd["var"])

			## Get the estimate
			mid = output.loc[name,"avg"].unstack().values
			inf_mu = (mask.columns.values[np.newaxis,:]*mid).sum(axis=1)

			## Make the plot
			year_min = ll["time"].min().year
			fig, axes = plt.subplots(figsize=(14,5))
			axes_setup(axes)
			axes.errorbar(mu.index,mu["mean"].values,
						  yerr=2.*mu["err"].values,
						  color="k",lw=2,
						  marker="o",markersize=12,zorder=4)
			axes.plot(rd.index,rd["mean"],
					  color="grey",ls="dashed",marker="o",
					  lw=2,zorder=5,label="Data by reporting year")
			axes.axvline(year_min,
						 color="grey",
						 ls="dashed",lw=3,zorder=2)
			axes.plot(mask.index[:len(inf_mu)],inf_mu,
					  lw=3,color="C0",zorder=7,label="Biased corrected model")
			axes.text(year_min+0.25,mu["mean"].max(),#+mu["err"].max(),
					  "Start of the\nline list",
					  fontsize=22,color="grey",
					  horizontalalignment="left",verticalalignment="bottom")
			axes.set_xticks(mu.index[::2])
			axes.set_ylabel("Avg. infection age")
			axes.set_xlabel("Birth year")
			axes.legend(frameon=False,loc=1)
			
			## Finish up
			fig.suptitle("Avg. infection age by birth cohort in "+name.title())
			fig.tight_layout(rect=[0, 0.0, 1, 0.9])
			book.savefig(fig)
			plt.close(fig)

	print("...finished.")

	