""" SurvivalPrior.py

Script to create a state's immunity profile and use it to constrain variation in the 
reporting rate and initial susceptible population. """
import os
import sys
import warnings
import methods

## Standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For model fitting
from scipy.optimize import minimize

## For plot arrangement
import matplotlib.gridspec as gridspec

## For moving between age and time
from pandas.tseries.offsets import DateOffset,\
                                   MonthBegin,\
                                   MonthEnd

## For reference
colors = ["#FF420E","#00ff07","#0078ff","#BF00BA"]

def axes_setup(axes,inplace=True):
    axes.spines["left"].set_position(("axes",-0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    if inplace:
    	return None
    else:
    	return axes

def model_overview(state,imm_profile,sias,survival,sf,result,phi,r_ls):

    '''This function takes the key outputs from the script and arranges them as an 
    overview plot. See Appendix 2, Fig 1 in the paper. '''

    ## Set up the figure
    fig = plt.figure(figsize=(14,9))
    gs = gridspec.GridSpec(6, 3, figure=fig)
    vax_ax = axes_setup(fig.add_subplot(gs[:4,0]),inplace=False)
    vax_leg = fig.add_subplot(gs[-1,0])
    fit_ax = axes_setup(fig.add_subplot(gs[:3,1:3]),inplace=False)
    rr_ax = axes_setup(fig.add_subplot(gs[3:,1:3],sharex=fit_ax),inplace=False)

    ## Plot the vaccine derived immunity profile as a set of 
    ## stacked bars
    profile_colors = ["k","grey"]
    sia_cmap = plt.get_cmap("Blues")
    profile_colors += [sia_cmap(i) for i in np.linspace(0.1,0.9,len(sias.md))]
    profile_colors += ["xkcd:light red"]
    bottom = 0*survival[0]
    for i, item in enumerate(imm_profile.loc[survival.index].items()):
        label, comp = item
        if i < 2 or (i == len(sias.md)//2):
            label = label.split(" ")[0].upper()
            label = label.replace("SIA","Catch-up vaccine")
        elif label == "infected":
            label = "Infected by 2024"
        else:
            label = None
        vax_ax.bar(comp.index,comp.values,0.666,
                 bottom=bottom,
                 color=profile_colors[i],
                 )
        vax_leg.fill_between([],[],color=profile_colors[i],label=label)
        bottom += comp
    vax_ax.set_ylim((0,1))
    vax_ax.set_xticks(survival.index[0::4])
    vax_ax.set_xticklabels(["`"+str(i)[-2:] for i in survival.index[0::4]])
    vax_ax.set_xlabel("Birth cohort")
    vax_ax.set_ylabel("Immune fraction")

    ## Compute the variance in the fits via sampling
    samples = np.random.multivariate_normal(result["x"],cov,size=(1000,))
    samples = 1./(1. + np.exp(-samples))
    samples = samples*(phi.values[None,:])
    fit_low = np.percentile(samples,2.5,axis=0)
    fit_high = np.percentile(samples,97.5,axis=0)
    
    ## Plot the fit and inference
    fit_ax.fill_between(sf.index,fit_low,fit_high,
                         facecolor="k",edgecolor="None",alpha=0.25,zorder=2)
    fit_ax.plot(sf["fit"],color="k",lw=4,label="Survival model",zorder=3)
    fit_ax.plot(sf["cases"],color=colors[2],lw=4,label="Observed cases",zorder=4)
    fit_ax.plot(r_ls*phi,color="xkcd:saffron",lw=6,zorder=3,ls="dashed",label=r"With const. $r_t$")
    fit_ax.set_ylabel("Cases (per year)")
    fit_ax.legend(frameon=False,fontsize=20,loc=1)
    rr_ax.fill_between(sf["rr"].index,100*(sf["rr"]-2.*sf["rr_std"]),100*(sf["rr"]+2.*sf["rr_std"]),
                         facecolor=colors[3],edgecolor="None",alpha=0.2,zorder=0)
    rr_ax.fill_between(sf["rr"].index,100*(sf["rr"]-sf["rr_std"]),100*(sf["rr"]+sf["rr_std"]),
                         facecolor=colors[3],edgecolor="None",alpha=0.4,zorder=1)
    rr_ax.plot(100*sf["rr"],color=colors[3],lw=6,label="Estimated reporting rate",zorder=2)
    nmfr = 100*(sf["rr"].mean())*sf["rejected"]/(sf["rejected"].mean())
    rr_ax.plot(nmfr,color="k",lw=2,ls="dashed",
               marker="o",markersize=10,
               label="The trend in rejected cases")
    rr_ax.legend(frameon=False,fontsize=20)
    rr_ax.set_ylabel("Reporting rate (%)")
    fit_ax.set_ylim((0,None))
    rr_ax.set_ylim((0,None))

    ## Finish up
    vax_leg.axis("off")
    #fig.suptitle("Coarse regression in {}".format(state.title()))
    fig.tight_layout()#rect=[0.0, 0.0, 1, 0.965])
    vax_leg.legend(frameon=False,loc="center",
                       fontsize=18,
                       bbox_to_anchor=(0.5,2.8))
    
    return fig

def smoothing_cost(theta,expI,cases,lam):
    '''This is the objective function for the regression of expected
    infections against cases. Theta is a time-correlated set of numbers 
    that get logistic-transformed into a 0 to 1 scale-factor representing 
    annualized reporting rates. See the discussion around Appendix 2, Fig 1. '''
    beta = 1./(1. + np.exp(-theta))
    f = beta*expI
    ll = np.sum((cases - f)**2)
    lp = np.dot(theta.T,np.dot(lam,theta))
    return ll+lp

def smoothing_grad(theta,expI,cases,lam):
    ''' Gradient of the objective function above for optimization. '''
    beta = 1./(1. + np.exp(-theta))
    f = beta*expI
    grad = 2.*(f-cases)*f*(1.-beta)
    grad += 2.* np.dot(lam,theta)
    return grad

def smoothing_hessian(theta,expI,cases,lam):
    ''' Hessian of the objective function above to quantify uncertainty. '''
    beta = 1./(1. + np.exp(-theta))
    hess = np.diag(beta*(1.-beta)*expI*((1.-2*beta)*(beta*expI-cases)+beta*(1-beta)*expI))
    hess += lam
    return 2.*hess

def code_to_offset(c):

    ''' Translation of SIA age ranges, in months (M) or
    years (Y) to offset types. '''

    if c.endswith("M"):
        return DateOffset(months=int(c[:-1]))
    elif c.endswith("Y"):
        return DateOffset(years=int(c[:-1]))
    return DateOffset(months=int(c))

class CampaignCalendar(object):

    '''This object encapulates SIA metadata (like how many doses were delivered, start and
    end dates, etc.) and a table of eligibility by birth cohort. Partially eligible birth cohorts
    are estimated to the nearest month and aggregating the birth seasonality profile (or using a
    uniform distribution if no seasonality is provided). '''

    def __init__(self,cal,b_season=None):

        ## Save the raw data
        self.md = cal.reset_index(drop=True)

        ## Create a single SIA date (set to the median) to
        ## get approximate birth ranges. Ensure everything is chronological.
        self.md["time"] = self.md["start_date"] +\
                        pd.to_timedelta(
                            (self.md["end_date"]-self.md["start_date"]).dt.days // 2,
                            unit="d")
        self.md = self.md.sort_values("time").reset_index(drop=True)

        ## Add an SIA name to identify across data structures
        self.md["name"] = "SIA " + self.md["time"].dt.strftime("%b %Y")
       
        ## Get the birth date eligibility bounds, with a 
        ## filter for the pandas performance warning on this operation
        ## being serial under the hood.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.md["start_elig"] = self.md["time"] -\
                                self.md["age_group"].str.split("-")\
                                                    .str[1]\
                                                    .apply(lambda s: code_to_offset(s))
            self.md["end_elig"] = self.md["time"] -\
                                self.md["age_group"].str.split("-")\
                                                    .str[0]\
                                                    .apply(lambda s: code_to_offset(s))

        ## Make a data structure to capture birth-month eligibility
        ## by campaign, to be aggregated to a birth-cohort. Start by 
        ## collecting months with some eligibility
        time = pd.date_range(start=self.md["start_elig"].min()+MonthBegin(-1), 
                             end=self.md["end_elig"].max()+MonthEnd(1),
                             freq="d",
                             )

        ## Loop over campaigns and indicator eligibility by day,
        ## then average into covered months
        cohorts = pd.DataFrame(np.zeros((len(time),len(self.md))),
                               index=time)
        for i, r in self.md.iterrows():
            cohorts.loc[r.loc["start_elig"]:r.loc["end_elig"],
                        i] = 1.
        cohorts = cohorts.resample("MS").mean()

        ## Now account for birth seasonality, i.e. the fraction
        ## of a birth cohort each month represents
        if b_season is not None:
            cohorts = cohorts*\
                    b_season.loc[cohorts.index.month].values[:,None]
        else:
            cohorts *= 1./12.

        ## Aggregate up and save into the 
        ## probability(targeted | born in birth cohort b).
        self.pr_tb = cohorts.groupby(lambda t: t.year).sum()

    def __str__(self):

        ''' Method to print in a sensible way. '''
        
        out = "Campaign metadata:\n"+\
              self.md.__str__()+\
              "\nWith associated cohort coverage:\n"+\
              self.pr_tb.__str__()
        return out

    def __iter__(self):

        '''Organized iteration through the SIA names, meta data and
        the eligibility column. '''

        name_gen = (c for c in self.pr_tb.columns)
        md_gen_exp = (r for _, r in self.md.iterrows())
        elig_gen_exp = (c for _, c in self.pr_tb.items())
        return zip(name_gen,md_gen_exp,elig_gen_exp)

def compute_profile(pI,epi,prA_given_I,
                    b_season,sias,
                    mcv1_effic=0.825,mcv2_effic=0.95,sia_vax_effic=0.8):

    ''' This function evaluates the immunity profile, i.e. the recursive equation in
    the paper's appendix 2 at a fixed value of pr(s = I | b) == pI '''

    ## Compute the cumulative infection probability for an
    ## input destined to be infected fraction pI, which can either be a
    ## scalar or a vector across cohorts.
    if np.isscalar(pI):
        FprA_and_I = pI*\
            np.cumsum(prA_given_I.reindex(epi.index).fillna(method="bfill"),axis=1)
    else:
        FprA_and_I = pI.values[:,None]*\
            np.cumsum(prA_given_I.reindex(pI.index).fillna(method="bfill"),axis=1)

    ## Compute the RI component to immunity
    ## First for MCV1, which means that you're eligible, not yet
    ## infected when you are, you get a vaccine, and it takes.
    pr_mcv1 = 0*FprA_and_I
    pr_mcv1.loc[:,0] = b_season.loc[1:3].sum() #3./12.
    pr_mcv1.loc[:,1] = b_season.loc[4:12].sum() #9./12
    pr_mcv1 *= (1.-FprA_and_I).values[:,0][:,None]*\
               mcv1_effic*\
               epi["mcv1"].values[:,None]
    pr_mcv1 = pr_mcv1.sum(axis=1)

    ## Set up the MCV2 component, for an MCV2 dose at
    ## 15 months, which requires eligibility, MCV1 failure,
    ## mcv2 delivery, and mcv2 take, all before infection.
    pr_mcv2 = 0*FprA_and_I
    pr_mcv2.loc[:,1] = b_season.loc[1:9].sum()#9./12.
    pr_mcv2.loc[:,2] = b_season.loc[10:12].sum() #3./12.
    pr_mcv2 *= mcv2_effic*\
              epi["mcv2"].values[:,None]*\
              (1 - (FprA_and_I.values[:,1][:,None] + pr_mcv1.values[:,None]))
    pr_mcv2 = pr_mcv2.sum(axis=1)

    ## Set up the RI portion of the profile, upon which to build
    imm_profile = pd.concat([pr_mcv1.rename("mcv1"),
                             pr_mcv2.rename("mcv2")],
                            axis=1)

    ## And then for SIAs, to get immunity, you must be susceptible at the campaign
    ## time, so have missed MCV1 and MCV2 (or had vaccine failure), not been immunized in 
    ## previous campaign, and receive a dose that takes in the campaign
    for i, md, elig in sias:

        ## Get the survival function at the time of this campaign, and use it
        ## to estimate the component surviving infection by birth cohort
        age_at_campaign = pd.Series(md["time"].year - imm_profile.index,
                                    index=imm_profile.index)
        age_at_campaign = age_at_campaign.loc[(age_at_campaign>=0) &\
                                              (age_at_campaign<=FprA_and_I.columns[-1])]
        sia_surv = pd.Series(np.diag(FprA_and_I.loc[
                                       age_at_campaign.index,
                                       age_at_campaign.values]),
                             index=age_at_campaign.index)
        
        ## Create a dataframe to organize immunity components
        pr_SIA = elig.copy()
        pr_SIA = pd.concat([pr_SIA,
                            imm_profile.sum(axis=1).rename("vax"),
                            sia_surv.rename("inf.")],
                            axis=1)

        ## Compute the overall effect, note that this assumes that
        ## coverage is total - so outbreak response campaigns for example
        ## are assumed to target the full population.
        pr_SIA = (sia_vax_effic*pr_SIA[i])*(1. - (pr_SIA[["inf.","vax"]].sum(axis=1)))
        pr_SIA = pr_SIA.fillna(0).rename(md["name"])
        
        ## And add it to the vaccine profile, so that SIA effects
        ## on the same cohort compound.
        imm_profile = pd.concat([imm_profile,pr_SIA],axis=1)

    return imm_profile

if __name__ == "__main__":

    ## Get the state name from command line input
    ## otherwise stick with a default.
    state = " ".join(sys.argv[1:])
    if state == "":
        state = "lagos"

    ## Process the flags, used by generator scripts
    ## that call this script for every state.
    state = state.split("-")
    if len(state) > 1:
        _serialize = True
        import pickle
    else:
        _serialize = False
    state = state[0].rstrip()

    ## Get the state's region from the look up CSV for 
    ## reference.
    s_and_r = pd.read_csv(os.path.join("_data",
    					  "states_and_regions.csv"),
    					  index_col=0)
    region = s_and_r.loc[s_and_r["state"] == state,
                         "region"].values[0] 

    ## Get the epi data from CSV, with some
    ## data type parsing and reshaping into a multiindex
    ## dataframe with state and time as levels.
    epi = pd.read_csv(os.path.join("_data",
                      "southern_states_epi_timeseries.csv"),
                      index_col=0,
                      parse_dates=["time"])\
                    .set_index(["state","time"])
    epi = epi.loc[state]

    ## Then aggregate to the annual scale
    df = epi[["cases","rejected","births","births_var","mcv1","mcv2"]].groupby(
                lambda t: t.year
                ).agg({"cases":"sum",
                       "rejected":"sum",
                       "births":"sum",
                       "births_var":"sum",
                       "mcv1":"mean",
                       "mcv2":"mean"}).loc[2009:2023]
    print("\nCoarse time data set for {}:".format(state.title()))
    print(df)

	## The birth seasonality profile (for estimating when
    ## people become eligible for different vaccines). This is
    ## estimated via DHS and MICS data.
    b_season = pd.read_csv(os.path.join("_data",
    					   "birth_seasonality_profiles.csv"),
    					   index_col=0)\
    					   .set_index(["state","birth_year","birth_month"])
    b_season = b_season.loc[state,"avg"].groupby("birth_month").mean()

    ## And then get the SIA calendar, via output from SIACalendar.py,
    ## and use it to create a campaign calendar object, which organizes
    ## campaign targets by birth cohort over time.
    cal = pd.read_csv(os.path.join("_data",
                      "imputed_sia_calendar_by_state.csv"),
                    index_col=0,
                    parse_dates=["start_date","end_date"]) 
    cal = cal.loc[cal["state"] == state]
    sias = CampaignCalendar(cal,b_season)

    ## And get the age at infection inferences, an output
    ## from a multinomial regression on the granular data. See the 
    ## discussion just under Appendix 2, Fig 1 for details.
    age_dists = pd.read_csv(os.path.join("_data",
    						"southern_age_at_infection.csv"),
    					index_col=0)\
    					.set_index(["region","birth_year","age"])
    prA_given_I = age_dists.loc[region,"avg"].unstack()
    prA_given_I_var = age_dists.loc[region,"var"].unstack()

    ## Compute the profile with pI = 0 and pI = 0.1 to
    ## find the balanced, normalized profile, taking advantage
    ## of the profile's linearity in terms of pI.
    ri_cov = df[["mcv1","mcv2"]].reindex(np.arange(1978, ## Intro year
                                    df.index[-1]+1,dtype=np.int32))
    ri_cov.loc[ri_cov.index[0],ri_cov.columns] = 0
    ri_cov = ri_cov.interpolate()
    right_profile = compute_profile(0.1,ri_cov,prA_given_I,
                                   b_season,sias)
    left_profile = compute_profile(0,ri_cov,prA_given_I,
                                   b_season,sias)
    fL = -1 + (left_profile.sum(axis=1))
    fR = 0.1 - 1 + (right_profile.sum(axis=1))
    pr_inf = (-fL*0.1)/(fR-fL)

   	## Now use the balanced pI estimate to create a 
    ## full immunity profile
    imm_profile = compute_profile(pr_inf,ri_cov,prA_given_I,
                                  b_season,sias)
    print("\nOverall profile")
    print(imm_profile)
    print("Check normalization:")
    print((pr_inf + (imm_profile.sum(axis=1))).values)

    ## Compute the contribution currently left to infection
    pr_inf = 1. - (imm_profile.sum(axis=1))
    prA_and_I = pr_inf.values[:,None]*(prA_given_I.reindex(pr_inf.index).fillna(method="bfill"))

    ## And how much of that has already happened? We use the age
    ## at infection distribution to compute the expected infections in
    ## before the current time.
    mask = np.zeros(prA_and_I.shape)
    mask[np.where(
        (prA_and_I.index.values[:,None] + prA_and_I.columns.values[None,:])\
        == prA_given_I.index[-1]
        )] = 1.
    mask[:prA_given_I.index[-1]-prA_and_I.columns[-1]-prA_and_I.index[-1],-1] = 1.
    imm_profile["infected"] = (mask*np.cumsum(prA_and_I,axis=1)).sum(axis=1)

    ## Compute immunizations that happen during the model period, to
    ## estimate the initial susceptible population
    ## For SIAs
    model_period_sias = sias.md.loc[sias.md["time"].dt.year >= df.index[0],"name"]
    S0 = (df["births"].reindex(imm_profile.index).fillna(method="bfill").values[:,None]\
         *imm_profile[model_period_sias]).loc[:df.index[0]-1].sum().sum()
    S0_var = (df["births"].reindex(imm_profile.index).fillna(method="bfill").values[:,None]\
            *(imm_profile[model_period_sias])*(1-imm_profile[model_period_sias])\
            +df["births_var"].reindex(imm_profile.index).fillna(method="bfill").values[:,None]\
            *(imm_profile[model_period_sias]**2))\
            .loc[:df.index[0]-1].sum().sum()

    ## For routine MCV1 (MCV1 intro happens in the model period)
    S0 += df.loc[df.index[0],"births"]*(b_season.loc[4:12].sum())\
          *imm_profile.loc[df.index[0],"mcv1"]
    S0_var += df.loc[df.index[0],"births"]*(b_season.loc[4:12].sum())\
              *imm_profile.loc[df.index[0],"mcv1"]\
              *(1. - (b_season.loc[4:12].sum())*imm_profile.loc[df.index[0],"mcv1"])\
              + df.loc[df.index[0],"births_var"]*((b_season.loc[4:12].sum()\
              *imm_profile.loc[df.index[0],"mcv1"])**2)

    ## And then for infection
    mask = np.zeros(prA_and_I.shape)
    mask[np.where(
        (prA_and_I.index.values[:,None] + prA_and_I.columns.values[None,:])\
        >= df.index[0]
        )] = 1.
    inf_after_t0 = (mask*prA_and_I).sum(axis=1)
    S0 += (df["births"].reindex(pr_inf.index).fillna(method="bfill")\
            *inf_after_t0).loc[:df.index[0]-1].sum()
    S0_var += (df["births"].reindex(pr_inf.index).fillna(method="bfill")\
            *inf_after_t0*(1.-inf_after_t0)\
            + df["births_var"].reindex(pr_inf.index).fillna(method="bfill")\
            *(inf_after_t0**2)).loc[:df.index[0]-1].sum()

    ## Compute estimates of infectious populations. Start by making
    ## a matrix mapping birth cohort to expected infections.
    T = prA_and_I.index[-1]+prA_and_I.columns[-1]-prA_and_I.index[0]+1
    PST = pd.DataFrame([np.hstack([np.zeros((i,)),
                                   prA_and_I.values[i,:],
                                   np.zeros((T-i-len(prA_and_I.columns),))])\
                        for i in np.arange(len(prA_and_I.index))],
                        index=prA_and_I.index,
                        columns=np.arange(prA_and_I.index[0],prA_and_I.index[0]+T,dtype=np.int32))
    
    ## Calculate the expected infections and the variance from birth 
    ## cohort uncertainty.
    expI = df["births"].reindex(PST.index).fillna(method="bfill").values[:,None]*PST
    varI = df["births_var"].reindex(PST.index).fillna(method="bfill").values[:,None]*(PST**2)
    phi = expI.sum(axis=0).loc[df.index]

    ## And the least-squares reporting rate estimate, as a simple initial
    ## guess.
    r_ls = np.sum(phi.values*df["cases"].values)/(np.sum(phi.values**2))

    ## Then solve the smoothing problem
    ## Start with the regularization matrix for the random walk, i.e. the
    ## time correlated set of parameters which get logistic transformed into
    ## the reporting rate.
    T = len(df)
    D2 = np.diag(T*[-2])+np.diag((T-1)*[1],k=1)+np.diag((T-1)*[1],k=-1)
    D2[0,2] = 1
    D2[-1,-3] = 1
    lam = np.dot(D2.T,D2)*((3.**4)/8.)*(df["cases"].var())

    ## Solve the problem, with initial guess being a constant
    ## reporting rate (in the logit space).
    x0 = np.array(len(df)*[np.log(r_ls/(1.-r_ls))])
    result = minimize(lambda x: smoothing_cost(x,
                                phi.values,
                                df["cases"].values,lam),
                      x0=x0,
                      jac= lambda x: smoothing_grad(x,
                                    phi.values,
                                    df["cases"].values,lam),
                      method="BFGS",
                      )

    ## Unpack the results, compute the uncertainty.
    df["rr"] = 1./(1. + np.exp(-result["x"]))
    df["fit"] = (df["rr"]*phi).values
    sig_nu2 = np.sum((df["cases"]-df["fit"])**2)/len(df)
    hess = smoothing_hessian(result["x"],phi.values,df["cases"].values,lam)/sig_nu2
    cov = np.linalg.inv(hess)
    
    ## Compute and print some stats
    df["rr_var"] = (np.diag(cov))*((df["rr"]*(1.-df["rr"]))**2)
    df["rr_std"] = np.sqrt(df["rr_var"])

    ## Add the S0 and variance estimate, the latter is 
    ## based on the average estimate fluctuation in susceptibility
    ## each year. We store them in the table as a constant over time.
    df["S0"] = S0*np.ones((len(df),))
    df["S0_var"] = (phi.mean()**2)*np.ones((len(df),))

    ## Plot the result
    fig = model_overview(state,
                         imm_profile,
                         sias,
                         prA_given_I,
                         df,
                         result,
                         phi,
                         r_ls,
                         )
    
    ## Save at all
    fig.savefig(os.path.join("_plots","survival_prior.png"))
    print("\nOverall prior information:")
    print(df)
    df.to_pickle(os.path.join("pickle_jar","survival_prior.pkl"))

    ## Finish based on flags above.
    if _serialize:
        pickle.dump(fig, 
                    open(os.path.join("pickle_jar",
                         "survival_prior.fig.pickle"),
                         "wb"))
    if not _serialize:
        plt.show()