# intensification/methods
This directory contains some functions and classes used throughout the scripts in `intensification`. 

From the perspective of the [manuscript](https://www.medrxiv.org/content/10.1101/2025.02.24.25322796v1), the only part of this library used directly is `neighborhood_sir.py`. The other scripts illustrate data processing workflows used to make the processed outputs in `../_data/`, but the raw data needed to run them is *not* included. They're here in this repository for reference purposes only, in case they help others apply these models and ideas to their own work. 

There are three main workflows included:
1. `epi_curves`, which covers the logistic regression applied at individual level to estimate probabilities that untested, clinically compatible cases are actually measles. This is the analysis described in the manuscript's appendix 1.
2. `age_at_inf`, which covers the smoothing process used to estimate the age distribution of infections by birth cohort, referenced in the manuscript's second appendix.
3. `demography`, which covers the multiple-regression with post-stratification methods applied to the DHS and MICS surveys to estimate birthrates and routine coverage. Outputs are like those visualized in the manuscript's Figure 1 for Lagos State.  