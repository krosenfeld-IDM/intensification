# intensification
Using state-level measles transmission models to compare Southern Nigeria's 2019 intensification of routine immunization to its mass vaccination campaigns since 2010.

This is a repository of Python 3.8 code associated with the manuscript [*Routine immunization
intensification, vaccination campaigns, and measles transmission in Southern Nigeria*](https://nthakkar.github.io/), 2025. In that paper, we fit mechanistic transmission models to surveillance and survey data to estimate catch-up vaccination events' effects on measles immunity. We find that the age-targeted 2019 effort was more than twice as effective per dose as the mass campaigns.

This repository focuses on the transmission model, and raw data (the surveys, line list surveillance data, etc.) are not included. Instead, the `_data/` directory includes the processed model inputs, like those visualized in the manuscript's first figure but for every southern state.

For the repository to function correctly, the scripts `GeneratePriors.py` and `GenerateStateSummary.py` have to be run first, in that order, and it's worth noting this will take about 20 minutes to run on a laptop. These scripts call the others as subprocesses, and they generate some serialized output used throughout the repo.

Then, scripts are set up to create the paper's figures (for Lagos State) by default. 
1. `VisualizeInputs.py` makes the paper's Figure 1.
2. `TransmissionModel.py` makes the paper's Figure 2.
3. `OutOfSampleTest.py` makes Figure 3.
4. `SIAImpactPosteriors.py` makes Figure 4.
5. `SIAImpactAnalysis.py` will make Figure 5.
6. `SurvivalPrior.py` will make Appendix 2, Figure 1.

Running these with a different state's name as a command line argument will produce the associated version of the figure for that state. For example, you can try `python TransmissionModel.py oyo`. 

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/NThakkar-IDM/intensification)