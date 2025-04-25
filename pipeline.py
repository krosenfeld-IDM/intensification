#!/usr/bin/env python3
"""
Pipeline module for intensification workflow.
"""
import os
import subprocess
import argparse
import pandas as pd

# Ordered scripts for pipeline
PRIOR_SCRIPTS = [
    'GeneratePriors.py',
    'GenerateStateSummary.py'
]
FIGURE_SCRIPTS = [
    'VisualizeInputs.py',
    'TransmissionModel.py',
    'OutOfSampleTest.py',
    'SIAImpactPosteriors.py',
    'SIAImpactAnalysis.py',
    'SurvivalPrior.py'
]


def load_states(data_dir='_data'):
    """
    Load southern states from states_and_regions.csv.
    """
    path = os.path.join(data_dir, 'states_and_regions.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"State file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    for col in ('region', 'state'):
        if col not in df.columns:
            raise ValueError(f"State file missing required column: '{col}'")
    states = df[df['region'].str.startswith('south')]['state'].tolist()
    if not states:
        raise ValueError("No southern states found in state file")
    return states


def validate_script(script):
    """
    Ensure script file exists in current dir.
    """
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Script file not found: {script}")


def run_script(script, args=None, verbose=False):
    """
    Run a Python script via subprocess, raising on failure.
    """
    validate_script(script)
    cmd = ['python', script] + (args or [])
    result = subprocess.run(
        cmd,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script {script} failed with exit code {result.returncode}")


def generate_priors(verbose=False):
    """Run prior generation scripts."""
    for s in PRIOR_SCRIPTS:
        run_script(s, verbose=verbose)


def generate_state_summary(verbose=False):
    """Run state summary script."""
    for s in ['GenerateStateSummary.py']:
        run_script(s, verbose=verbose)


def generate_figures(state, data_dir='_data', verbose=False):
    """Run all figure scripts for a given state"""
    states = load_states(data_dir)
    if state not in states:
        raise ValueError(f"State '{state}' not found in southern states list")
    for script in FIGURE_SCRIPTS:
        args = [state]
        if script == 'SurvivalPrior.py':
            args.append('-s')
        run_script(script, args=args, verbose=verbose)


def run_all(state, data_dir='_data', verbose=False):
    """Run full pipeline: priors, summary, and figures"""
    print("Starting full pipeline...")
    generate_priors(verbose)
    generate_state_summary(verbose)
    generate_figures(state, data_dir, verbose)
    print("Pipeline completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline for intensification workflow'
    )
    parser.add_argument('step', choices=['priors', 'summary', 'figures', 'all'])
    parser.add_argument('--state', default='lagos')
    parser.add_argument('--data-dir', default='_data')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.step == 'priors':
        generate_priors(verbose=args.verbose)
    elif args.step == 'summary':
        generate_state_summary(verbose=args.verbose)
    elif args.step == 'figures':
        generate_figures(args.state, args.data_dir, args.verbose)
    elif args.step == 'all':
        run_all(args.state, args.data_dir, args.verbose)


if __name__ == '__main__':
    main()
