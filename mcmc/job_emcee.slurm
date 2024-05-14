#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=50:00:00
#SBATCH --output=job_log.out
#SBATCH --partition=standard
#SBATCH --account=StarFormationTheory

python prob_emcee.py


