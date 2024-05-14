#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=50:00:00
#SBATCH --output=job_log_GSS30IRS3.out
#SBATCH --partition=standard
#SBATCH --account=StarFormationTheory

python prob_GSS30IRS3.py


