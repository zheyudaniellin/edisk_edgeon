# prob_IRS7B-A.py
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import prob_emcee

def main():
    # ==== settings ====
    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcmc'
    datdir = os.path.join(datdir, 'IRS7B-A_run1')

    nwalkers = 40
    nsteps = 1000

    # 0.9 sec for each calculation

    source = 'IRS7B-A'

    parset = {
        # on/off, value, sigma, lower limit, upper limit
        'xc':   [1, 0, 0.02, -0.1, 0.1], # arcsec
        'yc':   [1, 0, 0.02, -0.1, 0.1],
        'pa':   [1, 115, 5, 50, 150], # degrees
        'inc':  [1, 70, 5, 45, 90],
        'R0':   [1, 70, 10, 20, 200],
        'tau0': [1, 0.35, 0.1, 0.01, 1.5], #0.1, 0.5
        'T0':   [1, 100, 10, 0, 250],
        'Hc':   [1, 10, 2, 0.1, 50],
        }

    # ==== run pipeline ====
    prob_emcee.basic_pipeline(source, parset, nwalkers, nsteps, datdir)


if __name__ == '__main__':
    main()

