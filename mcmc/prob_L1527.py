import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import prob_emcee

def main():
    # ==== settings ====
    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcmc'
    datdir = os.path.join(datdir, 'L1527_run1')

    nwalkers = 40
    nsteps = 1000

    # 0.9 sec for each calculation

    source = 'L1527'

    parset = {
        # on/off, value, sigma, lower limit, upper limit
        'xc':   [1, 0, 0.02, -0.1, 0.1], # arcsec
        'yc':   [1, 0, 0.02, -0.1, 0.1],
        'pa':   [1, 0, 5, -30, 30], # degrees
        'inc':  [1, 87, 1, 75, 90],
        'R0':   [1, 60, 10, 20, 200],
        'tau0': [1, 0.35, 0.1, 0.01, 1.5], #0.1, 0.5
        'T0':   [1, 30, 5, 0, 150],
        'Hc':   [1, 20, 2, 0.1, 50],
        }

    # ==== run pipeline ====
    prob_emcee.basic_pipeline(source, parset, nwalkers, nsteps, datdir)


if __name__ == '__main__':
    main()

