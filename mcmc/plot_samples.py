# plot_samples.py
# plot the best fit parameters
# output the parameters
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pdb
import corner
import utils

def main():
    # ==== settings ====
    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcmc/'
    datdir = os.path.join(datdir, 'run1')
#    datdir = os.path.join(datdir, 'L1527_run1')
#    datdir = os.path.join(datdir, 'GSS30IRS3_run1')
#    datdir = os.path.join(datdir, 'IRS7B-A_run1')

    # read the samples
    sm = utils.samples_manager()
    sm.read_par(fdir=datdir)
    sm.read_samples(fdir=datdir)

    sm.get_flat(burnsteps=500)
    sm.get_values()

    # read the parameters
    parm = utils.parameter_manager()
    parm.setup_parset_from_file(fdir=datdir)

#    parm.write_bestfit(sm.val, fdir=datdir)

    # ==== plotting ====
    sm.print_solutions()

    sm.plot_steps()
    plt.show()

    sm.plot_corner()
    plt.show()

    pdb.set_trace()

if __name__ == '__main__': 
    main()

