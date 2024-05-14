# some basic utilities for organizing inputs
import json
import matplotlib.pyplot as plt
import corner
import numpy as np
import os
import pdb
import scipy.integrate as integrate
from scipy.interpolate import interp1d
rad = np.pi / 180.

def organizepar(parset):
    def store_elem(elem, kname):
        if len(elem) != 5: # currently 5
            raise ValueError('number of inputs in parset is not correct')
        if elem[0] == 0: # turned off
            cset[kname] = elem[1]
        else:
            parC.append(elem[1])
            parW.append(elem[2])
            parlim.append([elem[3], elem[4]])
            parlab.append(kname)

    pkeys = list(parset.keys())
    pkeys.sort()
    npar = len(pkeys)
    cset = {}
    parC = []
    parW = []
    parlim = []
    parlab = []
    for ikey in pkeys:
        # the usual element
        if type(parset[ikey][0]) is not list:
            store_elem(parset[ikey], ikey)
        else: # nested elements
            n_elems = len(parset[ikey])
            for ii in range(n_elems):
                kname = '%s_%d'%(ikey, ii)
                store_elem(parset[ikey][ii], kname)

    return parC, parW, parlim, parlab, cset

# function to fetch parameter
def getpar(klab, par, parlab, cset, idat=None):
    # klab = name of parameter to be fetched
    # idat = integer number for the ith data
    if idat is not None:
        klab = '%s_%d'%(klab, idat)

    if klab in parlab:
        inx = parlab.index(klab)
        kpar = par[inx]
    elif klab in cset.keys():
        kpar = cset[klab]
    else:
        raise ValueError('parameter to be fetched does not exist: %s'%klab)
    return kpar


class parameter_manager(object):
    """ object to manage the parameters
    Use this to replace calling organizepar and getpar functions 
    """
    def __init__(self):
        pass

    def setup_parset(self, parset):
        """ give a dictionary and organize into lists
        """
        
        def store_elem(elem, kname):
            if len(elem) != 5: # currently 5
                raise ValueError('number of inputs in parset is not correct')
            if elem[0] == 0: # turned off
                cset[kname] = elem[1]
            else:
                parC.append(elem[1])
                parW.append(elem[2])
                parlim.append([elem[3], elem[4]])
                parlab.append(kname)

        pkeys = list(parset.keys())
        pkeys.sort()
        npar = len(pkeys)
        cset = {}
        parC = []
        parW = []
        parlim = []
        parlab = []
        for ikey in pkeys:
            # the usual element
            if type(parset[ikey][0]) is not list:
                store_elem(parset[ikey], ikey)
            else: # nested elements
                n_elems = len(parset[ikey])
                for ii in range(n_elems):
                    kname = '%s_%d'%(ikey, ii)
                    store_elem(parset[ikey][ii], kname)

        # keep a record of the original argument
        self.parset = parset

        # the starting value
        self.parC = parC

        # the width for the gaussian ball
        self.sigma = parW

        # the limits
        self.lim = parlim

        # name of the parameter
        self.lab = parlab

        # dictionary of constant values
        self.cset = cset

    def setup_parset_from_file(self, fdir=None):
        fname = 'parset.json'
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'r') as f:
            parset = json.load(f)

        self.setup_parset(parset)

    def get_value(self, klab, par, idat=None):
        """ fetch the value of a parameter
        """
        if idat is not None:
            klab = '%s_%d'%(klab, idat)

        if klab in self.lab:
            inx = self.lab.index(klab)
            kpar = par[inx]
        elif klab in self.cset.keys():
            kpar = self.cset[klab]
        else:
            raise ValueError('parameter to be fetched does not exist: %s'%klab)
        return kpar

    def write_parset(self, fdir=None):
        """ write the parameter settings file
        sometimes the ordering of the dictionary items may vary, 
        it's better to also write explicitly the order of the fitted parameters
        """
        # write the parameter settings file
        fname = 'parset.json'
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'w') as f:
            json.dump(self.parset, f, indent=4)

        # write the order
        fname = 'par_order.txt'
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'w') as f:
            for ival in self.lab:
                f.write(ival + '\n')

    def write_bestfit(self, bestfit, fdir=None):
        """ write a file that containts the best fit values
        """
        if len(bestfit) != len(self.lab):
            raise ValueError('number of parameters incorrect')

        for i in range(len(bestfit)):
            if (bestfit[i] < self.lim[i][0]) | (besfit[i] > self.lim[i][1]):
                raise ValueError('the best fit values are not in the limit')

        fname = 'bestfit.txt'

        # not done yet

    def write_par_instance(self, par, fdir=None):
        """ write the parameters that were actually used in a certain iteration
        """
        fname = 'par_used.txt'
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        out = {}
        for ival in self.lab:
            out[ival] = self.get_value(self, ival, par)

        with open(fname, 'w') as f:
            json.dump(out, f, indent=4)

def write_radpar(radpar, outdir):
    """ write radmc3d.inp file """
    fname = os.path.join(outdir, 'radmc3d.inp')
    with open(fname, 'w') as wfile:
        for ikey in radpar.keys():
            wfile.write('%s = %s\n'%(ikey, radpar[ikey]))
        wfile.close()

def spread_gaussian_ball(parC, parW, nwalkers):
    """ produce starting positions for different walkers
    parC : 1d list or ndarray
    parW : 1d list or ndarray
        the standard deviation of the distribution
    nwalkers : int
    """
    ndim = len(parC)
    pos = np.zeros([nwalkers, ndim])
    for i in range(ndim):
        pos[:,i] = np.random.normal(parC[i], parW[i], nwalkers)

    return pos

class samples_manager(object):
    """ class to manager the samples
    """
    def __init__(self):
        pass

    def read_samples(self, fdir=None):
        """ read the samples result
        """
        fname = 'samples.npy'
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        self.samples = np.load(fname)

        self.nsteps, self.nwalkers, self.ndim = self.samples.shape

    def read_par(self, fdir=''):
        """ read the parameter names
        """
        fname = 'par_order.txt'
        fname = os.path.join(fdir, fname)

        par = []
        with open(fname, 'r') as f:
            for iline in f:
                par.append(iline.rstrip('\n'))
        self.lab = par

    def get_flat(self, burnsteps=100):
        """ flatten the samples after removing some steps as burn-in
        """
        if burnsteps >= self.nsteps:
            raise ValueError('number of steps to discard for burn-in is larger than the number of steps')

        s = self.samples[burnsteps:,:,:]
        dim = s.shape
        self.flat = np.reshape(s, (dim[0]*dim[1], dim[2]))

    def trim_walkers(self, reg):
        """ take out some walkers
        reg : list of bool
            the walkers to keep
        """
        self.samples = self.samples[:,reg,:]
        self.nwalkers = self.samples.shape[1]

    def get_values(self, quantiles=[16, 50, 84]):
        """ calculate the values
        """
        self.val = np.zeros([self.ndim])
        self.val_p = np.zeros_like(self.val)
        self.val_n = np.zeros_like(self.val)
        for i in range(self.ndim):
            per = np.percentile(self.flat[:,i], quantiles)
            self.val[i] = per[1]
            self.val_p[i] = per[2] - per[1]
            self.val_n[i] = per[0] - per[1]

    def write_values(self, fname='bestfit_values.txt', fdir=''):
        """
        write the values to a file
        only possible after calling self.get_values()
        """

        out = np.zeros([3, len(self.val)])
        out[0,:] = self.val_n
        out[1,:] = self.val
        out[2,:] = self.val_p

        fname = os.path.join(fdir, fname)
        np.savetxt(fname, out.T)

    def read_values(self, fname='bestfit_values.txt', fdir=''):
        fname = os.path.join(fdir, fname)
        out = np.loadtxt(fname).T
        self.val_n = out[0,:]
        self.val = out[1,:]
        self.val_p = out[2,:]

    def fetch_value(self, ikey):
        """
        convenience function to get the value
        """
        inx = self.lab.index(ikey)
        return val_n[inx], val[inx], val_p[inx]

    # ==== default plottings ====
    def print_solutions(self):
        for i in range(len(self.lab)):
            txt = self.lab[i] + r' = %.2f +%.4f, %.4f'%(self.val[i], self.val_p[i], self.val_n[i])
            print(txt)

    def plot_corner(self):
        """ some default plot for the cornder
        """
        fig = corner.corner(self.flat, labels=self.lab, 
            quantiles=[0.16, 0.5, 0.84], 
            show_titles=True, )

        return fig

    def plot_steps(self):
        nrow, ncol = self.ndim, 1
        fig, axes = plt.subplots(nrow,ncol,sharex=True,sharey=False, squeeze=True, figsize=(10,6))
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(self.samples[:,:,i], 'k', alpha=0.3)
            ax.set_xlim(0, self.nsteps)
            ax.set_ylabel(self.lab[i])

        return fig, axes

