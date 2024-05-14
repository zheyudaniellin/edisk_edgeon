""" determine the different radiation sources
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

from . import natconst, grid

# ==== sources ====
class baseSource(object):
    def __init__(self):
        pass

    def set_grid(self, grid):
        """ we need the wavelength grid
        """
        self.grid = grid

class discrete_stars(baseSource):
    def __init__(self, ):
        baseSource.__init__(self)       

        self.rstar = []
        self.mstar = []
        self.tstar = []
        self.pstar = []

    def add_star(self, r, m, t, p):
        """ add a single star to the collection
        Parameters
        ---------
        r : float
            radius of star
        m : float
            mass of star
        t : float
            temperature
        p : list of 3 floats
            position
        """
        self.rstar.append(r)
        self.mstar.append(m)
        self.tstar.append(t)
        if len(p) != 3:
            raise ValueError('the position should have 3 elements')
        self.pstar.append(p)

    def write_stars(self, fdir=None):
        """ write the stars.inp file
        """
        # quickly check if the number of stars are the same for each property

        # write file
        fname = 'stars.inp'
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        nstar = len(self.rstar)
        with open(fname, 'w') as f:
            f.write('%d\n'%2)
            f.write('%d %d\n'%(nstar, self.grid.nw))
            # write radius, mass, position
            for i in range(nstar):
                f.write('%.9e %.9e %.9e %.9e %.9e\n'%(self.rstar[i], self.mstar[i], self.pstar[i][0], self.pstar[i][1], self.pstar[i][2]))

            f.write('%s\n'%' ')

            # write wavlength
            for i in range(self.grid.nw):
                f.write('%.9e\n'%self.grid.w[i])

            f.write('%s\n'%' ')

            # simply to a black body function for now
            for i in range(nstar):
                f.write('%.9e\n'%(-self.tstar[i]))

    def read_stars(self):
        pass

class external(baseSource):
    def __init__(self):
        baseSource.__init__(self)

    def use_bby(self, T):
        """ some simple prescription using blackbody for the radiation field
        ergs / cm^2 / sec / Hz / steradian
        """
        self.I = planck_law(T, self.grid.w)

    def write(self, fdir=None):
        """ write the file for external radiation
        """
        # write file
        fname = 'external_source.inp'
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'w') as f:
            f.write('%d\n'%2)
            f.write('%d\n'%(self.grid.nw))

            # write wavlength
            for i in range(self.grid.nw):
                f.write('%.9e\n'%self.grid.w[i])

            # write intensity
            for i in range(self.grid.nw):
                f.write('%.9e\n'%self.I[i])

# ==== some simple prescriptions ====
def planck_law(T, wav_micron):
    """ planck function in cgs units
    """
    f = natconst.cc * 1e4 / wav_micron
    hnu = natconst.hh * f
    hnu3_c2 = natconst.hh * f**3 / natconst.cc**2
    x = hnu / natconst.kk / T
    return 2 * hnu3_c2 / (np.exp(x) - 1.)
