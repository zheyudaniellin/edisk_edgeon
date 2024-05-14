# powerlaw.py
# just simple power laws for profiles
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
from scipy.integrate import quad
import sys
#sys.path.append('/home/zdl3gk/mainProjects/edisk_edgeon/modeling/mymodels')
sys.path.append('..')
import toolbox as tlx
au = tlx.natconst.au
rad = np.pi / 180.

class model(object):
    """ 
    class for calculations and input
    """
    def __init__(self, outdir, par):
        """ 
        Parameters
        ----------
        outdir : str
            directory for outputs 
        par : dict
            dictionary of input parameters:
        """
        self.outdir = outdir
        self.par = par
        self.dat = tlx.data.fieldData()

        self.gas_to_dust = 100

    def get_rho0(self):
        """ this value is used across calculations
        """
        self.rho0 = self.par['mstar'] * tlx.natconst.ms / np.pi / np.sqrt(2*np.pi) / (self.par['R0'] * au)**3

    # ==== opacity ====
    def get_opac(self):
        """ write the opacity 
        The kabs will be a default 1 because only tau0 matters
        use albedo to consider scattering
        """
        # quickly check if the number of elements for imwav, kabs, alb are the same
        if len(self.par['imwav']) != len(self.par['tau0']):
            raise ValueError('number of tau0 and imwav are not consistent')

        # prepare the opacity
        op = tlx.dust.opacity()
        op.set_ext('manual')
        op.w = np.zeros([len(self.par['imwav']) + 2])
        op.w[1:-1] = self.par['imwav']
        op.w[0] = min(self.par['imwav']) * 0.5
        op.w[-1] = max(self.par['imwav']) * 2

        op.kabs = np.zeros_like(op.w)
        op.kabs[1:-1] = np.array(self.par['tau0']) / self.rho0 / self.par['R0'] / au * self.gas_to_dust
        op.kabs[0] = op.kabs[1]
        op.kabs[-1] = op.kabs[-2]

        op.ksca = np.zeros_like(op.w)
        op.ksca[1:-1] = op.kabs[1:-1] * np.array(self.par['alb'])
        op.ksca[0] = op.ksca[1]
        op.ksca[-1] = op.ksca[-2]

        # prepare manager
        mop = tlx.dust.opacityManager()
        mop.add_opac(op)
        mop.write_dustopac(fdir=self.outdir)
        mop.write_opac(fdir=self.outdir)

        self.mop = mop

    # ==== grid parameters for the data ====
    def get_grid(self,
        xbound=[5*au, 300*au], nx=[128],
        ybound=[80*rad, 90*rad, 100*rad], ny=[64, 64],
        zbound=[0, 2*np.pi], nz=[64]):

        # setup the grid object
        grid = tlx.grid.regularGrid()
        grid.make_spatial(crd_sys='sph', act_dim=[1,1,0],
            xbound=xbound, nx=nx,  
            ybound=ybound, ny=ny,)

        grid.get_cell_center()

        grid.write_spatial(fdir=self.outdir)

        # wavelength
        grid.make_wavelength(wbound=[1, 1e4], nw=[20])
        grid.write_wavelength(fdir=self.outdir)

        self.grid = grid
        self.dat.set_grid(self.grid)

    def get_dustrho(self):
        """ produce dust density distribution 
        Lynden-Bell
        """
        # read in parameters
        Rc = self.par['Rc'] * au # characteristic radius
        R0 = self.par['R0'] * au # edge of the disk
        Hc = self.par['Hc'] * au
        hs = self.par['hs']
        p = self.par['p']

        # grid coordinates
        rr, tt, pp = np.meshgrid(self.grid.x, self.grid.y, self.grid.z, indexing='ij')
        cyrr = rr * np.sin(tt)
        zz = rr * np.cos(tt)

        # density
        hh = Hc * (cyrr / Rc)**hs
        rho = self.rho0 * (cyrr / R0)**(-p) * np.exp(-0.5 * (zz/hh)**2) / self.gas_to_dust

        # density cut-off
        reg = cyrr >= R0
        rho[reg] = 0.

        self.dat.dustrho = rho[...,None]
        reg = self.dat.dustrho <= 1e-30
        self.dat.dustrho[reg] = 1e-30

        self.dat.write_dustrho(fdir=self.outdir, binary=True)

    def get_dusttemp(self):
        """ produce dust temperature
        """
        R0 = self.par['R0'] * au
        T0 = self.par['T0']
        q = self.par['q']

        # grid coordinates
        rr, tt, pp = np.meshgrid(self.grid.x, self.grid.y, self.grid.z, indexing='ij')
        cyrr = rr * np.sin(tt)
        zz = rr * np.cos(tt)
 
        ndust = self.mop.nopac

        # temperature
        self.dat.dusttemp = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz,ndust])
        for ii in range(ndust):
            self.dat.dusttemp[...,ii] = T0 * (cyrr / R0)**(-q)

        self.dat.write_dusttemp(fdir=self.outdir, binary=True)

    def get_stars(self):
        """ produce the stars.inp file
        """
        src = tlx.radiation.discrete_stars()
        src.set_grid(self.dat.grid)

        src.add_star(tlx.natconst.rs, self.par['mstar']*tlx.natconst.ms, tlx.natconst.ts, [0,0,0])

        src.write_stars(fdir=self.outdir)

