# prob_emcee.py
# main parameters
# x0, y0, inc, pa, density, temperature
# ---- pseudo code ----
# - read the observations
# - determine pixel size from observations
# - get the interpolation from observations
# - create common stuff for radmc3d: directory, grid
# - iterate
#   - rewrite density, temperature
#   - image with inc
#   - match sky with x0, y0, pa
#   - calculate difference
# ----
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import emcee
import pdb
from scipy.interpolate import RectBivariateSpline
import utils
import sys
sys.path.append('..')
import toolbox as tlx
sys.path.append('../modeling/mymodels')
import powerlaw as model
au = tlx.natconst.au
rad = np.pi / 180

def resample_obs(im, npix=None, frac_of_beam=1):
    """ resample the pixels
    npix : int
        sample the image every so many pixels
    frac_of_beam : float
        The fraction of a beam desired (must be greater than the pixel size). If this value equals 2, then it's Nyquist sampling.
        Only used if npix is not provided
    """
    if npix is None:
        fwhm = im.psfarg['bmin'] * im.grid.dpc * au
        if (frac_of_beam < im.grid.dx/fwhm):
            raise ValueError('we cannot sample less than the pixel size')

        npix = int(np.floor(fwhm * frac_of_beam / im.grid.dx))

    i = np.linspace(0, im.grid.nx-1, im.grid.nx//npix, dtype=int)
    j = np.linspace(0, im.grid.ny-1, im.grid.ny//npix, dtype=int)

    im.grid.x = im.grid.x[i]
    im.grid.y = im.grid.y[j]

    im.grid.dx = abs(im.grid.x[0] - im.grid.x[1])
    im.grid.dy = abs(im.grid.y[0] - im.grid.y[0])
    im.grid.nx = len(im.grid.x)
    im.grid.ny = len(im.grid.y)

    for iquant in im.quantity:
        setattr(im, iquant, getattr(im, iquant)[np.ix_(i,j)] )

    return im

def prep_obs(source, baseline='SBLB', imcor='image'):
    """ 
    get the tlx.image.intensity for the observations
    """
    # read the observations
    im = tlx.kits.easy_read(source, baseline=baseline, imcor=imcor, 
        apply_center=True, apply_trim=True)

    # sometimes the pixel over samples the beam too much
    im = resample_obs(im, npix=None, frac_of_beam=0.5)

    # create the interpolation object
    fnobs = RectBivariateSpline(im.grid.x, im.grid.y, im.I[:,:,0])
   
    return im, fnobs

def prep_model(datdir, parm, obs):
    """ prepare the directory and some files that are common to all modeling
    """
    # determine the imaging size which affects zoomau
    # give the length that covers the major axis of the disk. 
    # simply use the maximal length of the grid from obs
    imsizeau = max(abs(obs.grid.x[-1] - obs.grid.x[0]), abs(obs.grid.y[-1] - obs.grid.y[0])) / au # [au]

    # now hard code the length that covers the minor axis of the disk
    imsizeau_y = imsizeau / 2 # [au]
    
    # some parameters depend on the observations
    dx = obs.grid.dx / au
    npix = np.ceil(imsizeau / dx)

    # create a directory to run the model
    modeldir = os.path.join(datdir, 'sim0')
    os.system('rm -rf %s'%modeldir)
    os.system('mkdir %s'%modeldir)

    # default parameters
    par = {
        # grid parameters
        'nx': [64],
        'xbound': [1*au, imsizeau/2*1.1*au],
        'ny': [16, 32, 32, 16], 
        'ybound': [70*rad, 85*rad, 90*rad, 95*rad, 110*rad], 
        'nz': [32], 
        # imaging wavelength
        'imwav': [1300], 
        'inc': 45, 
        'nphot_scat': 1e3, 
        'imsizeau': imsizeau,
        'npix': npix, 
        'imsizeau_y': imsizeau_y, 
        # dust structure
        'R0': 300, # edge-of the disk
        'Rc': 100, # arbitrary radius for Hc
        'tau0': [0.1], 
        'alb': [0], 
        'p': 3, 
        'Hc': 1, 
#        'hs': 1.25, 
        'T0': 30, 
        'q': 0.5, 
        'mstar': 1.5, 
        }
    par['hs'] = 1.5 - par['q'] / 2.

    # change the default parameters to cset parameters since it's shared
    for ikey in parm.cset.keys():
        if ikey in ['xc', 'yc']: # ignore these
            continue
        elif ikey in ['tau0', 'alb']: # these are wavelength dependent
            par[ikey] = [parm.cset[ikey]]
        else:
            par[ikey] = parm.cset[ikey]

    # prepare the model object
    mod = model.model(modeldir, par)
    mod.setthreads = 1
    mod.do_optical_depth = False

    return mod

def get_model_image(par, parm, mod, dpc):
    """ run radmc3d
    """
    modkeys = mod.par.keys()

    # modify the parameters
    for ikey in parm.lab:
        if ikey in ['xc', 'yc', 'pa']:
            continue
        elif ikey in ['tau0', 'alb']: 
            # these keys are wavelength dependent for the model, 
            # but ignoring that for mcmc here
            mod.par[ikey] = [parm.get_value(ikey, par)]
        else:
            mod.par[ikey] = parm.get_value(ikey, par) * 1

    # prep
    mod.get_rho0()

    # create the opacity 
    mod.get_opac()

    # create the grid
    # determine a suitable ybound
    # Rc is definitely fixed
    Hc = parm.get_value('Hc', par)
    Rc = mod.par['Rc']
    dy1 = np.arctan(Hc / Rc)
    dy2 = np.arctan(10 * Hc / Rc)
    mod.par['ybound'] = [90*rad-dy2, 90*rad-dy1, 90*rad, 90*rad+dy1, 90*rad+dy2]
    mod.get_grid(
        xbound=mod.par['xbound'], nx=mod.par['nx'], 
        ybound=mod.par['ybound'], ny=mod.par['ny'],
        nz=mod.par['nz'])

    # get the dust temperature
    mod.get_dusttemp()

    # get the dust density
    mod.get_dustrho()

    # get the star input
    mod.get_stars()

    # write the imaging wavelength
    tlx.image.write_camera_wavelength(mod.par['imwav'], fdir=mod.outdir)

    # calculate image
    cwd = os.getcwd()

    os.chdir(mod.outdir)

    radpar = {
        'nphot_scat': '%d'%(mod.par['nphot_scat']),
        'istar_sphere': '0',
        'setthreads': '1',
        'mc_scat_maxtauabs': '30',
        'scattering_mode_max' : '1',
        'alignment_mode' : 0
        }
    tlx.utils.write_radpar(radpar, mod.outdir)

    arg = {'npix':mod.par['npix'], 
        'sizeau':mod.par['imsizeau'], 
        'stokes':False, 'circ':False, 
        'secondorder':False, 
        'nphot_scat':mod.par['nphot_scat'], 
        'incl':mod.par['inc'], 
        'zoomau': '%d %d %d %d'%(-mod.par['imsizeau']/2, mod.par['imsizeau']/2, -mod.par['imsizeau_y']/2, mod.par['imsizeau_y']/2)}

    com = tlx.image.commandImage()
    for ikey in arg.keys():
        setattr(com, ikey, arg[ikey])
    com.set_wavelength(loadlambda=True)

    com.get_command()

    com.make(fname='myimage.out')

    os.chdir(cwd)

    # read image
    fname = os.path.join(mod.outdir, 'myimage.out')
    im = tlx.image.intensity()
    im.read(fname)
    im.grid.set_dpc(dpc)

    return im

def ln_prior(par, parm):
    """ 
    check if the parameters are within the boundaries
    """
    fail_prior = False

    for i in range(len(par)):
        if (par[i] < parm.lim[i][0]) | (par[i] > parm.lim[i][1]):
            fail_prior = True

    return fail_prior

def match_sim_with_obs(obs, im, inp):
    """ resample the model image so that is matches with the points of the observations 
    """
    pa = inp['pa']
    xc = inp['xc'] * obs.grid.dpc * au
    yc = inp['yc'] * obs.grid.dpc * au

    # convolve the model image
    bpa = obs.psfarg['bpa'] - pa
    conv = im.convolve_gaussian(obs.psfarg['bmaj'], obs.psfarg['bmin'], bpa)
    conv.convert_stokes_unit('jypbeam')

    # create the interpolation object
    fn = RectBivariateSpline(conv.grid.x, conv.grid.y, conv.I[:,:,0])

    # calculate the coordinates in the plane-of-sky
    xx, yy = np.meshgrid(obs.grid.x - xc, obs.grid.y - yc, indexing='ij')
    xsky = np.cos(pa*rad) * xx + np.sin(pa*rad) * yy
    ysky = -np.sin(pa*rad) * xx + np.cos(pa*rad) * yy

    im1d = fn(xsky, ysky, grid=False)

    im2d = np.reshape(im1d, (obs.grid.nx, obs.grid.ny))

    return obs.I[:,:,0], im2d

def get_chi(par, parm, obs, mod, just_give_chi=True):
    # get the model
    im = get_model_image(par, parm, mod, obs.grid.dpc)

    # now we have to match the resolution and coordinates in the sky
    pa = parm.get_value('pa', par)
    xc = parm.get_value('xc', par)
    yc = parm.get_value('yc', par)
    obs2d, im2d = match_sim_with_obs(obs, im, {'pa':pa, 'xc':xc, 'yc':yc})

    # calculate the reduced chi-squared
    chi = np.sum( ((obs2d - im2d) / obs.I_rms[0])**2 )

    if just_give_chi:
        return chi
    else:
        return {'chi':chi, 'rms':obs.I_rms[0], 
            'x':obs.grid.x-xc, 'y':obs.grid.y-yc, 'dpc':obs.grid.dpc, 
            'obs2d':obs2d, 'im2d':im2d}

def ln_prob(par, parm, obs, mod):
    """ calculate the likelihood
    """
    fail_prior = ln_prior(par, parm)

    if fail_prior:
        return -np.inf
    else:
        chi = get_chi(par, parm, obs, mod)
        return - chi / 2.

def basic_pipeline(source, parset, nwalkers, nsteps, datdir):
    """
    basic pipeline for each source
    """
    # ==== prepare the directory for these calculations ====
    os.system('rm -rf %s'%datdir)
    os.system('mkdir %s '%datdir)

    # ==== prepare the parameters and write ====
    parm = utils.parameter_manager()
    parm.setup_parset(parset)
    parm.write_parset(fdir=datdir)

    # ==== prepare the observations ====
    obs, fnobs = prep_obs(source, imcor='image', baseline='SBLB')

    # ==== prepare the modeling ====
    mod = prep_model(datdir, parm, obs)

    # ==== begin emcee sampling ====
    ndim = len(parm.parC)
    pos = utils.spread_gaussian_ball(parm.parC, parm.sigma, nwalkers)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, ln_prob, args=(parm, obs, mod)
        )

    sampler.run_mcmc(pos, nsteps, progress=False)

    # ==== output ====
    samples = sampler.get_chain(flat=False)
    fname = os.path.join(datdir, 'samples.npy')
    np.save(fname, samples)

def main():
    # ==== settings ====
    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcmc'
    datdir = os.path.join(datdir, 'run1')

    nwalkers = 40
    nsteps = 2000

    # 0.5 sec for each calculation

    source = 'IRAS04302'

    parset = {
        # on/off, value, sigma, lower limit, upper limit
        'xc': 	[1, 0, 0.02, -0.1, 0.1], # arcsec
        'yc': 	[1, 0, 0.02, -0.1, 0.1], 
        'pa': 	[1, -5.23, 5, -30, 30], # degrees
        'inc':	[1, 86, 1, 80, 90], 
        'R0':	[1, 300, 10, 20, 400], 
        'tau0': [1, 0.35, 0.03, 0.1, 0.5], #0.1, 0.5
        'T0': 	[1, 7, 1, 0, 20], 
        'Hc': 	[1, 6, 2, 0.1, 20], 
        }

    # ==== run pipeline ====
    basic_pipeline(source, parset, nwalkers, nsteps, datdir)


if __name__ == '__main__':
    main()

