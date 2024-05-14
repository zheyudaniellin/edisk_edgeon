# plot_final.py
# produce the final calculation and also be able to plot it as an inspection
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pdb
from scipy.interpolate import RectBivariateSpline
import pickle
from prob_emcee import prep_obs, prep_model, get_chi, match_sim_with_obs
import utils
import sys
sys.path.append('..')
import toolbox as tlx
au = tlx.natconst.au
rad = np.pi / 180

def plot2d(out):
    """ plot the matched results in the image plane
    """
    nrow = 2
    ncol = 2
    fig, axgrid = plt.subplots(nrow,ncol,sharex=True,sharey=True,
        squeeze=False, figsize=(13, 8))
    axes = axgrid.flatten()

    x = out['x'] / au
    y = out['y'] / au

    # common color scale
    vmin = 0
    vmax = np.max(out['obs2d'])

    clevs = np.max(out['obs2d']) * np.linspace(0, 0.95, 5)
    clevs[0] = out['rms'] * 5

    # observations
    ax = axes[0]
    pc = ax.contourf(x, y, out['obs2d'].T, 32, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(pc, ax=ax)
    cc = ax.contour(x, y, out['obs2d'].T, clevs, colors='w')

    # model
    ax = axes[1]
    pc = ax.contourf(x, y, out['im2d'].T, 32, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(pc, ax=ax)
    cc = ax.contour(x, y, out['im2d'].T, clevs, colors='w')

    # residual
    ax = axes[2]
    res = out['obs2d'] - out['im2d']
    vmax = np.max(abs(res))
    pc = ax.contourf(x, y, res.T, 32, 
        cmap=plt.cm.RdBu_r, vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(pc, ax=ax)

    ax = axes[3]
    c1 = ax.contour(x, y, out['im2d'].T, clevs, colors='C0')
    c2 = ax.contour(x, y, out['obs2d'].T, clevs, colors='k')

    for ax in axes:
        ax.set_aspect('equal')
    fig.tight_layout()
    plt.show()

def plot_cut(mat, track):
    """ compare the cuts along an arbitrary direction
    """
    xc = track['xc']
    yc = track['yc']
    theta = track['theta']
    lmax = track['lmax']
    nl = track['nl']

    # determine the sky plane locations
    laxis = np.linspace(-lmax, lmax, nl)

    x = xc + laxis * np.cos(theta)
    y = yc + laxis * np.sin(theta)

    # interpolate profile
    obs = mat['fnobs'](x, y, grid=False)
    im = mat['fnim'](x, y, grid=False)

    # plot the results
    ax = plt.gca()
    fac = 1e-3
#    ax.fill_between(laxis/au, obs/fac-mat['rms']/fac, obs/fac+mat['rms']/fac, color='grey', alpha=0.3)
    ax.plot(laxis/au, obs/1e-3, 'k', label='Obs')
    ax.plot(laxis/au, im/1e-3, 'C0', label='Model')
    ax.fill_between(laxis/au, laxis*0, laxis*0+mat['rms']/1e-3*5, color='grey', alpha=0.3)
    ax.set_xlabel('position [au]')
    ax.set_ylabel('mJy/beam')
    ax.legend()

    plt.show()

def do_last_model():
    # ==== settings ====
    source = 'IRAS04302'

    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcmc'
    datdir = os.path.join(datdir, '%s_run1'%source)

    # ==== prepare the parameters and write ====
    parm = utils.parameter_manager()
    parm.setup_parset_from_file(fdir=datdir)

    # ==== retrieve the good values ====
    sm = utils.samples_manager()
    sm.read_par(fdir=datdir)
    sm.read_samples(fdir=datdir)

    sm.get_flat(burnsteps=500)
    sm.get_values()

    # ==== prepare the observations ====
    obs, fnobs = prep_obs(source, imcor='image', baseline='SBLB')

    # ==== prepare the modeling ====
    mod = prep_model(datdir, parm, obs)

    # ==== do the calculation ====
    out = get_chi(sm.val, parm, obs, mod, just_give_chi=False)

    fname = os.path.join(datdir, 'matched.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(out,f)

    # ==== output the best fit parameters ====
    sm.write_values(fdir=datdir)

def main():
    # ==== settings ====
    source = 'IRAS04302'
#    source = 'L1527'
#    source = 'GSS30IRS3'
#    source = 'IRS7B-A'

    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcmc'
    datdir = os.path.join(datdir, '%s_run1'%source)

    # ==== prepare the parameters ====
    parm = utils.parameter_manager()
    parm.setup_parset_from_file(fdir=datdir)

    # ==== prepare the observations ====
    obs, fnobs = prep_obs(source, imcor='image', baseline='SBLB')

    # basic data
    dpc = tlx.kits.fetch_distance(source)

    # ==== read model image ====
    im = tlx.image.intensity()
    fname = os.path.join(datdir, 'sim0', 'myimage.out')
    im.read(fname)
    im.grid.dpc = dpc

    # ==== read the matched data ====
    fname = os.path.join(datdir, 'matched.pickle')
    with open(fname, 'rb') as f:
        mat = pickle.load(f)

    # ==== read the best fit parameters ====
    xc = -0.03
    yc = -0.02
    pa = -5.3 * rad

    # ==== prepare a few interpolation objects ====
    mat['fnobs'] = RectBivariateSpline(mat['x'], mat['y'], mat['obs2d'])
    mat['fnim'] = RectBivariateSpline(mat['x'], mat['y'], mat['im2d'])

    mat['fwhm'] = np.sqrt(obs.psfarg['bmaj'] * obs.psfarg['bmin'])

    # ==== plotting ====
    plot2d(mat)

    stepsize = np.sqrt(obs.psfarg['bmaj']*obs.psfarg['bmin']) * dpc * au / 5

    # major axis
    lmax = 350 * au
    nl = int( np.ceil(2*lmax / stepsize))
    track = {'xc':xc, 'yc':yc, 'theta':pa, 'nl':nl, 'lmax':lmax}
    plot_cut(mat, track)

    # minor axis
    track['lmax'] = 100 * au
    track['nl'] = int( np.ceil(2* track['lmax'] / stepsize))
    track['theta'] = pa + 90*rad
    plot_cut(mat, track)

    pdb.set_trace()

if __name__ == '__main__':
    do_last_model()
    main()

