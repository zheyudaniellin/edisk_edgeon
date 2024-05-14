# plot_image.py
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import astropy.visualization as vsl
import pdb
import sys
sys.path.append('..')
import toolbox as tlx
au = tlx.natconst.au
rad = np.pi / 180

def plot1(im, opt):
    nrow, ncol = 2, 1
    fig, axgrid = plt.subplots(nrow,ncol,sharex=True,sharey=True)
    axes = axgrid.flatten()

    pltx = im.grid.x/au
    plty = im.grid.y/au

    ax = axes[0]
    im2d = im.I[:,:,0]
#    im2d = im.tb[:,:,0]
    norm = vsl.ImageNormalize(im2d, stretch=vsl.SqrtStretch(),
#        interval=vsl.ManualInterval(1e-4, None))
        interval=vsl.MinMaxInterval())
    pc = ax.contourf(pltx, plty, im2d.T, 32, norm=norm)
    cb = plt.colorbar(pc, ax=ax)

    # optical depth
    ax = axes[1]
    im2d = opt.image[:,:,0]
    norm = vsl.ImageNormalize(im2d, stretch=vsl.SqrtStretch(), 
        interval=vsl.ManualInterval(1e-4, None))
    pc = ax.pcolormesh(pltx, plty, im2d.T, norm=norm)
    clevs = np.array([0.1, 1, 5])
    clines = ['--', '-']
    cc = ax.contour(pltx, plty, im2d.T, clevs, colors='w', linestyles=clines)
    cb = plt.colorbar(pc, ax=ax)

    for i, ax in enumerate(axes):
        ax.set_aspect('equal')

    fig.tight_layout()
    plt.show()

def main():
    # ==== settings ====
    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/modeling/'
    datdir = os.path.join(datdir, 'powerlaw1')
    dpc = 160

    # ==== read image ====
    fname = os.path.join(datdir, 'myimage.out')
    im = tlx.image.intensity()
    im.read(fname)
    im.grid.set_dpc(dpc)
    im.get_tb()

    fname = os.path.join(datdir, 'myopt.out')
    opt = tlx.image.opticalDepth()
    opt.read(fname)
    opt.grid.set_dpc(dpc)

    conv = im.convolve_gaussian(0.05, 0.05, 0)
    conv.convert_stokes_unit('jypbeam')
    conv.get_tb()

    # ==== plotting ====
#    plot1(im, opt)
    plot1(conv, opt)

if __name__ == '__main__':
    main()

