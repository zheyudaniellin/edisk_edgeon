# plot_obs.py
# just plot the image as a test
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import sys
sys.path.append('..')
import toolbox as tlx
au = tlx.natconst.au
rad = np.pi / 180

def plot1(im, north='up'):
    """ simple plot
    """
    ax = plt.gca()
    if north == 'right':
        pltx = im.grid.x
        plty = im.grid.y
        im2d = im.I[:,:,0]
    else:
        pltx = im.grid.y
        plty = im.grid.x
        im2d = im.I[:,:,0].T

    pc = ax.contourf(pltx/au, plty/au, im2d.T)

    clevs = im.I_rms[0] * np.array([3])
    cc = ax.contour(pltx/au, plty/au, im2d.T, clevs, colors='w')

    im.plot_beam(ax=ax, beamxy=(-100, -100), north=north)

    ax.plot([0], [0], 'w+')

    ax.set_aspect('equal')
    ax.invert_xaxis()
    plt.show()

def main():
    # ==== settings ====
    source = 'IRS7B-A'
    baseline = 'SBLB'
    imcor = 'image'

    # ==== read ====
    im = tlx.kits.easy_read(source, baseline=baseline, imcor=imcor)
#    im.trim(trimpar)

    plot1(im)

if __name__ == '__main__':
    main()

