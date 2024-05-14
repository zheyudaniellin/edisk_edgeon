"""
first test to create the pdspy model

Follow the default tutorial 

conda activate pdspy_env

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pdspy.modeling as modeling
import pdspy.interferometry as uv
import pdspy.dust as dust
import pdspy.gas as gas

# generate model
m = modeling.YSOModel()

# generate model grid
nr, ntheta, nphi = 100, 100, 2
rmin, rmax = 0.1, 300

m.set_spherical_grid(rmin, rmax, nr, ntheta, nphi, code="radmc3d")

# generate dust
dust_gen = dust.DustGenerator(dust.__path__[0]+"/data/diana_wice.hdf5")

a_max = 10 # microns
p = 3.5

d = dust_gen(a_max / 1e4, p) # dust_gen wants units of cm

# add different structures
m.add_star(mass=0.5, luminosity=1., temperature=4000.)
m.add_disk(mass=0.01, rmin=0.1, rmax=50., plrho=1., h0=0.1, plh=1., \
        dust=d)

# The below sets up the wavelength grid for RADMC3D.
m.grid.set_wavelength_grid(0.1, 1.0e5, 500, log=True)

# calculate temperature
# this line deletes the files that were used for radmc3d 
m.run_thermal(nphot=1e5, modified_random_walk=True, verbose=True, \
        setthreads=1, code="radmc3d")

# generate image
# pixelsize is in units of arcseconds.
# the code produces files for radmc3d, and then deletes it after we get the image
m.run_image(name="870um", nphot=1e5, npix=256, pixelsize=0.01, \
        lam="870", incl=45, pa=30, dpc=140, code="radmc3d", \
        verbose=True, setthreads=2)

# this one also produces files for radmc3d and then deletes it
m.run_visibilities(name="870um", nphot=1e5, npix=256, pixelsize=0.01, \
        lam="870", incl=45, pa=30, dpc=140, code="radmc3d", \
        verbose=True, setthreads=2)

# plotting image
plt.imshow(m.images["870um"].image[:,:,0,0], origin="lower", \
        interpolation="nearest")
plt.show()

# plot visibilities
m1d = uv.average(m.visibilities["870um"], gridsize=10000, binsize=3500, \
        radial=True)
plt.semilogx(m1d.uvdist, m1d.amp, "-")
plt.show()
