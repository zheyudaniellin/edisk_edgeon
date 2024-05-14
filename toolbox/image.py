"""
new version of image.py

In this one, I'll combine imageCut and intensity as the same thing

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pdb
import copy
import subprocess
from astropy.io import fits
from scipy import signal
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.ndimage import map_coordinates
from . import natconst, grid

# ==== 
# image grid
# ====
class base_grid(object):
    """ image grid object
    It will always have a 1d wavelength grid
    the spatial grid can be 1d or 2d
    """
    def __init__(self):
        pass

    def set_frequency(self, f):
        self.f = f

    def get_frequency(self):
        self.f = natconst.cc * 1e4 / self.w

    def set_wavelength(self, w):
        self.w = w
        self.nw = len(self.w)

    def get_wavelength(self):
        self.w = natconst.cc * 1e4 / self.f
        self.nw = len(self.w)

    def set_dpc(self, dpc):
        """ distance in pc, optional
        """
        self.dpc = dpc

    def set_restfreq(self, restfreq):
        """ set the rest frequency for a line
        """
        self.restfreq = restfreq

    def get_velocity(self):
        """ get the line of sight velocity for lines
        in cm/s
        """
        if hasattr(self, 'f') is False:
            self.get_freqeuncy()

        if hasattr(self, 'restfreq') is False:
            raise ValueError('restfreq must be known to get the velocity')

        self.v = natconst.cc * (self.restfreq - self.f) / self.restfreq

    def set_psfarg(self, psfarg):
        """
        experimenting with placing psfarg in the grid object
        However, the bmaj and bmin should always be in arcsec

        let bmaj and bmin be constant of frequency
        """
        self.psfarg = psfarg

    def set_spatial_unit(self, unit):
        if unit not in ['cm', 'au', 'arcsec']:
            raise ValueError('argument for the spatial unit unknown')

        self.spatial_unit = unit

    def set_xy(self, x, y, xy_type):
        """
        The sky coordinates of each point relative to a center
        """
        self.x = x
        self.nx = len(self.x)

        self.y = y
        self.ny = len(self.y)

        self.xy_type = xy_type

        # quick check
        if self.xy_type == 'flat':
            if self.nx != self.ny:
                raise ValueError('The number of points for x and that for y should be the same if xy_type is flat.')

    def set_dec_ra(self, dec, ra):
        """
        The absolute sky coordinates. 
        Note that dec corresponds to x and ra corresponds to y
        """
        self.ra = ra
        self.dec = dec

        # check if the dimensions of ra, dec are the same as y, x respectively
        if len(self.x) != len(self.dec):
            raise ValueError('The size of x should be the same as the size of dec')
        if len(self.y) != len(self.ra):
            raise ValueError('The size of y should be the same as the size of dec')

    def get_spatial_unit_conversion(self, unit):
        """
        It's useful to have this function to calculate the factor for the conversion factor given the desired units

        """
        if self.spatial_unit == 'cm':
            if unit == 'au':
                fac = 1. / natconst.au
            elif unit == 'arcsec':
                fac = 1. / natconst.au / self.dpc
            else:
                raise ValueError('The argument for the spatial unit unknown.')
        elif self.spatial_unit == 'au':
            if unit == 'cm':
                fac = natconst.au
            elif unit == 'arcsec':
                fac = 1. / self.dpc
            else:
                raise ValueError('The argument for the spatial unit unknown.')

        elif self.spatial_unit == 'arcsec':
            if unit == 'cm':
                fac = natconst.au * self.dpc
            elif unit == 'au':
                fac = self.dpc
            else:
                raise ValueError('The argument for the spatial unit unknown.')

        else:
            raise ValueError('internal error: spatial_unit unknown')

        return fac

    def convert_spatial_unit(self, unit):
        """
        Parameters
        ----------
        unit : str
        """
        if unit not in ['cm', 'au', 'arcsec']:
            raise ValueError('The argument for the spatial unit unknown.')

        # ==== no need to do anything if the units are the same ====
        if self.spatial_unit == unit:
            return 
#            raise ValueError('The current unit is already the same as the input unit.')

        # ==== determine the factor ====
        fac = self.get_spatial_unit_conversion(unit)

        # multiply the value
        for ikey in ['x', 'y']:
            setattr(self, ikey, getattr(self, ikey) * fac)

        # change the value
        self.set_spatial_unit(unit)

    def rescale_dpc(self, dpc):
        """
        rescale the axes based on new dpc
        """
        old_dpc = self.dpc * 1
        self.dpc = dpc
        fac = old_dpc / dpc
        for ikey in ['x', 'y']:
            setattr(self, ikey, getattr(self, ikey) * fac)

class grid1d(base_grid):
    """
    a one dimensional grid

    Attributes
    ----------
    axis : 1d array
        a parameter that iterates through different values that characterizes the 1d profile. It usually corresponds to the length, but it can also be the azimuthal angle, or anything. 
        The image quantities will have to be in the same length as this
    """
    def __init__(self):
        base_grid.__init__(self)

    def set_axis(self, axis):
        self.axis = axis

class rectangular_grid(base_grid):
    """
    A two dimensional grid that is rectangular
    """
    def __init__(self):
        base_grid.__init__(self)

    def get_pixel_area(self):
        """
        the pixel area corresponds to a solid angle, thus we need to know a distance
        The physical area of the image cell isn't as useful
        Attributes
        ----------
        pixel_area : float
            the solid angle in steradians
        """
        # calculate the pixel sizes
        self.dx = abs(self.x[1] - self.x[0])
        self.dy = abs(self.y[1] - self.y[0])

        # determine the unit conversion
        # we want the pixel_area in arcsec^2
        fac = self.get_spatial_unit_conversion('arcsec')

        self.pixel_area = (dx*fac/3600) * (dy*fac/3600)

    def convert_to_mesh(self):
        """
        convert the xy_type from a grid to flat

        Caution that by doing this, the object forgets the original grid.
        If you still want the original grid information, it's better to save it somewhere else
        """
        if self.xy_type == 'flat':
            raise ValueError('The x, y coordinates are already flattend.')

        xx, yy = np.meshgrid(self.x, self.y, indexing='ij')

        self.set_xy(xx.flatten(), yy.flatten(), 'flat')

        # also do the same for ra, dec
        if hasattr(self, 'ra') & hasattr(self, 'dec'):
            dd, rr = np.meshgrid(self.dec, self.ra, indexing='ij')
            self.set_dec_ra(dd.flatten(), yy.flatten())

    def recenter(self, x0, y0):
        """ reset the center
        x0, y0 : float 
            the center in the current coordinate system 
        """
        self.x -= x0
        self.y -= y0


# ====
# image classes
# ====
class base_image(object):
    """
    the parent of all images. This will keep the intensity, tau surface, optical depth information, etc
    all image dimensions are in x by y by f

    tau3d :
        tau=? location in model coordinates
    """

    def __init__(self):
        self.quantity = [] # keep track of the physical quantity names

    def set_quantity(self, quantname, quant):
        """ sets a  3d quantity. the quantity can be anything
        as long as it's in nx by ny by nw
        Parameters
        ----------
        quant : 3d np.ndarray
            the array
        quantname : str
            name of the quantity
        """
        self.quantity.append(quantname)
        setattr(self, quantname, quant)


class base_intensity(base_image):
    """
    basic manager for the intensity
    """
    def __init__(self):
        base_image.__init__(self)

    def set_stokes_unit(self, unit):
        self.stokes_unit = unit

    def set_rms(self, rms, stokes):
        """
        set the noise level for a stokes parameter
        """
        key = '%s_rms'%stokes
        setattr(self, key, rms)

    def get_tb(self, stokes):
        """
        calculate the brightness temperature of a certain stokes parameter
        """
        # it's easier to fetch the intensity first
        inten = getattr(self, stokes)

        tb = np.zeros_like(inten)

        for i in range(self.grid.nw):
            if self.stokes_unit == 'cgs':
                tb[...,i] = cgs_to_tb(inten[...,i], self.grid.w[i])

            elif self.stokes_unit == 'jyppix':
                tb[...,i] = jyppix_to_tb(inten[...,i], self.grid.pixel_area, self.grid.w[i])

            elif self.stokes_unit == 'jypbeam':
                bmaj = self.psfarg['bmaj']
                bmin = self.psfarg['bmin']
                tb[...,i] = jypbeam_to_tb(inten[...,i], bmaj, bmin, self.grid.w[i])

            else:
                raise ValueError('stokes_unit unknown')

        key = '%s_tb'%stokes
        setattr(self, key, tb)

class intensity1d(base_intensity):
    def __init__(self): 
        base_intensity.__init__(self)

class intensity2d(base_intensity): 
    def __init__(self):
        base_intensity.__init__(self)

    def convolve_gaussian(self, bmaj, bmin, bpa):
        """ convolve the image with a gaussian beam 
        Parameters
        ----------
        bmaj : float
            beam major axis in arcsec. We will only consider it fixed as a function of wavelength
        bmin : float
        bpa : float
            in degrees

        Returns
        -------
        a copy of itself, but the stokes I (Q,U,V, if applicable) are replaced.
        The polarization properties and brightness temperature do not exist

        Additional Attributes
        ---------------------
        psf : 3d ndarray
        psfarg : dict
            bmaj : float 
                beam major axis in arcsec
            bmin : float
                beam minor axis in arcsec
            bpa : float
                beam position angle in degrees 
        """
        if isinstance(self.grid, rectangular_grid) is False:
            raise ValueError('currently only rectangular grid works')

        if self.grid.xy_type != 'grid':
            raise ValueError('currently, only xy_type="grid" will work')

        # check if dpc is set
        if hasattr(self.grid, 'dpc') is False:
            raise ValueError('dpc must be set')

        # ==== start calculation ====
        # create a new image
        out = intensity2d()

        # keep the grid, dpc
        out.grid = copy.deepcopy(self.grid)
        out.set_stokes_unit(self.stokes_unit)

        # pixel size in arcseconds
        fac = self.grid.get_spatial_unit_conversion('arcsec')
        dx = self.grid.dx * fac
        dy = self.grid.dy * fac

        # ==== calculate the psf ====
        psf = get_gaussian_psf(self.grid.nx, self.grid.ny, bmaj, bmin, bpa, pscale=[dx, dy])

        # ==== determine which stokes ====
        stokes4 = ['I', 'Q', 'U', 'V']
        stokes = []
        for istokes in stokes4:
            if hasattr(self, istokes):
                stokes.append(istokes)

        # iterate through each stokes and wavelength
        for istokes in stokes:
            inten = getattr(self, istokes)
            c3d = np.zeros_like(inten)
            for i in range(self.grid.nw):
                cc = convolve_image_psf(inten[:,:,i], psf)
            out.set_quantity(istokes, cc)

        psfarg = {'bmaj':bmaj, 'bmin':bmin, 'bpa':bpa}
        out.grid.set_psfarg(psfarg)

        return out

    def cut_slit(self, laxis, quant=['I'], width=1,
        trackkw={'x0':0, 'y0':0, 'theta':0}):
        """ instead of interpolating, we use a slit and average the pixels perpendicular to the cut. 
        The slit can only be linear

        Parameters
        ----------
        laxis : 1d ndarray
            the location in the slit direction in cm
        width : int
            The number of pixels. Default is 1
        """
        # check if it's Rectangular grid
        if isinstance(self.grid, rectangularGrid) is False:
            raise ValueError('grid must be a rectangular grid')

        # the angle in the coordinates of the image
        angle = trackkw['theta']

        # unit direction of the slit
        vec =  (np.cos(angle), np.sin(angle))

        # the direction perpendicular to the slit
        per = (np.cos(angle + np.pi/2), np.sin(angle + np.pi/2))

        # the pixel length in the direction perpendicular to the slit
        if (-1 <= np.tan(angle+np.pi/2)) & (np.tan(angle+np.pi/2) <= 1):
            dp = self.grid.dx / np.cos(angle+np.pi/2)
        else:
            dp = self.grid.dy / np.sin(angle+np.pi/2)
        dp = abs(dp)

        # the different centers
        if np.mod(width, 2) == 1:
            offset = np.arange(-(width//2), width//2 + 1)
        else:
            offset = np.arange(- (width//2), width//2) + 0.5
        xcen = trackkw['x0'] + offset * per[0] * dp
        ycen = trackkw['y0'] + offset * per[1] * dp

        # ==== begin interpolation ====
        profs = []
        for iquant in quant:
            # interpolate along different lines
            prof = np.zeros([len(laxis), self.grid.nw, width])
            for i in range(width):
                # the coordinates for this line
                xpnt = xcen[i] + laxis * vec[0]
                ypnt = ycen[i] + laxis * vec[1]

                # convert to pixel coordinates
                x = np.interp(xpnt, self.grid.x, np.arange(self.grid.nx))
                y = np.interp(ypnt, self.grid.y, np.arange(self.grid.ny))

                # iterate through wavelength
                for j in range(self.grid.nw):
                    prof[:,j,i] = map_coordinates(getattr(self, iquant)[...,j], [x,y], order=1)


            profs.append( np.mean(prof, axis=2) )

        # prepare the 1d image
        grid = grid1d()
        grid.set_axis(laxis)
        x = trackkw['x0'] + laxis * vec[0]
        y = trackkw['y0'] + laxis * vec[1]
        grid.set_xy(x, y, 'flat')
        grid.set_w(self.grid.w)
        grid.get_f()
        # additional attributes
        for ikey in ['dps', 'psfarg', 'restfreq']:
            if hasattr(self, ikey):
                setattr(grid, ikey, getattr(self, ikey))

        cut = intensity1d()
        cut.grid = grid
        cut.stokes_unit = self.stokes_unit
        for i, iquant in enumerate(quant):
            cut.set_quantity(iquant, profs[i])

def get_gaussian_psf(nx, ny, bmaj, bmin, bpa, pscale=None):
    """ calculate 2d gaussian psf
    nx, ny : int
        number of grid cells in x, y
    bmaj, bmin: float
        beam major and minor axis, in units corresponding to pscale
    bpa : float
        direction of the beam major axis East-of-North, ie., relative to the y-axis, following the left-hand rule. degrees
    """
    # determine pixel scale
    if pscale is None:
        dx, dy = 1, 1
    else:
        dx, dy = pscale

    # calculate x,y axis
    x = (np.arange(nx, dtype=np.float64) - nx / 2.) * dx
    y = (np.arange(ny, dtype=np.float64) - ny / 2.) * dy

    # Calculate the standard deviation of the Gaussians
    sigmax = bmaj / (2.0 * np.sqrt(2.0 * np.log(2.)))
    sigmay = bmin / (2.0 * np.sqrt(2.0 * np.log(2.)))
    norm = (2. * np.pi * sigmax * sigmay) / dx / dy

    sin_bpa = np.sin(bpa / 180. * np.pi - np.pi / 2.)
    cos_bpa = np.cos(bpa / 180. * np.pi - np.pi / 2.)

    # Define the psf
    xx, yy = np.meshgrid(x, y, indexing='ij')
    bxx = cos_bpa * xx - sin_bpa * yy
    byy = sin_bpa * xx + cos_bpa * yy
    psf = np.exp(-0.5 * (bxx / sigmax)**2 - 0.5 * (byy / sigmay)**2)

    psf /= norm

    return psf

def convolve_image_psf(image, psf):
    """ convolves two images
    """
    conv = signal.fftconvolve(image, psf, mode='same')
    return conv

# ====================
# image unit conversions
# ====================
def jyppix_to_cgs(jyppix, dxy):
    """
    convert jy/pixel image to inu in cgs
    Parameters
    ----------
    jyppix       : ndarray
        the image in jy/pixel units
    dxy     : float
        the solid angle size the pixel in arcsec**2
    """
    solid_angle = dxy / 3600.**2 * natconst.rad**2
    return jyppix * natconst.jy / solid_angle

def cgs_to_jyppix(cgsim, dxy):
    solid_angle = dxy / 3600.**2 * natconst.rad**2
    return cgsim * solid_angle / natconst.jy

def jypbeam_to_cgs(jypbeam, bmaj, bmin):
    """ convert image of jy/beam to cgs units
    """
    beam = calc_beam_solid_angle(bmaj, bmin)
    return jypbeam * natconst.jy / beam

def cgs_to_jypbeam(cgsim, bmaj, bmin):
    beam = calc_beam_solid_angle(bmaj, bmin)
    return cgsim * beam / natconst.jy

def cgs_to_tb(cgsim, wav_micron):
    """ convert image in cgs units to brightness temperature
    """
    freq = natconst.cc * 1e4 / wav_micron
    ld2 = (wav_micron*1e-4)**2
    hnu = natconst.hh * freq
    hnu3_c2 = natconst.hh * freq**3 / natconst.cc**2
    tb = hnu / natconst.kk / np.log(2. * hnu3_c2 / abs(cgsim) + 1.) * np.sign(cgsim)
    return tb

def tb_to_cgs(tb, wav_micron):
    """ convert image in brightness temperature to cgs units
    basically the planck function
    """
    freq = natconst.cc * 1e4 / wav_micron
    ld2 = (wav_micron * 1e-4)**2
    hnu = natconst.hh * freq
    hnu3_c2 = natconst.hh * freq**3 / natconst.cc**2
    cgs = 2 * hnu3_c2 / (np.exp(hnu / natconst.kk / tb) - 1.)
    return cgs

def jyppix_to_tb(jyppix, dxy, wav_micron):
    """
    converts jy/pixel image to brightness temperature 
    """
    cgsim = jyppix_to_cgs(jyppix, dxy)
    tb = cgs_to_tb(cgsim, wav_micron)
    return tb

def jypbeam_to_tb(jypbeam, bmaj, bmin, wav_micron):
    """
    converts jy/beam image to brightness temperature 
    """
    cgsim = jypbeam_to_cgs(jypbeam, bmaj, bmin)
    tb = cgs_to_tb(cgsim, wav_micron)
    return tb

def tb_to_jypbeam(tb, bmaj, bmin, wav_micron):
    """ convert brightness temperature to jy/beam
    """
    cgsim = tb_to_cgs(tb, wav_micron)
    jypbeam = cgs_to_jypbeam(cgsim, bmaj, bmin)
    return jypbeam


# ====================
# elementary calculations
# ====================
def calc_beam_solid_angle(bmaj, bmin):
    """ bmaj, bmin in arcsec
    """
    return (bmaj / 3600. * natconst.rad) * (bmin / 3600. * natconst.rad) * np.pi/4. / np.log(2.)

def calc_lpa(Q, U):
    """ function to calculate polarization angle
    Parameters 
    ----------
    Q : ndarray
    U : ndarray

    Returns
    -------
    ang : ndarray
        the polarization angle in radians and in between 0, 2pi
    """
    ang = np.arctan2(U, Q) / 2.
    reg = ang < 0.
    ang[reg] = ang[reg] + 2*np.pi
    reg = ang > np.pi
    ang[reg] = ang[reg] - np.pi
    return ang


