# kits.py
# some useful settings and tools
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import pdb
from . import image, natconst
au = natconst.au

def fetch_basic_data(source, baseline='SBLB', 
    robust=0.5, taper=None, imcor='image'):
    """
    """
    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/data'
    if source == 'IRAS04302':
        # determine file name
        fname = 'IRAS04302' + '_%s'%baseline + '_continuum' + '_robust_%.1f'%robust + '.%s'%imcor + '.tt0.fits'
        fname = os.path.join(datdir, fname)

        # determine center
        hms, dms = '04h33m16.498650s', '+22d53m20.230640s'
        frame = 'icrs'

        # noise level depends on the baseline and robust
        if baseline == 'SB':
            if robust == 0:
                rms = 0
            else:
                raise ValueError('robust unknown')

        elif baseline == 'SBLB':
            if robust == 0.5:
                rms = 1.45e-5
            else:
                raise ValueError('robust unknown')
        else:
            raise ValueError('baseline unknown')

    elif source == 'L1527':
        # determine file name
        fname = 'L1527IRS' + '_%s'%baseline + '_continuum' + '_robust_%.1f'%robust + '.%s'%imcor + '.tt0.fits'
        fname = os.path.join(datdir, fname)

        # determine center
        hms, dms = '04h39m53.878s', '+26d03m09.4s'
        frame = 'icrs'

        # noise level depends on the baseline and robust
        if baseline == 'SB':
            if robust == 0:
                rms = 0
            else:
                raise ValueError('robust unknown')

        elif baseline == 'SBLB':
            if robust == -0.5:
                rms = 29e-6
            else:
                raise ValueError('robust unknown')
        else:
            raise ValueError('baseline unknown')
    elif source == 'CB68':
        # file name
        fname = 'CB68' + '_%s'%baseline + '_continuum' + '_robust_%.1f'%robust + '.%s'%imcor + '.tt0.fits'
        fname = os.path.join(datdir, fname)

        # center
        hms, dms = '16h57m19.6428s', '-16d09m24.016s'
        frame = 'icrs'

        # noise level depends on the baseline and robust
        if baseline == 'SB':
            if robust == 0:
                rms = 0
            else:
                raise ValueError('robust unknown')

        elif baseline == 'SBLB':
            if robust == 0:
                rms = 21e-6
            else:
                raise ValueError('robust unknown')
        else:
            raise ValueError('baseline unknown')

    elif source == 'GSS30IRS3':
        # file name
        fname = 'GSS30IRS3' + '_%s'%baseline + '_continuum' + '_robust_%.1f'%robust + '.%s'%imcor + '.tt0.fits'
        fname = os.path.join(datdir, fname)

        # center
        hms, dms = '16h26m21.7155s', '-24d22m51.093s'
        frame = 'icrs'

        # noise level depends on the baseline and robust
        if baseline == 'SB':
            if robust == 0:
                rms = 0
            else:
                raise ValueError('robust unknown')

        elif baseline == 'SBLB':
            if robust == 0:
                rms = 18.5e-6
            else:
                raise ValueError('robust unknown')
        else:
            raise ValueError('baseline unknown')

    elif source in ['IRS7B-A', 'IRS7B-B']:
        # file name
        fname = 'IRS7B' + '_%s'%baseline + '_continuum' + '_robust_%.1f'%robust + '.%s'%imcor + '.tt0.fits'
        fname = os.path.join(datdir, fname)

        # center
        if source == 'IRS7B-A':
            hms, dms = '19h01m56.420s', '-36d57m28.66s'
        else:
            hms, dms = '19h01m56.392s', '-36d57m28.11s'
        frame = 'icrs'

        # noise level depends on the baseline and robust
        if baseline == 'SB':
            if robust == 0:
                rms = 0
            else:
                raise ValueError('robust unknown')

        elif baseline == 'SBLB':
            if robust == 0:
                rms = 3.096913e-5
            else:
                raise ValueError('robust unknown')
        else:
            raise ValueError('baseline unknown')
    else:
        raise ValueError('source unknown')

    dpc = fetch_distance(source)

    return {'fname':fname, 'rms':rms, 'hms':hms, 'dms':dms, 'frame':frame,
        'dpc':dpc}

def fetch_distance(source):
    if source == 'IRAS04302':
        dpc = 160
    elif source == 'L1527':
        dpc = 140
    elif source == 'CB68':
        dpc = 151
    elif source == 'GSS30IRS3':
        dpc = 137
    elif source in ['IRS7B-A', 'IRS7B-B']:
        dpc = 152
    else:
        raise ValueError('source unknown')

    return dpc

def fetch_default_robust(source, baseline='SBLB'):
    """
    get the default robust value and taper
    """
    if source == 'IRAS04302':
        if baseline == 'SB':
            robust = 0
            taper = None
        elif baseline == 'SBLB':
            robust = 0.5
            taper = None
        else:
            raise ValueError('baseline unknown')
    elif source == 'L1527':
        if baseline == 'SB':
            robust = 0
            taper = None
        elif baseline == 'SBLB':
            robust = -0.5
            taper = None
        else:
            raise ValueError('baseline unknown')
    elif source == 'CB68':
        if baseline == 'SB':
            robust = 0
            taper = None
        elif baseline == 'SBLB':
            robust = 0
            taper = None
        else:
            raise ValueError('baseline unknown')
    elif source == 'GSS30IRS3':
        if baseline == 'SB':
            robust = 0
            taper = None
        elif baseline == 'SBLB':
            robust = 0
            taper = None
        else:
            raise ValueError('baseline unknown')
    elif source in ['IRS7B-A', 'IRS7B-B']:
        if baseline == 'SB':
            robust = 0
            taper = None
        elif baseline == 'SBLB':
            robust = 0
            taper = None
        else:
            raise ValueError('baseline unknown')
    else:
        raise ValueError('source unknown')

    return robust, taper

def fetch_default_trim(source):
    """
    get some default image sizes
    """
    if source == 'IRAS04302':
        xlim = [-350*au, 350*au]
        ylim = [-75*au, 75*au]
    elif source == 'L1527':
        xlim = [-120*au, 120*au]
        ylim = [-70*au, 70*au]
    elif source == 'CB68':
        xlim = [-50*au, 50*au]
        ylim = [-50*au, 50*au]
    elif source == 'GSS30IRS3':
        xlim = [-150*au, 150*au]
        ylim = [-150*au, 150*au]
    elif source == 'IRS7B-A':
        xlim = [-60*au, 60*au]
        ylim = [-100*au, 100*au]
    elif source == 'IRS7B-B':
        xlim = [-30*au, 30*au]
        ylim = [-40*au, 40*au]
    else:
        raise ValueError('source unknown')

    return xlim, ylim

def easy_read(source, baseline='SBLB', imcor='image', 
    apply_center=True, apply_trim=True, ):
    """
    read the image
    use default robust parameters
    automatically set the noise level and centering
    """
    # get the default
    robust, taper = fetch_default_robust(source, baseline=baseline)

    # get the basic data
    src = fetch_basic_data(source=source, baseline=baseline, 
        robust=robust, taper=taper, imcor=imcor)

    # read the image
    im = image.intensity()
    im.read_fits(src['fname'], src['dpc'])
    im.grid.w = np.array([1300])
    im.grid.get_frequency()
    im.set_rms(np.array([src['rms']]))

    # recenter the image
    if apply_center:
        apply_center_from_hms_dms(im, src['hms'], src['dms'], src['frame'])

    # trim the image
    if apply_trim:
        xlim, ylim = fetch_default_trim(source)
        im.trim({'xlim':xlim, 'ylim':ylim})

    return im

def apply_center_from_hms_dms(im, hms, dms, frame):
    """
    actually apply the center
    """
    csky = SkyCoord(hms, dms, frame=frame)

    xc = np.interp(csky.dec.degree, im.grid.dec, im.grid.x)
    yc = np.interp(csky.ra.degree, im.grid.ra, im.grid.y)

    im.grid.recenter(xc, yc)


