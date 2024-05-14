"""
module for handling physical data
The following are directly related to radmc3d
    dustrho : dust volume density
    dusttemp : dust temperature 
    gasndens : number density
    gastemp : gas temperature
    gasvel : gas velocity
    vturb : turbulence
    jrad : mean radiation field 
    flux : flux field 
    qvis : heating 
    aldir : alignment direction 

additional properties
    gasrho : gas volume density
    gas_sdens : gas surface density 
    urad : energy density 
"""
import os
import numpy as np
import pdb

from . import natconst
from . grid import regularGrid, octreeGrid, layeredGrid

class fieldData(object):
    """
    place to keep the scalar field or vector field data

    Attributes
    ----------
    grid : grid object
    """
    def __init__(self):
        pass

    def set_grid(self, grid):
        self.grid = grid

    def read_dustrho(self, fname=None, fdir=None, binary=False):
        """ read the dust density
        """
        # determine the file name
        if fname is None:
            fname = 'dust_density'
            if binary:
                fname += '.binp'
            else:
                fname += '.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        # read file 
        self.dustrho = read_scalarfield(self.grid, fname, diff_species=True, binary=binary)

    def read_dusttemp(self, fname=None, fdir=None, binary=False):
        """ read the dust temperature 
        """
        # determine the file name
        if fname is None:
            fname = 'dust_temperature'

            if binary:
                fname += '.bdat'
            else:
                fname += '.dat'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        # read file
        self.dusttemp = read_scalarfield(self.grid, fname, diff_species=True, binary=binary)

    def read_aldir(self, fdir=None, fname=None, binary=False):
        """ read the dust alignment direction 
        """
        # determine the file name
        if fname is None:
            fname = 'dust_temperature'

            if binary:
                fname += '.bdat'
            else:
                fname += '.dat'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        # data
        self.aldir = read_vectorfield(self.grid, fname, diff_species=True, binary=binary)

    def read_gasndens(self, ispec, fname=None, fdir=None, binary=False):
        """ read the gas number density 
        ispec : str
            the name of the gas species
        """
        # determine the file name
        if fname is None:
            fname = 'numberden_%s'%ispec
            if binary:
                fname += '.binp'
            else:
                fname += '.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        igas = read_scalarfield(self.grid, fname, binary=binary, diff_species=False)

        if hasattr(self, 'gasndens'):
            self.gasndens = np.vstack( (self.gasndens.T, igas.T) ).T
        else:
            self.gasndens = igas[...,None]

        if hasattr(self, 'gas_species'):
            self.gas_species.append(ispec)
        else:
            self.gas_species = [ispec]

    def read_qvis(self, fname=None, fdir=None, binary=False):
        """ read the heatsource.inp 
        """
        if fname is None:
            fname = 'heatsource'

            if binary:
                fname += '.binp'
            else:
                fname += '.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        self.qvis = read_scalarfield(self.grid, fname, binary=binary, diff_species=False)

    def read_jrad(self, fdir=None, fname=None, binary=False, rtype=None):
        """ read the mean intensity field
        Parameters
        ----------
        rtype : str
            the type of radiation field 
            None = just the total radiation field
            'star' = radiation field from star
        """
        if fname is None:
            if rtype is None:
                fname = 'mean_intensity'
            elif rtype == 'star':
                fname = 'mean_intensity_star'
            else:
                raise ValueError('rtype unknown')

            if binary:
                fname += '.bout'
            else:
                fname += '.out'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'r') as f:
            # read header 
            if binary:
                sep = ''
                # iformat, precision, nrcells, nfreq
                hdr = np.fromfile(f, count=4, dtype=np.int64, sep=sep)
                iformat = hdr[0]
                precis = hdr[1]
                nrcells = hdr[2]
                nfreq = hdr[3]
                if precis == 4:
                    dtype = np.float32
                else:
                    dtype = np.float64
            else:
                sep = ' '
                # iformat, nrcells, nfreq
                hdr = np.fromfile(f, count=3, dtype=np.int64, sep=sep)
                iformat = hdr[0]
                nrcells = hdr[1]
                nfreq = hdr[2]
                dtype = np.float64

            # read frequency
            freq = np.fromfile(f, count=nfreq, dtype=dtype, sep=sep)

            # read data
            data = np.fromfile(f, count=-1, dtype=dtype, sep=sep)

        # reformat
        if isinstance(self.grid, regularGrid):
            shape = [self.grid.nx, self.grid.ny, self.grid.nz, nfreq]
            data = np.reshape(data, shape, order='f')
        else:
            raise ValueError('grid unknown')

        # designate the radiation field
        self.grid.set_wavelength(natconst.cc*1e4 / freq, usefor='mcmono')

        if rtype is None:
            self.jrad = data
        elif rtype == 'star':
            self.jrad_star = daata
        else:
            raise ValueError('rtype unknown')

    def read_flux(self, fdir=None, fname=None, binary=False):
        """ read the flux field. basically the same as mean_intensity.out, but with directions
        """
        raise ValueError('not done yet')

        if fname == None:
            if binary:
                fname = 'flux_field.bout'
            else:
                fname = 'flux_field.out'
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        if isinstance(self.grid, radmc3dOctree):
            raise ValueError('octree for readfluxfield not implemented yet')

    def write_dustrho(self, fdir=None, fname=None, binary=False):
        """ write the dust density
        """
        # determine file name
        if fname is None:
            fname = 'dust_density'
            if binary:
                fname += '.binp'
            else:
                fname += '.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        write_scalarfield(self.dustrho, fname, binary=binary, diff_species=True)

    def write_aldir(self, fdir=None, fname=None, binary=False):
        """ Writes the dust alignment direction 
        """
        # determine default file name
        if fname is None:
            if binary:
                fname = 'grainalign_dir.binp'
            else:
                fname = 'grainalign_dir.inp'

        # add on fdir
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        write_vectorfield(self.aldir, fname, binary=binary, diff_species=True)

    def write_dusttemp(self, fdir=None, fname=None, binary=False):
        """Writes the dust temperature
        """
        # determine file name
        if fname is None:
            fname = 'dust_temperature'
            if binary:
                fname += '.bdat'
            else:
                fname += '.dat'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        write_scalarfield(self.dusttemp, fname, binary=binary, diff_species=True)

    def write_gasndens(self, ispec, fdir=None, fname=None, binary=False):
        """ write the number density of a certain gas species
        ispec : str
            the index of the species
        """
        # first check if we even have that species
        try:
            inx = self.gas_species.index(ispec)
        except:
            raise ValueError('writing gasndens failed. Check if the gas species exists: %s'%ispec)

        # determine file name
        if fname is None:
            fname = 'numberdens_%s'%ispec
            if binary:
                fname += '.binp'
            else:
                fname += '.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        # write that certain species
        write_scalarfield(self.gasndens[...,inx], fname, binary=binary, diff_species=False)

    def write_gasvel(self, fdir=None, fname=None, binary=False):
        """ Writes the dust alignment direction 
        """
        # determine default file name
        if fname is None:
            if binary:
                fname = 'gas_velocity.binp'
            else:
                fname = 'gas_velocity.inp'

        # add on fdir
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        write_vectorfield(self.gasvel, fname, binary=binary, diff_species=False)

    def get_dustmass(self, idust=-1):
        """ calculate the total dust mass
        """
        if hasattr(self.grid, 'vol') is False:
            self.grid.get_vol()

        if idust >= 0:
            dmass = np.sum(self.grid.vol * self.rhodust[...,idust])
        else:
            dmass = np.sum(self.grid.vol[...,None] * self.rhodust)
        return dmass

    def get_qvis_lum(self):
        """ calculate heating luminosity
        """
        if hasattr(self.grid, 'vol') is False:
            self.grid.get_vol()

        return np.sum(self.grid.vol * self.qvis)

    def get_urad(self):
        """ calculate radiation energy density
        """
        dnu = abs(np.diff(self.grid.f))
        ndim = len(self.jrad.shape)
        for i in range(ndim - 1):
            dnu = dnu[None,...]

        self.urad = np.sum(self.jrad * dnu)

    # ===============
    # plotting 
    # ===============
    def plot(self, ):
        """ plot a slice 
        """
        pass

# ====================
# basic functions
# ====================
def write_scalarfield(quant, fname, binary=False, diff_species=False):
    """
    basic function to write scalar field information. For quantities that don't depend on the different dust species, they are in dimensions of nx by ny by nz
    The dimension for different species is in the last dimension
    Parameters
    ----------
    quant : ndarray
        scalar field data to be written 
    fname : str
        name of the file
    binary : bool
        If True the file will be in binary format, if False the file format is formatted ASCII text
    diff_species : bool
        if True, then quant has various species
    """
    # determine header 
    if diff_species:
        hdr = np.array([1, np.size(quant[...,0]), quant.shape[-1]], dtype=int)
    else:
        hdr = np.array([1, np.size(quant)], dtype=int)

    if binary:
        hdr = np.insert(hdr, 1, 8)

    with open(fname, 'w') as f:
        if binary:
            hdr.tofile(f)
            quant.flatten(order='f').tofile(f)
        else:
            hdr.tofile(f, sep=" ", format='%d\n')
            quant.flatten(order='f').tofile(f, sep=" ", format="%.9e\n")

def read_scalarfield(grid, fname, diff_species=False, binary=False):
    """
    read the scalar data 
    Parameters
    ----------

    fname  : str
            Name of the file containing a scalar variable

    grid : grid object

    binary : bool
            If True the file is in binary format, if False the file format is formatted ASCII text

    dim : tuple of int
        the dimensions of the data. This is should be obtained from amr_grid.inp

    Returns
    -------

    Returns a numpy Ndarray with the scalar field
    """

    if diff_species:
        nhdr = 3
    else:
        nhdr = 2

    if binary:
        nhdr += 1

    with open(fname, 'r') as f:
        # read the header first
        # np.fromfile will treat the binary/ASCII based on the sep keyword
        if binary:
            sep = ''
        else:
            sep = ' '
        hdr = np.fromfile(f, count=nhdr, sep=sep, dtype=np.int64)

        # determine the data type for the rest of the file
        if binary:
            if hdr[1] == 8:
                dtype = np.float64
            elif hdr[1] == 4:
                dtype = np.float32
            else:
                raise ValueError('binary header data type unknown')
        else:
            dtype = np.float64

        # read the whole file
        quant = np.fromfile(f, count=-1, sep=sep, dtype=dtype)

    if isinstance(grid, regularGrid):
        if diff_species:
            shape = [grid.nx, grid.ny, grid.nz, hdr[-1]]
        else:
            shape = [grid.nx, grid.ny, grid.nz]
        quant = np.reshape(quant, shape, order='f')

    else:
        raise ValueError('not done yet')

    return quant

def write_vectorfield(quant, fname, binary=False, diff_species=False):
    """ write the vector field. 
    the first index is for the direction by default
    the last index is for different species if applicable 
    """
    # determine header 
    if diff_species:
        hdr = np.array([1, np.size(quant[0,:,:,:,0]), quant.shape[-1]], dtype=int)
    else:
        hdr = np.array([1, np.size(quant[0,:,:,:])], dtype=int)

    if binary:
        hdr = np.insert(hdr, 1, 8)

    # write 
    with open(fname, 'w') as f:
        if binary:
            hdr.tofile(f)
            quant.flatten(order='f').tofile(f)
        else:
            hdr.tofile(f, sep=" ", format='%d\n')
            quant = np.reshape(quant, [3, np.prod(quant.shape[1:])], order='f')
            np.savetxt(f, quant.T, fmt='%.9e')

def read_vectorfield(grid, fname, binary=False, diff_species=False):
    """ read vector field
    [direction, x, y, z, (ndust)]
    """
    if isinstance(grid, regularGrid) is False:
        raise ValueError('not tested yet')

    if diff_species:
        nhdr = 3
    else: 
        nhdr = 2

    if binary:
        nhdr += 1
        sep = ''
    else:
        sep = ' '

    with open(fname, 'r') as f:
        hdr = np.fromfile(f, count=nhdr, sep=sep, dtype=np.int64)

        if binary:
            if hdr[1] == 8:
                dtype = np.float64
            elif hdr[1] == 4:
                dtype = np.float32
            else:
                raise ValueError('binary header data type unknown: hdr[1]=%d'%hdr[1])
        else:
            dtype = np.float64
        
        quant = np.fromfile(f, count=-1, sep=sep, dtype=dtype)

    if isinstance(grid, regularGrid):
        if diff_species:
            shape = [3, grid.nx, grid.ny, grid.nz, hdr[-1]]
        else:
            shape = [3, grid.nx, grid.ny, grid.nz]

        quant = np.reshape(quant, shape, order='f')

        # no longer want the last index to denote the direction
#        quant = np.moveaxis(quant, 0, -1)
    else:
        raise ValueError('not done yet')

    return quant
