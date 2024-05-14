"""
module for handling grid related

For some data, it has it's own grid (e.g., wavelength for mcmono). These
attributes will be optional 
    w_rad : wavelength for the radiation field for urad and flux
"""
import numpy as np
import pdb
import os

from . import natconst

# ====================
# objects
# ====================
class baseGrid(object):
    """
    basic grid 

    Attributes
    ----------
    crd_sys : str
        type of coordinate system
        'car' : cartesian
        'sph' : spherical 
        'cyl' : cylindrical 
        'ppl' : plane parallel
    """
    def __init__(self):
        self.crd_sys = ''
        self.act_dim = [0, 0, 0]

    def set_crd_sys(self, crd_sys):
        if crd_sys not in ['ppl', 'car', 'sph', 'cyl']:
            raise ValueError('crd_sys unknown')

        self.crd_sys = crd_sys

    def get_crd_sys(self):
        """ get the crd_sys from crd_val
        """
        if self.crd_val == 10:
            self.crd_sys = 'ppl'
        elif (self.crd_val < 100):
            self.crd_sys = 'car'
        elif (self.crd_val >= 100) & (self.crd_val < 200):
            self.crd_sys = 'sph'
        elif (self.crd_val >= 200) & (self.crd_val < 300):
            self.crd_sys = 'cyl'
        else:
            raise ValueError('Unsupported coordinate system')

    def get_crd_val(self):
        """ get a crd_val from crd_sys
        """
        if self.crd_sys == 'ppl':
            self.crd_val = 10
        elif self.crd_sys == 'car':
            self.crd_val = 0
        elif self.crd_sys == 'sph':
            self.crd_val = 100
        elif self.crd_sys == 'cyl':
            self.crd_val = 200
        else:
            raise ValueError('Unsupported coordinate system')

    def setup_from_hdr(self, hdr):
        """ basic properties based off of header of the file
        """
        if hdr[0] != 1:
            raise ValueError('iformat unknown')

        self.iformat = hdr[0]
        self.grid_style = hdr[1]
        self.crd_val = hdr[2]
        self.act_dim = [hdr[4], hdr[5], hdr[6]]

        self.nx = hdr[7]
        self.ny = hdr[8]
        self.nz = hdr[9]

        self.get_crd_sys()

    def read_wavelength(self, usefor='temperature', fname=None, fdir=None):
        """ read wavelength grid. 
            'wavelength_micron.inp' = for temperature calculations
            'wavelength_micron_mcmono.inp' = from mcmono calculations 
        """
        if fname is None:
            if usefor == 'temperature':
                fname = 'wavelength_micron.inp'
            elif usefor == 'mcmono':
                fname = 'wavelength_micron_mcmono.inp'
            else:
                raise ValueError('data unknown')

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)

        if usefor == 'temperature':
            self.nw = int(data[0])
            self.w = data[1:]
            self.f = natconst.cc / self.w * 1e4
        elif usefor == 'mcmono':
            self.nw_rad = int(data[0])
            self.w_rad = data[1:]
            self.f_rad = natconst.cc / self.w_rad * 1e4

    def write_wavelength(self, usefor='temperature', fname=None, fdir=None):
        """ write wavelength grid. 
        """
        if fname is None:
            if usefor == 'temperature':
                fname = 'wavelength_micron.inp'
            elif usefor == 'mcmono':
                fname = 'wavelength_micron_mcmono.inp'
            else:
                raise ValueError('data unknown')

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        if usefor == 'temperature':
            nw = self.nw
            w = self.w
        elif usefor == 'mcmono':
            nw = self.nw_rad
            w = self.w_rad

        with open(fname, 'w') as f:
            f.write('%d\n'%(nw))
            for w in self.w:
                f.write('%.9e\n'%w)

    def set_wavelength(self, w, usefor = 'temperature'):
        if usefor == 'temperature':
            self.nw = len(w)
            self.w = w
            self.f = natconst.cc / self.w * 1e4
        elif usefor == 'mcmono':
            self.nw_rad = len(w)
            self.w_rad = w
            self.f_rad = natconst.cc / self.w_rad * 1e4
        else:
            raise ValueError('usefor unknown')

    def make_wavelength(self, wbound=[1, 1e4], nw=[20], usefor='temperature'):
        """ convenience function to make wavelength from scratch 
        """
        w, wi = make_axis(nw, wbound, 'geo')
        self.set_wavelength(wi, usefor=usefor)

class regularGrid(baseGrid):
    """
    the regular grid
    """
    def __init__(self):
        baseGrid.__init__(self)

    def read_spatial(self, fname='amr_grid.inp', fdir=None):
        """ read the amr_grid.inp
        """
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        hdr = np.fromfile(fname, count=10, sep=" ", dtype=int)
        if hdr[1] != 0:
            raise ValueError('this is not a regular grid')

        self.setup_from_hdr(hdr)

        with open(fname, 'r') as f:
            # read the header info
            hdr = np.fromfile(f, count=10, sep=" ", dtype=int)

            data = np.fromfile(f, count=-1, sep=" ", dtype=np.float64)

        # get cell interfaces
        self.xi = data[:self.nx+1]

        data = data[self.nx+1:]
        self.yi = data[:self.ny+1]

        data = data[self.ny+1:]
        self.zi = data[:self.nz+1]

        self.get_cell_center()

    def write_spatial(self, fname='amr_grid.inp', fdir=None):
        """Writes the wavelength grid to a file (e.g. amr_grid.inp).

        Parameters
        ----------

        fname : str, optional
                File name into which the spatial grid should be written. If omitted 'amr_grid.inp' will be used.

        """
        self.get_crd_val()

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'w' ) as wfile:

            # Format number
            wfile.write('%d\n' % 1)

            # grid_style
            wfile.write('%d\n' % self.grid_style)

            # crd_val
            wfile.write('%d\n' % self.crd_val)

            # Gridinfo
            wfile.write('%d\n' % 0)

            # Active dimensions
            wfile.write('%d %d %d \n' % (self.act_dim[0], self.act_dim[1], self.act_dim[2]))

            # Grid size (x,y,z or r,phi,theta, or r,phi,z)
            wfile.write('%d %d %d \n' % (self.nx, self.ny, self.nz))

            for x in self.xi:
                wfile.write('%.9e\n'%x)
            for y in self.yi:
                wfile.write('%.9e\n'%y)
            for z in self.zi:
                wfile.write('%.9e\n'%z)

    def set_xi(self, xi):
        if xi is None:
            self.xi = np.array([0, 0])
            self.nx = 1
        else:
            self.xi = xi
            self.nx = len(xi) - 1

    def set_yi(self, yi):
        if yi is None:
            self.yi = np.array([0, 0])
            self.ny = 1
        else:
            self.yi = yi
            self.ny = len(yi) - 1

    def set_zi(self, zi):
        if zi is None:
            self.zi = np.array([0, 0])
            self.nz = 1
        else:
            self.zi = zi
            self.nz = len(zi) - 1

    def make_spatial(self, crd_sys='car', act_dim=[1,1,1], 
        xbound=[-0.5*natconst.au, 0.5*natconst.au], nx=[20], 
        ybound=[-0.5*natconst.au, 0.5*natconst.au], ny=[20],
        zbound=[-0.5*natconst.au, 0.5*natconst.au], nz=[20],
        ):
        """ convenience function to create a grid from scratch
        """
        self.set_crd_sys(crd_sys)

        self.act_dim = act_dim

        self.grid_style = 0

        if self.crd_sys == 'ppl':
            raise ValueError('not done yet')

        elif self.crd_sys == 'car':
            for i, nc, bd, set_x in zip(
                [0,1,2], 
                [nx, ny, nz], 
                [xbound, ybound, zbound], 
                [self.set_xi, self.set_yi, self.set_zi] ):

                if act_dim[i] == 1:
                    x, xi = make_axis(nc, bd, 'lin')
                    set_x(xi)
                else:
                    set_x(None)

        elif self.crd_sys == 'sph':
            for i, nc, bd, set_x, spacing in zip(
                [0,1,2], 
                [nx, ny, nz],
                [xbound, ybound, zbound],
                [self.set_xi, self.set_yi, self.set_zi], 
                ['geo', 'lin', 'lin'] ):

                if act_dim[i] == 1:
                    x, xi = make_axis(nc, bd, spacing)
                    set_x(xi)
                else:
                    set_x(None)
        else:
            raise ValueError('crd_sys unknown')

        self.get_cell_center()

    def get_vol(self):
        """ calculate volume of the grid cells
        """
        if self.crd_sys == 'car':
            dx = self.xi[1:] - self.xi[:-1]
            dy = self.yi[1:] - self.yi[:-1]
            dz = self.zi[1:] - self.zi[:-1]
            self.vol = dx[:,None,None] * dy[None,:,None] * dz[None,None,:]

        elif self.crd_sys == 'sph':
            dr3 = self.xi[1:] ** 3 - self.xi[:-1] ** 3
            dcost = np.cos(self.yi[:-1]) - np.cos(self.yi[1:])
            dphi = self.zi[1:] - self.zi[:-1]
            self.vol = 1./3. * dr3[:,None, None] * dcost[None,:,None] * dphi[None, None,:]
        else:
            raise ValueError('Coordinate system ' + self.crd_sys + ' is not yet supported.')

    def get_cell_center(self):
        """ radmc3d uses the cell walls as inputs, but sometimes it's easier to use cell centers
        """
        # prepare some other properties 
        if (self.crd_sys == 'ppl') | (self.crd_sys == 'car'):
            self.x = (self.xi[:-1] + self.xi[1:]) * 0.5
            self.y = (self.yi[:-1] + self.yi[1:]) * 0.5
            self.z = (self.zi[:-1] + self.zi[1:]) * 0.5
        elif (self.crd_sys == 'sph') | (self.crd_sys == 'cyl'):
            self.x = np.sqrt(self.xi[:-1] * self.xi[1:])
            self.y = (self.yi[:-1] + self.yi[1:]) * 0.5
            self.z = (self.zi[:-1] + self.zi[1:]) * 0.5
        else:
            raise ValueError('internal error. crd_sys unknown')

class octreeGrid(baseGrid):
    """
    octree grid 
    """
    def __init__(self):
        baseGrid.__init__(self)

    def read_spatial(self, fname='amr_grid.inp', fdir=None):
        """ read the amr_grid.inp
        """
        raise ValueError('not done yet')

        hdr = np.fromfile(fname, count=10, sep=" ", dtype=int)
        if hdr[1] != 1:
            raise ValueError('this is not an octree grid')

        self.setup_from_hdr(hdr)

        with open(fname, 'r') as f:
            # basegrid header 
            hdr = np.fromfile(f, count=10, sep=" ", dtype=int)

            self.levelmax, self.nleafsmax, self.nbranchmax = np.fromfile(f, count=3, sep=" ", dtype=int)

            # read base grid
            count = (self.nx+1) * (self.ny + 1) * (self.nz + 1)
            base = np.fromfile(f, count=count, sep=" ", dtype=np.float64)

            data = np.fromfile(f, count=-1, sep=" ", dtype=int)
        
        # get cell interfaces
        self.xi = data[:self.nx+1]

        data = data[self.nx+1:]
        self.yi = data[:self.ny+1]

        data = data[self.ny+1:]
        self.zi = data[:self.nz+1]

    def putNode(self, crd=(), cellsize=(), level=None, parentID=-1, cellID=None):
        """
        Function to put the data of a single node into the tree. This funcion assumes that all the arrays
        have already been allocated for the tree so input cell indices must refer to already existing array elements.

        Parameters
        ----------
        crd      : tuple
                   Cell center coordinates of the node

        cellsize : tuple
                   Full size of the cell in each dimension

        level    : int
                   Level of the cell in the tree

        parentID : int
                   Tree index of the parent cell

        cellID   : int
                   Tree index of the cell to be added
        """

        #
        # Add the cell centre and cell half width to the arrays
        #
        self.x[cellID] = crd[0]
        self.y[cellID] = crd[1]
        self.z[cellID] = crd[2]

        self.dx[cellID] = cellsize[0] * 0.5
        self.dy[cellID] = cellsize[1] * 0.5
        self.dz[cellID] = cellsize[2] * 0.5

        self.isLeaf[cellID] = True
        self.level[cellID] = level
        self.parentID[cellID] = parentID
        self.childID.append(np.zeros(self.nChild, dtype=np.int))

class layeredGrid(baseGrid):
    """ layered adaptive mesh grid
    """

    def __init__(self):
        baseGrid.__init__(self)

    def read_spatial(self, fname='amr_grid.inp', fdir=None):
        """ read the amr_grid.inp
        """
        raise ValueError('not done yet')

        hdr = np.fromfile(fname, count=10, sep=" ", dtype=int)
        if hdr[1] != 10:
            raise ValueError('this is not a layered amr grid')

        self.setup_from_hdr(hdr)

        with open(fname, 'r') as f:
            # basegrid header 
            hdr = np.fromfile(f, count=10, sep=" ", dtype=int)


# ====================
# reading / writing 
# ====================
def read_spatial(fname='amr_grid.inp', fdir=None):
    """ read the file
    """
    hdr = np.fromfile(os.path.join(fdir, fname), count=2, sep=" ", dtype=int)
    if hdr[1] == 0:
        grid = regularGrid()
    elif hdr[1] == 1:
        grid = octreeGrid()
    elif hdr[1] == 10:
        grid = layeredGrid()
    else:
        raise ValueError('grid_style unknown')

    grid.read_spatial(fname=fname, fdir=fdir)

    return grid

# ====================
# unit conversions
# ====================

# ====================
# elementary calculations
# ====================
def make_axis(ncell, xbound, xtype):
    """
    calculate an axis that can be linear or geometric

    Parameters
    ----------
    ncell : list of int, or int
        for the number of cells per interval
    xbound : list
        at least two elements to define wall
    xtype : str
            'lin' = linear 
            'geo' = geometric
    """
    if not isinstance(ncell, list):
        ncell = [ncell]

    # check if the number of cells are consistent with xbound
    if len(ncell) != (len(xbound)-1):
        raise ValueError('number of cells should be one less than the boundaries')

    nwall = [i + 1 for i in ncell]
    totcell = np.sum(ncell)

    x = np.zeros([totcell], dtype=np.float64)   # cell
    xi = np.zeros([totcell + 1], dtype=np.float64)      # wall

    # mark the first one
    xi[0] = xbound[0]
    istart = 1
    for ii in range(len(nwall)):
        if xtype == 'lin':
            iwall = xbound[ii] + (xbound[ii+1] - xbound[ii]) * (
                np.arange(nwall[ii], dtype=np.float64) / float(ncell[ii]))
        elif xtype == 'geo':
            iwall = xbound[ii] * (xbound[ii+1] / xbound[ii])**(
                np.arange(nwall[ii], dtype=np.float64) / float(ncell[ii]))
        else:
            raise ValueError('xtype can only be lin or geo')
        xi[istart:istart+ncell[ii]] = iwall[1:]
        istart = istart + ncell[ii]

    # now calculate cells
    if xtype == 'lin':
        x = 0.5 * (xi[1:] + xi[:-1])
    else:
        x = np.sqrt(xi[1:] * xi[:-1])

    return x, xi

def extrapolate_cell_to_wall(cell, mode='lin'):
    """ calculate wall coordinate based on the cell coordinates
    mode : str
        lin = this extrapolates linearly
        geo = extrapolate geometrically
    """
    nwall = len(cell) + 1
    wall = np.zeros([nwall])
    if mode == 'lin':
        wall[1:-1] = 0.5 * (cell[:-1] + cell[1:])
        # extrapolate for the first wall
        wall[0] = wall[1] - (cell[1] - cell[0])
        # extrapolate for the last wall
        wall[-1] = wall[-2] + (cell[-1] - cell[-2])
    elif mode == 'geo':
        wall[1:-1] = np.sqrt(cell[:-1] * cell[1:])
        # extrapolate for the end points
        wall[0] = wall[1] * (cell[1] / cell[0])**(-1)
        wall[-1] = wall[-2] * (cell[-1] / cell[-2])
    else:
        raise ValueError('mode unknown')

    return wall

def rot_x(theta, y, z):
    """ rototate frame (fixing the object) around the x-axis by some angle theta
    """
    y1 = np.cos(theta) * y + np.sin(theta) * z
    z1 = - np.sin(theta) * y + np.cos(theta) * z
    return y1, z1

def rot_z(theta, x, y):
    x1 = np.cos(theta) * x + np.sin(theta) * y
    y1 = - np.sin(theta) * x + np.cos(theta) * y
    return x1, y1

# ====================
# commands
# ====================

