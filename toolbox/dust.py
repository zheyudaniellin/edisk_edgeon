"""
dust.py
read/write dust opacity files
and calculations 
"""
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os

class opacity(object):
    """ dust opacity files
    """
    def __init__(self):
        pass 

    def set_ext(self, ext):
        self.ext = ext

    def get_ksca_eff(self):
        """ calculate effective ksca
        """
        self.ksca_eff = (1. - self.g) * self.ksca

    def get_ksca_from_z11(self):
        self.ksca_from_z11 = get_ksca_from_z11(self.z_ang, self.z11)

    def replace_ksca_using_z11(self):
        """ replace ksca with ksca_from_z11
        calculates ksca_from_z11 if it doesn't have it already 
        """
        try:
            self.ksca = self.ksca_from_z11
        except AttributeError:
            self.get_ksca_from_z11()
            self.ksca = self.ksca_from_z11

    def get_g_from_z11(self):
        self.g_from_z11 = get_g_from_z11(self.z_ang, self.z11)

    def replace_g_using_z11(self):
        if hasattr(self, 'g_from_z11') is False:
            self.get_g_from_z11()
        self.g = self.g_from_z11

    def set_zmat(self, z_ang, z11, z12, z22, z33, z34, z44):
        """ convenience function to set attributes relevant to scattering matrix
        The elements are in wavelength by angle
        """
        self.z_ang = z_ang
        self.z11 = z11
        self.z12 = z12
        self.z22 = z22
        self.z33 = z33
        self.z34 = z34
        self.z44 = z44

    def scale_ksca(self, fac):
        """ scale the scattering opacity and related quantities by some factor
        """
        self.ksca *= fac

        try:
            self.z11 *= fac
            self.z12 *= fac
            self.z22 *= fac
            self.z33 *= fac
            self.z34 *= fac
            self.z44 *= fac

        except AttributeError:
            pass

    def set_alignkap(self, a_ang, korth, kpara):
        """ convenience function to set attributes relating to alignment properties
        """
        self.a_ang = a_ang
        self.korth = korth
        self.kpara = kpara

    def get_ptherm(self):
        """ get intrinsic polarization fraction
        """
        self.ptherm = (self.korth - self.kpara) / (self.korth + self.kpara)

    def get_dustopac_format(self):
        """ when writing out to dustopac.inp, we need to specify the input format
        """
        if (hasattr(self, 'korth') & hasattr(self, 'kpara')):
            iformat = 20
        elif hasattr(self, 'z11'):
            iformat = 10
        else:
            iformat = 1

        self.dustopac_format = iformat
        
    def read_kappa(self, fdir=None):
        """ read the dustkappa_*.inp file
        """
        fname = 'dustkappa_%s.inp'%self.ext

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'r') as f:
            iformat, nwav = np.fromfile(f, count=2, sep=" ", dtype=int)

            data = np.fromfile(f, count=-1, sep=" ", dtype=np.float64)

        if iformat == 1:
            data = np.reshape(data, [nwav, 2])
            self.w = data[:,0]
            self.kabs = data[:,1]
        elif iformat == 2:
            data = np.reshape(data, [nwav, 3])
            self.w = data[:,0]
            self.kabs = data[:,1]
            self.ksca = data[:,2]
        elif iformat == 3:
            data = np.reshape(data, [nwav, 4])
            self.w = data[:,0]
            self.kabs = data[:,1]
            self.ksca = data[:,2]
            self.g = data[:,3]
        else:
            raise ValueError('unknown iformat: %d'%iformat)

    def write_kappa(self, fdir=None):
        """ write the dustkappa_*.inp file
        """
        fname = 'dustkappa_%s.inp'%self.ext

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        # determine the iformat
        if hasattr(self, 'g'):
            iformat = 3
        elif hasattr(self, 'ksca'):
            iformat = 2
        elif hasattr(self, 'kabs'):
            iformat = 1
        else:
            raise ValueError('unable to determine format for dustkappa_*.inp file')

        with open(fname, 'w') as f:
            # iformat
            f.write('%d\n'%iformat)

            # number of wavelengths
            f.write('%d\n'%len(self.w))

            # write data
            for i in range(len(self.w)):
                if iformat == 1:
                    f.write('%13.6e %13.6e \n'%(self.w[i], self.kabs[i]) )
                elif iformat == 2:
                    f.write('%13.6e %13.6e %13.6e \n'%(self.w[i], self.kabs[i], self.ksca[i]))
                elif iformat == 3:
                    f.write('%13.6e %13.6e %13.6e %13.6e \n'%(self.w[i], self.kabs[i], self.ksca[i], self.g[i]))
                else:
                    raise ValueError('iformat unknown')

    def read_scat(self, fdir=None):
        """ read the dustkapscatmat_*.inp file
        """
        fname = 'dustkapscatmat_%s.inp'%self.ext

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'r') as f:
            iformat, nwav, nang = np.fromfile(f, count=3, sep=" ", dtype=int)

            # the usual opacity part
            data = np.fromfile(f, count=nwav*4, sep=" ", dtype=np.float64)
            data = np.reshape(data, [nwav, 4])
            self.w = data[:,0]
            self.kabs = data[:,1]
            self.ksca = data[:,2]
            self.g = data[:,3]

            # angular grid
            z_ang = np.fromfile(f, count=nang, sep=" ", dtype=np.float64)

            # matrix
            data = np.fromfile(f, count=-1, sep=" ", dtype=np.float64)

        data = np.reshape(data, [nwav, nang, 6])
        self.set_zmat(z_ang, 
                data[...,0], data[...,1], data[...,2], 
                data[...,3], data[...,4], data[...,5]
        )

    def write_scat(self, fdir=None):
        """ write the dustkapscatmat_*.inp file
        """
        fname = 'dustkapscatmat_%s.inp'%self.ext

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'w') as f:
            # iformat
            f.write('1\n')
            # number of wavelength
            f.write('%d\n'%len(self.w))
            # number of angular grid 
            f.write('%d\n'%len(self.z_ang))

            f.write('\n')

            for i in range(len(self.w)):
                f.write('%13.6e %13.6e %13.6e %13.6e \n'%(self.w[i], self.kabs[i], self.ksca[i], self.g[i])
                )

            f.write('\n')

            for i in range(len(self.z_ang)):
                f.write('%13.6e\n'%self.z_ang[i])

            f.write('\n')

            for i in range(len(self.w)):
                for j in range(len(self.z_ang)):
                    f.write( ( '%13.6e '*6 + '\n') %
                        (self.z11[i,j], self.z12[i,j], self.z22[i,j], 
                         self.z33[i,j], self.z34[i,j], self.z44[i,j])
                    )

            f.write('\n')

    def read_align(self, fdir=None):
        """ read the dustkapalignfact_*.inp' file
        """
        fname = 'dustkapalignfact_%s.inp'%self.ext

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'r') as f:
            iformat, nwav, nang = np.fromfile(f, count=3, sep=" ", dtype=int)
            wav = np.fromfile(f, count=nwav, sep=" ", dtype=np.float64)
            self.a_ang = np.fromfile(f, count=nang, sep=" ", dtype=np.float64)
            data = np.fromfile(f, count=-1, sep=" ", dtype=np.float64)

        data = np.reshape(data, [nwav, nang, 2])

        self.korth = data[...,0]
        self.kpara = data[...,1]

    def write_align(self, fdir=None):
        """ write the dustkapalignfact_*.inp' file
        """
        fname = 'dustkapalignfact_%s.inp'%self.ext

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'w') as f:
            hdr = np.array([1, len(self.w), len(self.a_ang)] )
            hdr.tofile(f, sep='\n', format='%d')
            f.write('\n')

            self.w.tofile(f, sep='\n', format='%13.6e')

            f.write('\n')

            self.a_ang.tofile(f, sep='\n', format='%13.6e')

            f.write('\n')

            for i in range(len(self.w)):
                for j in range(len(self.a_ang)):
                    f.write( '%13.6e %13.6e\n' % (self.korth[i,j], self.kpara[i,j]) ) 
                f.write('\n')

    # ==== plotting ====
    def plot_w(self, quant, theta=0, w_unit='micron', ax=None, **kwargs
        ):
        """ plot quantity as a function of wavelength 
        Parameters
        ----------
        quant : str
            the name of the attribute
        theta : float
            the angle to be plotted in degrees
        """
        # check if quant is allowed and if it is angle independent 
        if quant in ['kabs', 'ksca', 'g', 
            'ksca_eff', 'ksca_from_z11', 'g_from_z11']:
            angle_independent = True
        elif quant in ['z11', 'z12', 'z22', 'z33', 'z34', 'z44']:
            angle_independent = False
            theta_grid = self.z_ang
        elif quant in ['korth', 'kpara']:
            angle_independent = False
            theta_grid = self.a_ang
        else:
            raise ValueError('quant is unknown: %s'%quant)

        # create ax
        if ax is None: 
            ax = plt.gca()

        # wavelength unit
        if w_unit == 'micron':
            wfac = 1.
        elif w_unit == 'mm':
            wfac = 1e3
        elif w_unit == 'cm':
            wfac = 1e4
        else:
            raise ValueError('w_unit unknonwn: %s'%w_unit)

        # get the value
        val = getattr(self, quant)
        if angle_independent is False:
            itheta = np.argmin(abs(theta_grid - theta))
            val = val[:,itheta]

        ax.plot(self.w / wfac, val, **kwargs)
        if angle_independent is False:
            txt = r'$\theta=%.2f$'%(theta_grid[itheta])
            ax.text(0.9, 0.9, txt, ha='right', va='center', transform=ax.transAxes)

    def plot_a(self, quant, wav=870, a_unit='deg', ax=None, **kwargs
        ):
        """ plot angular depend quantities
        Parameters
        ----------
        quant : str
            quantity to plot
        """
        if quant in ['z11', 'z12', 'z22', 'z33', 'z34', 'z44']:
            theta_grid = self.z_ang
        elif quant in ['korth', 'kpara']:
            theta_grid = self.a_ang
        else:
            raise ValueError('quant is unavailable: %s'%quant)

        # create ax
        if ax is None:
            ax = plt.gca()

        # angle unit
        if a_unit == 'deg':
            fac = 1. 
        elif a_unit == 'rad':
            fac = 180. / np.pi 
        else:
            raise ValueError('a_unit unknown: %s'%a_unit)

        iwav = np.argmin(abs(self.w - wav))

        val = getattr(self, quant)[iwav, :]

        ax.plot(theta_grid / fac, val, **kwargs)
        
class opacityManager(object):
    """ manager level for opacity objects
    """
    def __init__(self):
        self.opac = []
        self.nopac = 0

    def add_opac(self, opac):
        """ add an opacity object
        """
        if isinstance(opac, opacity) is False:
            raise ValueError('the input must be an opacity object')

        self.opac.append(opac)
        self.nopac += 1

    def get_ext_from_opac(self):
        """ get all the extension names when various opac are set
        """
        self.ext = []
        for i in range(self.nopac):
            self.ext.append(self.opac[i].ext)

    def get_dustopac_format_from_opac(self):
        self.dustopac_format = np.zeros([len(self.opac)])

        for i in range(self.nopac):
            if hasattr(self.opac[i], 'dustopac_format') is False:
                self.opac[i].get_dustopac_format()
            self.dustopac_format[i] = self.opac[i].dustopac_format
        
    def write_dustopac(self, fname='dustopac.inp', fdir=None):
        """ write the dustopac.inp file
        """
        if self.nopac < 1:
            raise ValueError('no dust opacity available')

        self.get_dustopac_format_from_opac()

        self.get_ext_from_opac()

        if fdir != None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'w') as f:
            # file format
            f.write('%-15d %s\n'%(2, 'Format number of this file'))
            # number of species
            f.write('%-15d %s\n'%(self.nopac, 'Nr of dust species'))

            # separator
            f.write('%s\n' % '============================================================================')

            # iterate through each dust species
            for i in range(self.nopac):
                f.write('%-15d %s\n'%(self.dustopac_format[i], 'Way in which this dust species is read'))

                f.write('%-15d %s\n'%(0, '0=Thermal grain, 1=Quantum heated'))
                f.write('%-15s %s %s\n'%(self.ext[i], '    ', 'Extension of name of dust***.inp file'))
                # Separator
                f.write('%s\n' % '----------------------------------------------------------------------------')

    def read_dustopac(self, fname='dustopac.inp', fdir=None):
        if fdir != None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'r') as f:
            # file format
            dum = f.readline().split()
            if int(dum[0]) != 2:
                raise ValueError('format unknown')

            # number of dust species
            nopac = int(f.readline().split()[0])

            # separator
            dum = f.readline()

            # iterate through each dust species
            dustopac_format = []
            therm = []
            ext = []
            for i in range(nopac):
                dum = f.readline().split()
                dustopac_format.append( int(dum[0]) )

                dum = f.readline().split()
                therm.append( int(dum[0]) )

                dum = f.readline().split()
                ext.append( dum[0] )

                # separator
                dum = f.readline()

        self.dustopac_format = dustopac_format
        self.ext = ext

    def read_opac(self, fdir=None):
        """ read all the opacities based on dustopac.inp
        the dustopac.inp must be read before this
        """
        if hasattr(self, 'ext') is False:
            raise ValueError('ext not available')

        for i in range(len(self.ext)):
            opac = opacity()
            opac.set_ext(self.ext[i])

            # read file based on format
            if self.dustopac_format[i] <= 3:
                opac.read_kappa(fdir=fdir)
            elif self.dustopac_format[i] == 10:
                opac.read_scat(fdir=fdir)
            elif self.dustopac_format[i] == 20:
                opac.read_scat(fdir=fdir)
                opac.read_align(fdir=fdir)
            else:
                raise ValueError('dustopac_format unknown')

            self.add_opac(opac)

    def write_opac(self, fdir=None):
        """ write all the opacities 
        """
        for i in range(self.nopac):
            if self.dustopac_format[i] <= 3:
                self.opac[i].write_kappa(fdir=fdir)
            elif self.dustopac_format[i] == 10:
                self.opac[i].write_scat(fdir=fdir)
            elif self.dustopac_format[i] == 20:
                self.opac[i].write_scat(fdir=fdir)
                self.opac[i].write_align(fdir=fdir)
            else:
                raise ValueError('dustopac_format unknown')

# ==== simple functions ====
def get_ksca_from_z11(theta, z11):
    """ calculates ksca from z11
    theta : 1d ndarray
            in degrees, increase monotomically
    z11 : ndarray
           the last dimension is in theta 
    """
    mu = np.cos(theta * np.pi / 180.)
    dmu = np.abs(mu[1:] - mu[:-1])
    stg = 0.5 * ( z11[...,1:] + z11[...,:-1] )

    dim = z11.shape
    for i in range(len(dim) - 1):
        dmu = dmu[None,:]

    ksca_from_z11 = 2. * np.pi * np.sum(stg * dmu, axis=-1)

    return ksca_from_z11

def get_g_from_z11(theta, z11):
    """
    calculates phaseg from z11
    same arguments as getKscafromZ11
    look up polarization_total_scattering_opacity subroutine in polarization module in radmc3d
    """
    mu = np.cos(theta * np.pi / 180.)
    mus = 0.5 * (mu[1:] + mu[:-1])
    dmu = np.abs(mu[1:] - mu[:-1])
    stg = 0.5 * ( z11[...,1:] + z11[...,:-1] )

    dim = z11.shape
    for i in range(len(dim) - 1):
        dmu = dmu[None,:]
        mus = mus[None,:]

    ksca_from_z11 = 2. * np.pi * np.sum(stg * dmu, axis=-1)
    g_from_z11 = 2. * np.pi * np.sum(stg * mus * dmu, axis=-1) / ksca_from_z11

    return g_from_z11
