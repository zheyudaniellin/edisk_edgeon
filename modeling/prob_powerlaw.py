# prob_powerlaw.py
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import json
import sys
sys.path.append('mymodels')
import powerlaw as model
sys.path.append('..')
import toolbox as tlx
rad = np.pi / 180.
au = tlx.natconst.au

class pipeline(object):
    """ run one calculation
    """
    def __init__(self):
        pass
    def get_default_par(self):
        self.dustpar = {
            # grid parameters
            'nx': [64], 
            'xbound': [5*au, 300*au], 
            'ny': [64, 64], 
            'ybound': [85*rad, 90*rad, 95*rad],
            'nz': [32],
            # imaging wavelength
            'imwav': [1300],
            'inc': 45,
            'nphot_scat': 1e3,
            'imsizeau': 1000, 
            'npix': 200,
            'widthkms': 10, # for lines
            'linenlam': 11, # for lines
            # dust structure
            'R0': 500, # edge-of the disk 
            'Rc': 100, # characteristic radius
            'tau0': [0.1],
            'alb': [0],
            'p': -3, 
            'Hc' : 0.1, 
            'hs': 1.25,
            'T0': 30, 
            'q': 0.5, 
            'mstar': 1.5, 
            }
        self.setthreads = 1
        self.do_optical_depth = True

    def modify_dustpar(self, par):
        """ par = dictionary
        """
        for ikey in par.keys():
            if ikey not in self.dustpar.keys():
                raise ValueError('key not found in dustpar: %s'%(ikey))
            self.dustpar[ikey] = par[ikey]

    def prep_dir(self, datdir):
        """ clean and create output directory
        if we want to write in the current directory, let datdir = ''
        """
        self.datdir = datdir
        if self.datdir == '':
            pass
        else:
            if os.path.isdir(self.datdir):
                os.system('rm -rf %s'%self.datdir)

            os.system('mkdir %s '%self.datdir)

    def write_par(self):
        """ write parameter file
        """
        fname = os.path.join(self.datdir, 'dustpar.json')
        with open(fname, 'w') as f:
            json.dump(self.dustpar, f, indent=4, sort_keys=True)

    def run_input_model(self, ):
        """ calculate input model
        """
        mod = model.model(self.datdir, self.dustpar)
        mod.get_rho0()
        mod.get_grid(
            nx=self.dustpar['nx'],
            xbound=self.dustpar['xbound'], 
            ny=self.dustpar['ny'],
            nz=self.dustpar['nz'],
            ybound=self.dustpar['ybound']
        )

        mod.get_opac()

        mod.get_dusttemp()
        mod.get_dustrho()
        mod.get_stars()

        self.mod = mod

    def run_gas_input(self):
        """ if we want to do some line imaging, then use this after run_input_model()
        """
        self.mod.get_gasrho()
        self.mod.get_gasndens()
        self.mod.get_gasvel()

    def run_image(self):
        """ calculate the continuum image
        """
        tlx.image.write_camera_wavelength(self.dustpar['imwav'], fdir=self.datdir)

        cwd = os.getcwd()

        if self.datdir != '':
            os.chdir(self.datdir)

        radpar = {
                'nphot_scat': '%d'%(self.dustpar['nphot_scat']),
                'istar_sphere': '0',
                'setthreads': '%d'%self.setthreads,
                'mc_scat_maxtauabs': '30',
                'scattering_mode_max' : '1',
                'alignment_mode' : 0, 
                'incl_lines': 0, # ignore the lines in continuum imaging
                }
        tlx.utils.write_radpar(radpar, self.datdir)

        arg = {'npix':self.dustpar['npix'], 
            'sizeau':self.dustpar['imsizeau'], 
            'stokes':False, 'circ':False,
            'secondorder':True, 'nphot_scat':self.dustpar['nphot_scat'],
            'incl':self.dustpar['inc']} 

        com = tlx.image.commandImage()
        for ikey in arg.keys():
            setattr(com, ikey, arg[ikey])
        com.set_wavelength(loadlambda=True)

        com.get_command()
        com.make(fname='myimage.out')

        if self.do_optical_depth:
            radpar = {
                    'nphot_scat': '1000',
                    'istar_sphere': '0',
                    'setthreads': '%d'%self.setthreads,
                    'mc_scat_maxtauabs': '10',
                    'scattering_mode_max' : '1',
                    'alignment_mode' : 0
                }
            tlx.utils.write_radpar(radpar, self.datdir)
            args = {'npix':self.dustpar['npix'], 
                    'sizeau':self.dustpar['imsizeau'], 
                    'stokes':False, 
                    'circ':False, 
                    'secondorder':True, 
                    'nphot_scat':1e3,
                    'incl':self.dustpar['inc']}
            com = tlx.image.commandImage()
            com.set_mode('tracetau')
            for ikey in arg.keys():
                setattr(com, ikey, arg[ikey])
            com.set_wavelength(loadlambda=True)

            com.get_command()

            com.make(fname='myopt.out')

        os.chdir(cwd)

    def run_cube(self):
        """ calculate the image cube
        """
        cwd = os.getcwd()

        if self.datdir != '':
            os.chdir(self.datdir)

        # write the radmc3d.inp file
        radpar = {
                'nphot_scat': '%d'%(self.dustpar['nphot_scat']),
                'istar_sphere': '0',
                'setthreads': '%d'%self.setthreads,
                'mc_scat_maxtauabs': '30',
                'scattering_mode_max' : '1',
                'tgas_eq_tdust': 1, 
                'incl_dust': 1, # include the dust for cubes
                }
        tlx.utils.write_radpar(radpar, self.datdir)

        # write the lines.inp file
        tlx.utils.write_lines_inp(self.datdir)

        # move the co leiden data
        leiden_file = '/home/zdl3gk/mainProjects/edisk_edgeon/modeling/datafiles/molecule_co.inp'
        os.system('cp %s %s'%(leiden_file, self.datdir))

        # now calculate image cube
        arg = {'npix':self.dustpar['npix'],
            'sizeau':self.dustpar['imsizeau'],
            'stokes':False, 'circ':False,
            'secondorder':False, 
            'nphot_scat':self.dustpar['nphot_scat'],
            'incl':self.dustpar['inc']}

        com = tlx.image.commandImage()
        for ikey in arg.keys():
            setattr(com, ikey, arg[ikey])
        com.set_wavelength(iline=2, widthkms=self.dustpar['widthkms'], vkms=0., linenlam=self.dustpar['linenlam'])

        com.get_command()
        com.make(fname='mycube.out')

        os.chdir(cwd)

def write_script(par, fname):
    """ write the script in some directory
    par : dict
        the value for each element should be a string, as it would appear in the file
    """

    script_lines = [
        'import os', 
        'import numpy as np',
        'import sys',
        'sys.path.append("/home/zdl3gk/mainProjects/edisk_edgeon/modeling")',
        'sys.path.append("/home/zdl3gk/mainProjects/edisk_edgeon/modeling/mymodels")', 
        'from prob_powerlaw import pipeline',

        '# ==== settings ====',
        'datdir = ""',
        'par = {'
        ]
    # iterate through the necessary parameters
    # get the keys
    dumpipe = pipeline()
    dumpipe.get_default_par()

    str_keys = [] # hardcode the keys that are strings
    for ikey in dumpipe.dustpar.keys():
        if ikey in str_keys:
            txt = '"%s"'%ikey + ':' + '"%s"'%par[ikey] + ', '
            script_lines.append(txt)
        else:
            txt = '"%s"'%ikey + ':' + '%s'%str(par[ikey]) + ', '
            script_lines.append(txt)

    # finish the rest
    script_lines.extend( [
        '}',
        'pipe = pipeline()',
        'pipe.get_default_par()',
        'pipe.modify_dustpar(par)',
        'pipe.setthreads = 1',
        'pipe.imsizeau = %s'%par['imsizeau'],
        'pipe.npix = %s'%par['npix'],
        'pipe.circ_image = False',
        'pipe.do_optical_depth = True',
        '# ==== execution ====',
        'pipe.prep_dir(datdir)',
        'pipe.write_par()',
        'pipe.run_input_model()',
        'pipe.run_image()', 
        # delete some files to save space
        'os.system("rm -rf camera_wavelength_micron.inp")', 
        'os.system("rm -rf radmc3d.out")',
        'os.system("rm -rf wavelength_micron.inp")',
        'os.system("rm -rf stars.inp")',
        ])


    with open(fname, 'w' ) as f:
        for iline in script_lines:
            f.write(iline + '\n')

def write_slurm(fname, hours=24):
    """ write the slurm script
    """
    script_lines = [
        '#!/bin/bash',
        ' ',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks-per-node=1',
        '#SBATCH --time=%d:00:00'%hours,
        '#SBATCH --output=job_log.out',
        '#SBATCH --partition=standard',
        '#SBATCH --account=StarFormationTheory',
        ' ',
        'python pipe_script.py'
    ]
    with open(fname, 'w') as f:
         for iline in script_lines:
             f.write(iline + '\n')

def main():
    """ just run a single image
    """
    # ==== settings ====
    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/modeling/powerlaw1'

    pipe = pipeline()
    pipe.get_default_par()

    delta_ylim = 45.
    par = {
        # grid parameters
        'xbound': [1*au, 500*au], 
        'ybound': [(90-delta_ylim)*rad, 90*rad, (90+delta_ylim)*rad],
        'ny': [64, 64],
        # imaging wavelength
        'imwav': [1300],
        'inc': 83,  
        'nphot_scat': 1e3, 
        'imsizeau': 1200, 
        'npix': 300, 
        'widthkms': 10, 
        'linenlam': 51, 
        # dust structure
        'tau0': [0.35], 
        'alb': [0], 
        'R0':310,
        'p': 3, 
        'Hc' : 6,
        'T0':7.5,
        'q': 0.5,
        }

    pipe.modify_dustpar(par)
    pipe.setthreads = 1
    pipe.do_optical_depth = True
    pipe.circ_image = False

    # ==== execution ====
    pipe.prep_dir(datdir)
    pipe.write_par()
    pipe.run_input_model()

    pipe.run_image()

    # run image cube
#    pipe.run_gas_input()
#    pipe.run_cube()

if __name__ == '__main__':
    main()

