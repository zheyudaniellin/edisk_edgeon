import os
import numpy as np
import matplotlib.pyplot as plt

def write_radpar(radpar, outdir):
    """ write radmc3d.inp file """
    fname = os.path.join(outdir, 'radmc3d.inp')
    with open(fname, 'w') as wfile:
        for ikey in radpar.keys():
            wfile.write('%s = %s\n'%(ikey, radpar[ikey]))
        wfile.close()

def write_lines_inp(outdir):
    """ write the lines.inp file
    """
    fname = os.path.join(outdir, 'lines.inp')
    with open(fname, 'w') as f:
        f.write('2\n') # format
        f.write('1\n')
        f.write('co leiden 0 0 0')


