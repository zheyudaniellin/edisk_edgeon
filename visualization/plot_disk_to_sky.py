# plot_disk_to_sky.py
# plot the disk with its coordinates
# and show where the sky plane is
import numpy as np
import matplotlib.pyplot as plt
import tool_box
import pdb
rad = np.pi / 180.
lincol = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']

def add_text_sph(ax, r, theta, phi, text, **kwargs):
    """
    add the text using spherical coordinates
    """
    x = r * np.sin(theta*rad) * np.cos(phi*rad)
    y = r * np.sin(theta*rad) * np.sin(phi*rad)
    z = r * np.cos(theta*rad)

    ax.text(x, y, z, text, **kwargs)

class disk_artist():
    def __init__(self, ax):
        self.ax = ax

    def plot_radius(self, r, nphi=30, **kwargs):
        """ plot the radius of the disk
        r : float or 1d array
            the radius to be drawn
        """
        ax = self.ax
        phi = np.linspace(0, 2*np.pi, nphi, endpoint=True)
        if isinstance(r, (list, np.ndarray)):
            for i in r:
                ax.plot(r*np.cos(phi), r*np.sin(phi), phi*0, **kwargs)
        else:
            ax.plot(r*np.cos(phi), r*np.sin(phi), phi*0, **kwargs)


    def plot_line_from_origin(self, r, theta, phi, **kwargs):
        """
        plot a line from the center 
        """
        x = np.array([0, r*np.sin(theta*rad)*np.cos(phi*rad)])
        y = np.array([0, r*np.sin(theta*rad)*np.sin(phi*rad)])
        z = np.array([0, r*np.cos(theta*rad)])

        self.ax.plot(x, y, z, **kwargs)

    def plot_axis(self, axis_on, vlen=1, **kwargs):
        """ plot the x, y, or z-axis from the center
        axis_on : tuple
            (x_on, y_on, z_on) with each as bool
        """
        lens = [(1,0,0), (0,1,0), (0,0,1)]

        for i_on, vdir in zip(axis_on, lens):
            if i_on:
                self.ax.arrow3D(
#                    0, 0, 0, 
                    -vlen * vdir[0], -vlen * vdir[1], -vlen * vdir[2],
                    2*vlen * vdir[0], 2*vlen * vdir[1], 2*vlen * vdir[2],
                    mutation_scale=20,
                    arrowstyle="-|>",
                    **kwargs)

    def plot_midplane_axis(self, vlen=1, **kwargs):
        """
        just plot the x and y axis
        less stringent than self.plot_axis()
        These are symmetric with respect to the origin
        """
        lens = [(1,0,0), (0,1,0)]
        for vdir in lens:
            self.ax.arrow3D(
                -vlen * vdir[0], -vlen * vdir[1], -vlen * vdir[2],
                    2*vlen * vdir[0], 2*vlen * vdir[1], 2*vlen * vdir[2],
                    mutation_scale=20,
                    arrowstyle="-|>",
                    **kwargs)

    def plot_zaxis(self, vlen=1, mode='top', **kwargs):
        """
        just plot the z-axis
        """
        vdir = np.array([0,0,1])
        if mode == 'both':
            butt = - vlen * vdir
            ds = 2 * vlen * vdir
        elif mode == 'top':
            butt = np.array([0,0,0])
            ds = vlen * vdir
        else:
            raise ValueError('mode unknown')

        self.ax.arrow3D(butt[0], butt[1], butt[2], ds[0], ds[1], ds[2],
            mutation_scale=20,
            arrowstyle="-|>",
            **kwargs)

    def plot_elevation(self, theta, philim=(0,90), r=1., **kwargs):
        """ plot the elevation circle
        """
        phi = np.linspace(philim[0], philim[1], 45, endpoint=True) * rad

        self.ax.plot(
            r * np.sin(theta*rad) * np.cos(phi),
            r * np.sin(theta*rad) * np.sin(phi),
            r * np.cos(theta*rad) + phi*0,
            **kwargs)

    def plot_longitude(self, phi_deg, thetalim=(0,90), r=1., **kwargs):
        """ plot a longitude line
        phi_deg : float
            azimuthal angle from the x-axis in degrees
        """
        theta = np.linspace(thetalim[0], thetalim[1], 45, endpoint=True)*rad
        phi = phi_deg * rad
        self.ax.plot(
            r * np.sin(theta)*np.cos(phi),
            r * np.sin(theta)*np.sin(phi),
            r * np.cos(theta), **kwargs)

    def plot_A(self, phi_deg, alen, nlen, r=1, **kwargs):
        """
        show the A angle, which is the angle between the alignment axis to the observer in the midplane
        We want to show the alignment axis and also the projected direction

        Parameters
        ----------
        phi_deg : float
            the azimuthal angle of the grain in consideration
        alen : float
            length of the axis we want
        nlen : float
            length of the vector to the observer
        """
        phi = phi_deg * rad

        # location of the origin
        o_x = r * np.cos(phi)
        o_y = r * np.sin(phi)
        o_z = 0

        # endpoint of the alignment axis
        a_axis = np.array([0, 1]) * alen
        a_vec = 0

class sky_artist(object):
    """
    plot the quantities related to an observer
    """
    def __init__(self, ax):
        self.ax = ax

    def set_inc(self, inc):
        """
        determine the inclination
        """
        self.inc = inc

    def get_ndir(self):
        """
        calculate the unit vector to the observer
        """
#        return np.array([-np.sin(self.inc), 0, np.cos(self.inc)])
        return np.array([0, -np.sin(self.inc), np.cos(self.inc)])

    def plot_ndir(self, vlen=1, proj_dist=1, **kwargs):
        """
        plot the direction to the observer
        The observer is in the y-z plane 
        proj_dist : float
            if we want to plot 
        """
        butt = np.zeros([3])
        ndir = self.get_ndir()

        ds = vlen * ndir

        self.ax.arrow3D(butt[0], butt[1], butt[2], ds[0], ds[1], ds[2],
            mutation_scale=20,
            arrowstyle="-|>",
            **kwargs)

        if proj_dist > 0:
            point = proj_dist * ndir
            x = np.array([point[0], point[0]])
            y = np.array([point[1], point[1]])
            z = np.array([0, point[2]])
            self.ax.plot(x, y, z, linestyle='--', **kwargs)

    def plot_principal_frame(self, vlen=0.5, origin_dist=1, **kwargs):
        """
        plot the principal frame axis
        Parameters
        ----------
        vlen : float
            the length of the arrows for the 
        origin_dist : float
            the distance from the disk center to the origin of the frame
        """
        zdir = np.array([0,0,1])
        # direction to observer
        ndir = self.get_ndir()

        # origin of the frame with respect to the disk frame
        origin = origin_dist * ndir

        # the x-axis of the frame
        p_xdir = np.cross(zdir, ndir)
        p_xdir /= np.sqrt(np.sum(p_xdir**2))

        # the y-axis of the frame
        p_ydir = np.cross(ndir, p_xdir)
        p_ydir /= np.sqrt(np.sum(p_ydir**2))

        # plot them
        for idir in [p_xdir, p_ydir]:
            ds = vlen * idir
            self.ax.arrow3D(origin[0], origin[1], origin[2], ds[0], ds[1], ds[2],
                mutation_scale=20,
                arrowstyle="-|>",
                **kwargs)

        # plot the origin
        self.ax.plot(origin[0], origin[1], origin[2], color='k', marker='o')

        # add a vector in this frame to show azimuth? 
        
def plot_system(r, phi, inc, xlim=None, ylim=None, zlim=None, 
    pdfname=None):
    """ just plot the disk coordinates
    and plot the grains at different phi of the disk
    """
    # settings for text
    fontdict = {
        'fontsize': 18
        }

    # prepare figure
    fig = plt.figure(figsize=(8,7))
#    ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')

    # create disk artist
    disk = disk_artist(ax)
    disk.plot_radius(r, color='k', nphi=64)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_box_aspect((1,1,1))

    # perspective of plot
    ax.view_init(elev=25, azim=-110)
    ax.dist = 6. # default is 10

    ax.set_axis_off()

    # plot disk origin
    ax.plot([0], [0], [0], color='k', marker='o')

    # coordinate axes
#    disk.plot_axis((True,True,True), vlen=1.3)

    disk.plot_midplane_axis(vlen=r*1.5, color='k')
    
    disk.plot_zaxis(vlen=r*1.3, mode='top', color='k', zorder=10)

    # coordinate text
    ax.text(1.3*r, -0.2, 0, r'$x_{d}$', fontdict=fontdict)
    ax.text(0, 1.5*r, 0.1, r'$y_{d}$', fontdict=fontdict)
    ax.text(0, 0.4, 1.2, r'$z_{d}$', fontdict=fontdict)

    # plot azimuthal angle
    phi = 30
    disk.plot_elevation(90, philim=(0, phi), r=0.5, color='k')
    disk.plot_line_from_origin(1, 90, phi, color='k')
    add_text_sph(ax, 0.6, 90, phi/2, r'$\phi_{d}$', fontdict=fontdict)

    # ==== plot sky related quantities ====
    sky = sky_artist(ax)

    # direction to observer
    sky.set_inc(inc)
    sky.plot_ndir(vlen=1.5*r, proj_dist=1*r, color='C0', zorder=10)
    add_text_sph(ax, 1.5*r, inc/rad, 270, r'$n$', fontdict=fontdict)

    # principal frame axes
    sky.plot_principal_frame(vlen=0.5*r, origin_dist=1*r, color='C1')
    add_text_sph(ax, 1.0*r, inc/rad, 310, r'$x_{i}$', fontdict=fontdict)
    add_text_sph(ax, 1.1*r, inc/rad-30, 270, r'$y_{i}$', fontdict=fontdict)

    # add the azimuthal angle in the principal frame

    # show the inclination angle
    disk.plot_longitude(270, thetalim=(0,inc/rad), r=0.2, color='k')
    add_text_sph(ax, 0.25, inc/rad/2, 270, r'$i$', fontdict=fontdict)

    fig.tight_layout()
    if pdfname is not None:
        fig.savefig(pdfname)
    plt.show()

def main():
    # ==== settings ====
    rdisk = 1. # radius of disk where the grain is

    # observer
    inc = 45 * rad

    # ==== plotting ====
    pngname = None
    phi = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pdfname = 'results/system_to_sky.pdf'
    plot_system(rdisk, phi, inc, 
        xlim=(-1.3,1.3), ylim=(-1.3,1.3), zlim=(-0.5, 1.3), pdfname=pdfname)


if __name__ == '__main__':
    main()

