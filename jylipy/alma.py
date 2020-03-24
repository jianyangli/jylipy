'''ALMA planning and data analysis package'''

from .geometry import load_generic_kernels, unload_generic_kernels, obsgeom
from .core import ascii_read
from .apext import Time, time, units
from .plotting import pplot
from matplotlib import pyplot as plt
import spiceypy as spice
import numpy as np

class ALMA:
    '''ALMA class that contains the characteristics of ALMA'''
    lat = -23.02  # ALMA latitude


def obs_condition(target, schedule, spk=None, savefig=None):
    '''Generate plot to show the observational condition of ALMA under
    the specified configuration schedule

    target: str, the name of target to consider
    schedule: str, file name of csv table that lists the configuration schedule
    spk: str, SPK kernel name
    savefig: str, file name to save the figure
    '''

    schedule = ascii_read(schedule)

    if spk is not None:
        load_generic_kernels()
        spice.furnsh(spk)

    f,ax = plt.subplots(3,1,sharex=True,num=plt.gcf().number)
    for s in schedule:
        bmsz = 1.22*1.2e-3/s['baseline']/1000*206265  # diffraction limit in arcsec
        ts = Time(s['start'])+time.TimeDelta(1*units.hour)*np.linspace(0,s['duration']*24,s['duration']*24+1)
        #geom = obsgeom(ts.isot, 'ceres')
        #diam = 2*Dawn.Ceres.r.value/geom['range']/1.496e8*206265
        geom = obsgeom(ts.isot, 'psyche')
        diam = 226/geom['range']/1.496e8*206265
        res = bmsz/np.cos(np.deg2rad(geom['dec']-ALMA.lat))
        ax[0].plot_date(ts.plot_date, res,'r-',lw=5)
        ax[0].plot_date(ts.plot_date, diam,'b-',lw=5)
        ax[0].plot_date(ts.plot_date, np.repeat(s['mrs'],len(ts)),'g-',lw=3)
        ax[1].plot_date(ts.plot_date, diam/res,'-k',lw=5)
        ax[2].plot_date(ts.plot_date, geom['selong'],'r-',lw=5)
        ax[2].plot_date(ts.plot_date, geom['phase'],'g-',lw=5)
        ax[2].plot_date(ts.plot_date, geom['ra'],'b-',lw=5)

    pplot(ax[0],ylabel='arcsec',yscl='log',skipline=True)
    ax[0].legend(['Sky Resolution','Target Diameter','Max Recoverable Scale (Band 6)'])
    ax[0].grid(which='both')
    pplot(ax[1],ylabel='Number of Beams',skipline=True)
    ax[1].legend(['Target Diameter'])
    ax[1].grid()
    pplot(ax[2],ylabel='Angles (deg)',xlabel='Date',skipline=True)
    ax[2].legend(['Solar Elongation','Phase Angle','RA'])
    ax[2].grid()

    plt.show()

    if spk is not None:
        unload_generic_kernels()
        spice.unload(spk)

    if savefig is not None:
        plt.savefig(savefig)
