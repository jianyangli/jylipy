"""HST CSS data processing and analysis library"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Sine1D
from astropy.modeling import Parameter
from ..apext import Time


class Sine1D_C(Sine1D):
    """1D Sinusoidal model with a DC component"""
    dc = Parameter(default=0)

    @staticmethod
    def evaluate(x, a, f, p, dc):
        return Sine1D.evaluate(x, a, f, p) + dc

    @staticmethod
    def fit_deriv(x, a, f, p, dc):
        d1, d2, d3 = Sine1D.fit_deriv(x, a, f, p)
        return [d1, d2, d3, x/x]


class fit_amplitude():
    def __init__(self, phos, apts, binning=[3, 1, 1], shift=[0, -0.204, 0.181]):
        """Fit lightcurve amplitude for various aperture size
        phos : list of fits.HDUList
            Each HDU correponds to photometric data from one filter
        apts : number or array like number
            Aperture size for fitting
        binning : list of int, optional
            Binning factors for each filter.  3 for F775W, 1 for F689M and F845M
        shift : list of float, optional
            Shift factor for each filter in order to scale magnitudes together.
            Shifted magnitude = mag + shift.
        """
        self.phos = phos
        self.apertures = apts
        self.binning = binning
        self.shift = shift
        self.time = np.concatenate([(Time(p[1].data['Date']).jd \
                - 2456949.5) for p in phos])

    def rotph(self, period=8):
        """Rotational phase starting from 2014-10-19"""
        return self.time * 24 / period % 1

    def mag(self, apt):
        """Magnitudes and error at aperture size `apt`"""
        m = np.concatenate([p['vegamag'].data[:, apt//b-1] + s for p, b, s \
                in zip(self.phos, self.binning, self.shift)])
        e = np.concatenate([abs(p['magerror'].data[:, apt//b-1]) for p, b \
                in zip(self.phos, self.binning)])
        return m, e

    def fit(self, apt=None, rotph=True, fix_freq=True):
        """Return fitted lightcurve amplitude for aperture size `apt`

        If `rotph` is True, then the fitting is performed w/r to rotational
        phase, otherwise w/r to time (in days)
        """
        from astropy.modeling.fitting import LevMarLSQFitter
        from astropy import table
        from jylipy import Table
        self.fit_in_rotph = rotph
        f = LevMarLSQFitter(calc_uncertainties=True)
        if apt is None:
            apt = self.apertures
        if not hasattr(apt, '__iter__'):
            m, e = self.mag(apt)
            if rotph:
                #print('a')
                sc0 = Sine1D_C(amplitude=0.2, frequency=1, phase=0, dc=16.2)
                sc0.frequency.fixed = fix_freq
                sc = f(sc0, self.rotph(), m, weights=1/e)
            else:
                #print('b')
                sc0 = Sine1D_C(amplitude=0.2, frequency=3, phase=0, dc=16.2)
                sc0.frequency.fixed = fix_freq
                sc = f(sc0, self.time, m, weights=1/e)
            if sc.amplitude.value < 0:
                sc.amplitude.value = -sc.amplitude.value
                sc.phase.value = (sc.phase.value + 0.5 ) %1
            if (sc.phase.value < 0) or (sc.phase.value > 1):
                sc.phase.value = (sc.phase.value % 1 + 1) % 1
            pars = [[apt], [sc.amplitude.value], [sc.phase.value],
                    [sc.frequency.value], [sc.dc.value]]
            par_names = ['aperture', 'amplitude', 'phase', 'frequency', 'dc']
            stds = list(np.array(sc.stds.stds).reshape(-1, 1))
            std_names = [x+'_error' for x in sc.stds.param_names]
            self.pars = Table(pars + stds, names=par_names + std_names)
            return self.pars
        else:
            pars = [self.fit(apt=a, rotph=rotph,fix_freq=fix_freq) for a in apt]
            self.pars = table.vstack(pars)
            return self.pars

    def plot(self, apt, ax=None, rotph=None, **figure_kwargs):
        rotph = getattr(self, 'fit_in_rotph', True)
        if ax is None:
            fig = plt.figure(**figure_kwargs)
            ax = fig.add_subplot(111)
        if not hasattr(apt, '__iter__'):
            apt = [apt]
        for a in apt:
            par = self.pars.query('aperture', a)
            if len(par) > 0:
                xx = np.linspace(0, 1, 100) if rotph else \
                        np.linspace(self.time.min(), self.time.max(), 100)
                x0 = self.rotph() if rotph else self.time
                sc = Sine1D_C(amplitude=par['amplitude'],
                              frequency=par['frequency'], phase=par['phase'],
                              dc=par['dc'])
                m, e = self.mag(a)
                ax.errorbar(x0, m, e, fmt='o')
                ax.plot(xx, sc(xx))
