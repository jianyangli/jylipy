# Standard thermal model
# References
#   Lebofsky, L.A., et al., 1986.  A refined "standard" thermal model for
#     asteroids based on observatons of 1 Ceres and 2 Pallas.  Icarus 68,
#     239-251.

import numpy as np
import astropy.units as u
from sbpy.bib import cite
from .core import ThermalModelABC, NonRotTempDist

__all__ = ['STM']

class STM(NonRotTempDist, ThermalModelABC):
    """Standard thermal model"""

    @cite({'method': '1986Icar...68..239L'})
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        rh : u.Quantity
            Heliocentric distance
        R : u.Quantity
            Radius of asteroid
        albedo : float, u.Quantity
            Bolometric Bond albedo
        emissivity : float, u.Quantity
            Emissivity of surface
        beaming : float, u.Quantity
            Beaming parameter
        """
        kwargs.setdefault('beaming', 0.756)
        super().__init__(*args, **kwargs)

    @staticmethod
    @u.quantity_input(phase=u.deg, phase_slope=u.mag/u.deg)
    def _phase_corr(phase, phase_slope):
        return u.Magnitude((phase * phase_slope).to_value('mag')).physical

    @u.quantity_input(phase=u.deg, phase_slope=u.mag/u.deg)
    def fluxd(self, wave_freq, delta, phase=0*u.deg,
            phase_slope=0.01*u.mag/u.deg, **kwargs):
        """Calculate total flux density.
        """
        scl = self._phase_corr(phase, phase_slope)
        sublon = 0. * u.deg
        sublat = 0. * u.deg
        return super().fluxd(wave_freq, delta, sublon, sublat, **kwargs) * scl
