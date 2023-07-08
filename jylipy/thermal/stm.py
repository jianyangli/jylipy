# Standard thermal model
# References
#   Lebofsky, L.A., et al., 1986.  A refined "standard" thermal model for
#     asteroids based on observatons of 1 Ceres and 2 Pallas.  Icarus 68,
#     239-251.

import numpy as np
import astropy.units as u
import astropy.constants as const
from sbpy.bib import cite
from .core import ThermalModelABC
from .neatm import NEATM

__all__ = ['STM']

class STM(ThermalModelABC, NEATM):
    """Standard thermal model"""

    @cite({'method': '1986Icar...68..239L'})
    def __init__(self, rh, R, albedo=0.1, emissivity=1.):
        """Initialization

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
        beaming = 0.756
        super().__init__(rh, R, albedo, emissivity, beaming)

    @u.quantity_input(phase=u.deg, phase_slope=u.mag/u.deg)
    def fluxd(self, wave_freq, delta, phase=0*u.deg,
            phase_slope=0.01*u.mag/u.deg, **kwargs):
        """Calculate total flux density.
        """
        scl = u.Magnitude((phase * phase_slope).to_value('mag')).physical
        sublon = 0. * u.deg
        sublat = 0. * u.deg
        return super().fluxd(wave_freq, delta, sublon, sublat, **kwargs) * scl
