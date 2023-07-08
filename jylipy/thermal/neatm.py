# Near-Earth Asteroid Thermal Model (NEATM)
# References
#   Harris, A.W., 1998.  A Thermal Model for Near-Earth Asteroids.
#     Icarus 131, 291-301

import numpy as np
import astropy.units as u
from sbpy.bib import cite
from .core import ThermalModelABC, NonRotTempDist

__all__ = ['NEATM']

class NEATM(NonRotTempDist, ThermalModelABC):
    """Standard thermal model"""

    @cite({'method': '1998Icar..131..291H'})
    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)

    @u.quantity_input(phase=u.deg)
    def fluxd(self, wave_freq, delta, phase=0*u.deg, **kwargs):
        """Calculate total flux density.
        """
        sublon = phase
        sublat = 0. * u.deg
        return super().fluxd(wave_freq, delta, sublon, sublat, **kwargs)
