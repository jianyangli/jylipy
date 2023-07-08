# Near-Earth Asteroid Thermal Model (NEATM)
# References
#   Harris, A.W., 1998.  A Thermal Model for Near-Earth Asteroids.
#     Icarus 131, 291-301

import numpy as np
import astropy.units as u
import astropy.constants as const
from sbpy.bib import cite
from .core import ThermalModelABC

__all__ = ['NEATM']

class NEATM(ThermalModelABC):
    """Standard thermal model"""

    @cite({'method': '1998Icar..131..291H'})
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def Tss(self):
        """Subsolar temperature"""
        f_sun = const.L_sun / (4 * np.pi * self.rh**2)
        return (((1 - self.albedo) * f_sun / (self.beaming * self.emissivity
            * const.sigma_sb)) ** 0.25).decompose()

    @u.quantity_input(lon=u.deg, lat=u.deg)
    def T(self, lon, lat):
        """Surface temperature at specific (lat, lon)

        lon : u.Quantity in units equivalent to deg
            Longitude
        lat : u.Quantity in units equivalent to deg
            Latitude

        Returns
        -------
        u.Quantity : Surface temperature.
        """
        coslon = np.cos(lon)
        coslat = np.cos(lat)
        prec = np.finfo(coslat.value).resolution
        if (abs(coslon) < prec) or (abs(coslat) < prec) or (coslon < 0):
            return 0 * u.K
        else:
            return self.Tss * (coslon * coslat)**0.25

    @u.quantity_input(phase=u.deg)
    def fluxd(self, wave_freq, delta, phase=0*u.deg, **kwargs):
        """Calculate total flux density.
        """
        sublon = phase
        sublat = 0. * u.deg
        return super().fluxd(wave_freq, delta, sublon, sublat, **kwargs)
