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

class STM(ThermalModelABC):
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
        self.rh = rh
        self.R = R
        self.albedo = albedo
        self.emissivity = emissivity
        self.beaming = beaming

    @property
    def Tss(self):
        """Subsolar temperature"""
        f_sun = const.L_sun / (4 * np.pi * self.rh**2)
        return (((1 - self.albedo) * f_sun / (self.beaming * self.emissivity
            * const.sigma_sb)) ** 0.25).decompose()

    @u.quantity_input(lon=u.deg, lat=u.deg)
    def T(self, lon, lat):
        """Surface temperature at specific lat, lon"""
        coslon = np.cos(lon)
        coslat = np.cos(lat)
        prec = np.finfo(coslat.value).resolution
        if (abs(coslon) < prec) or (abs(coslat) < prec) or (coslon < 0):
            return 0 * u.K
        else:
            return self.Tss * (coslon * coslat)**0.25

    def fluxd(self, wave_freq, delta):
        sublon = 0. * u.deg
        sublat = 0. * u.deg
        return super().fluxd(wave_freq, delta, sublon, sublat)
