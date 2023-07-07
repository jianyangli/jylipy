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
    @u.quantity_input(rh=u.km, R=u.km, albedo=u.dimensionless_unscaled,
        emissivity=u.dimensionless_unscaled, beaming=u.dimensionless_unscaled)
    def __init__(self, rh, R, albedo=0.1, emissivity=1., beaming=1.):
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

    @u.quantity_input(lat=u.deg, lon=u.deg)
    def T(self, lat, lon):
        """Surface temperature at specific lat, lon"""
        return self.Tss * (np.cos(lat) * np.cos(lon))**0.25

    @u.quantity_input(wave_freq=u.m, delta=u.km, phase=u.deg,
        equivalencies=u.spectral())
    def flux(self, wave_freq, delta, phase=0.*u.deg):
        """Total observed thermal flux

        wave_freq : u.Quantity
            Wavelength or frequency of observations
        delta : u.Quantity
            Observer range
        phase : u.Quantity
            Phase angle of observations
        """
        return super().flux(wave_freq, delta, phase, 0 * u.deg)
