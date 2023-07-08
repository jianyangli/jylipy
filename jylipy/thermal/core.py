# core functions for thermal modeling submodule

import abc
import numpy as np
from scipy.integrate import dblquad
import astropy.units as u
import astropy.constants as const
from astropy.modeling.models import BlackBody
from ..vector import twovec, xyz2sph, sph2xyz


__all__ = ['ThermalModelABC', 'InstEquiTempDist', 'NonRotTempDist',
           'FastRotTempDist']


class ThermalModelABC(abc.ABC):
    """Abstract base class for thermal models.

    This class implements the basic calculation for thermal models,
    such as integration of total flux based on a temperature distribution.
    """

    @u.quantity_input()
    def __init__(self, rh, R, albedo=0.1, emissivity=1., beaming=1.):
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
        self.rh = rh
        self.R = R
        self.albedo = albedo
        self.emissivity = emissivity
        self.beaming = beaming

    @abc.abstractmethod
    @u.quantity_input(lon=u.deg, lat=u.deg)
    def T(self, lon, lat) -> u.T:
        """Temperature on the surface of an object.

        Needs to be overridden in subclasses.  This function needs to be able
        to return a valid quantity for the full range of lon and lat, i.e.,
        include the night side of an object.

        lon : u.Quantity
            Longitude
        lat : u.Quantity
            Latitude
        """
        pass

    @u.quantity_input(wave_freq=u.m, equivalencies=u.spectral())
    def _int_func(self, lon, lat, m, unit, wave_freq):
        """Integral function for `fluxd`.

        Parameters
        ----------
        lon : float
            Longitude in radiance
        lat : float
            Latitude in fradiance
        m : numpy array of shape (3, 3)
            Transformation matrix to convert a vector in the frame to perform
            integration to body-fixed frame.  This matrix can be calculated
            with private method `_transfer_to_bodyframe`.  The integration
            is performed in a frame where the sub-observer point is defined at
            lon = 0, lat = 0.
        unit : str or astropy.units.Unit
            Unit of the integral function.
        wave_freq : u.Quantity
            Wavelength or frequency of calculation

        Returns
        -------
        float : Integral function to calculate total flux density.
        """
        _, lon1, lat1 = xyz2sph(
            m.dot(sph2xyz([np.rad2deg(lon), np.rad2deg(lat)]))
            )
        lon1 *= u.deg
        lat1 *= u.deg
        T = self.T(lon1, lat1)
        if np.isclose(T, 0 * u.K):
            return 0.
        else:
            f = BlackBody(T)(wave_freq) * np.cos(lat)
            return f.to_value(unit)

    @staticmethod
    @u.quantity_input(sublon=u.deg, sublat=u.deg)
    def _transfer_to_bodyframe(sublon, sublat):
        """Calculate transformation matrix.

        The numerical integration to calculate total flux density is performed
        in a reference frame where the sub-observer point is at
        (lon, lat) = (0, 0).  This matrix supports the transformation from
        this frame to the body-fixed frame to facilitate the calculation of
        surface temperature.
        """
        coslat = np.cos(sublat).value
        if abs(coslat) < np.finfo(type(coslat)).resolution:
            if sublat.value > 0:
                m = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            else:
                m = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        else:
            m = twovec([sublon.to_value('deg'), sublat.to_value('deg')], 0,
                       [0, 90], 2).T
        return m

    @u.quantity_input(wave_freq=u.m, delta=u.m, lon=u.deg, lat=u.deg,
                      equivalencies=u.spectral())
    def fluxd(self, wave_freq, delta, sublon, sublat):
        """Calculate total thermal flux density of an object.

        Parameters
        ----------
        wave_freq : u.Quantity
            Wavelength or frequency of observations
        delta : u.Quantity
            Observer range
        sublon : u.Quantity
            Observer longitude in target-fixed frame
        sublat : u.Quantity
            Observer latitude in target-fixed frame

        Returns
        -------
        u.Quantity : Integrated flux
        """
        unit = 'W m-2 Hz-1 sr-1'
        m = self._transfer_to_bodyframe(sublon, sublat)
        f, _ = dblquad(self._int_func,
                       -np.pi/2,
                       np.pi/2,
                       lambda x: -np.pi/2,
                       lambda x: np.pi/2,
                       args=(m, unit, wave_freq))
        return u.Quantity(f, unit) * ((self.R / delta)**2).to('sr',
            u.dimensionless_angles()) * self.beaming * self.emissivity


class InstEquiTempDist():
    """Instantaneous temperature distribution class.

    A number of thermal models uses instantaneous equilibrium temperature
    distribution for asteroids, such as STM, FRM, and NEATM.  This class
    performs some basic calculation for such a model and can be used as
    the base class for those models.

    """

    @u.quantity_input(rh=u.km, albedo=u.dimensionless_unscaled,
        emissivity=u.dimensionless_unscaled, beaming=u.dimensionless_unscaled)
    def __init__(self, rh, R, albedo=0.1, emissivity=1., beaming=1.):
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


class NonRotTempDist(InstEquiTempDist):
    """Non-rotating object temperature distribution, i.e., STM
    """

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


class FastRotTempDist(InstEquiTempDist):
    """Fast-rotating object temperature distribution, i.e., FRM
    """

    @property
    def Tss(self):
        """Subsolar temperature"""
        return super().Tss * (self.beaming / np.pi)**0.25

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
        coslat = np.cos(lat)
        return self.Tss * coslat**0.25
