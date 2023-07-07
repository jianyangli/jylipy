# core functions for thermal modeling submodule

import abc
import numpy as np
from scipy.integrate import dblquad
import astropy.units as u
from astropy.modeling.models import BlackBody
from ..vector import twovec, xyz2sph, sph2xyz

class ThermalModelABC(abc.ABC):
    """Abstract base class for thermal models.

    This class implements the basic calculation for thermal models,
    such as integration of total flux based on a temperature distribution.
    """

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

    @u.quantity_input(wave_freq=u.m, delta=u.m, lon=u.deg, lat=u.deg,
                      equivalencies=u.spectral())
    def fluxd(self, wave_freq, delta, sublon, sublat):
        """Total thermal flux density of an object

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
        def int_func(lon, lat, m, unit):
            # The integration is performed in a frame where the sub-observer
            # point is at lon = 0, lat = 0.  So the [lon, lat] passed to this
            # function needs to be converted to the body-fixed frame first via
            # coordinate transformation matrix `m` passed to the function:
            #     m.dot(v)
            # in order to calculate the temperature and then flux.
            #
            # Default unit of `lon` and `lat` is radiance
            #
            # All input parameters and returned values should be float point.
            # Parameter `unit` is used to specify the unit of the integrated
            # quantity.
            _, lon1, lat1 = xyz2sph(
                m.dot(sph2xyz([np.rad2deg(lon), np.rad2deg(lat)]))
                )
            lon1 *= u.deg
            lat1 *= u.deg
            f = BlackBody(self.T(lon1, lat1))(wave_freq) * np.cos(lat)
            return f.to_value(unit)

        unit = 'W m-2 Hz-1 sr-1'
        m = twovec([sublon.to_value('deg'), sublat.to_value('deg')], 0,
                       [0, 90], 2).T
        f, _ = dblquad(int_func,
                       -np.pi/2,
                       np.pi/2,
                       lambda x: -np.pi/2,
                       lambda x: np.pi/2,
                       args=(m, unit))
        return u.Quantity(f, unit) * ((self.R / delta)**2).to('sr',
            u.dimensionless_angles()) * self.beaming * self.emissivity
