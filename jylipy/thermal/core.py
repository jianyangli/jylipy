# core functions for thermal modeling submodule

import abc
import astropy.units as u

class ThermalModel(abc.ABC):
    """Abstract base class for thermal models.

    This class implements the basic calculation for thermal models,
    such as integration of total flux based on a temperature distribution.
    """

    @abc.abstractmethod
    @u.quantity_input(lon=u.deg, lat=u.deg)
    def T(self, lon, lat):
        """Temperature on the surface of an object.

        Needs to be overridden in subclasses.

        lon : u.Quantity
            Longitude
        lat : u.Quantity
            Latitude
        """
        pass

    @u.quantity_input(wave_freq=u.m, delta=u.m, lon=u.deg, lat=u.deg,
                      equivalencies=u.spectral())
    def flux(self, wave_freq, delta, lon, lat):
        """Total thermal flux of an object

        Parameters
        ----------
        wave_freq : u.Quantity
            Wavelength or frequency of observations
        delta : u.Quantity
            Observer range
        lon : u.Quantity
            Observer longitude in target-fixed frame
        lat : u.Quantity
            Observer latitude in target-fixed frame

        Returns
        -------
        u.Quantity : Integrated flux
        """
        pass
