# core functions for thermal modeling submodule

import abc
from scipy.integrate import dblquad
import astropy.units as u
from astropy.modeling.models import BlackBody
from ..vector import twovec, xyz2sph, sph2xyz

class ThermalModel(abc.ABC):
    """Abstract base class for thermal models.

    This class implements the basic calculation for thermal models,
    such as integration of total flux based on a temperature distribution.
    """

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
    def flux(self, wave_freq, delta, sublon, sublat):
        """Total thermal flux of an object

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
        @u.quantity_input(lon=u.deg, lat=u.deg)
        def bb_flux(lon, lat, m):
            # The integration is performed in a frame where the sub-observer
            # point is at lon = 0, lat = 0.  So the [lon, lat] passed to this
            # function needs to be converted to the body-fixed frame first via
            # coordinate transformation matrix `m` passed to the function:
            #     m.dot(v)
            # in order to calculate the temperature and then flux.
            _, lon1, lat1 = xyz2sph(
                m.dot(sph2xyz([lon.to_value('deg'), lat.to_value('deg')]))
                )
            lon1 *= u.deg
            lat1 *= u.deg
            return BlackBody(self.T(lon1, lat1))(wave_freq) * np.cos(lat)

        m = twovec([sublon.to_value('deg'), sublat.to_value('deg')], 0,
                   [0, 90], 2).T
        return dblquad(bb_flux,
                       -np.pi/2,
                       np.pi/2,
                       lambda x: -np.pi/2,
                       lambda x: np.pi/2,
                       args=(m,))
