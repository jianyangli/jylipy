# Package for polar image analysis.
#
# JYL @PSI, 4/22/2021

import numpy as np
from astropy.io import fits
from .core import centroiding, LinearFit

__all__ = ['PolarProjection', 'RadialProfile']

class PolarProjection():
    """Generate polar projection of an image.
    """

    def __init__(self, im, center=None, ra_bin=None, ra_max=None,
                 az_bin=361, fill_value=0., method='linear'):
        """
        Parameters
        ----------
        im : 2D array
            Image to be analyzed
        center : sequence of two numbers, optional
            The center pixel coordinate (yc, xc)
        rastep, azstep : number, optional
            The step sizes in radial (pixels) and azimuthal directions
            (degrees in ccw direction from up).  Default is 1 pixel
            and 1 degree, respectively.
        ramax : number, optional
            The maximum radial distance (om pixels) from the center in
            the analysis.  Default is the maximum integer pixel
            distance from the center in the image.
        fill_value : number, optional
            See keyword `fill_value` for `scipy.interpolate.interpn()`
        method : ['linear', 'nearest', 'splinef2d']
            See keyword `method` for `scipy.interpolate.interpn()`
        """
        self.im = im
        self._config = {'center': center,
                       'ra_bin': ra_bin,
                       'az_bin': az_bin,
                       'ra_max': ra_max,
                       'fill_value': fill_value,
                       'method': method}
        self.info = {'ra': np.array([]), 'az': np.array([])}
        self.im_polar = None

    @property
    def config(self):
        cf = self._config.copy()
        if cf['center'] is None:
            return cf
        if cf['ra_max'] is None:
            sz = self.im.shape
            cor = np.array([[0, 0], [sz[0], 0], [sz[0], sz[1]], [0, sz[1]]])
            cf['ra_max'] = int(np.floor(np.sqrt(((cor-cf['center'])**2).
                sum(axis=1)).max()))
        if cf['ra_bin'] is None:
            cf['ra_bin'] = cf['ra_max']+1
        return cf

    def centroid(self, **kwargs):
        """Centroiding the image interactively

        **kwargs : keyword arguments accepted by `jylipy.centroiding()`
        """
        self._config['center'] = jp.centroiding(self.im, **kwargs)

    def unwrap(self):
        """Unwrap image into polar coordinates.

        The unwrapped image will be saved to attribute `.im_polar`.
        The unwrapped image has azimuthal axis along the horizontal
        direction, and radial direction along verticle direction.  The
        azimuthal direction starts from PA=0, i.e., up direction,
        increases ccw to 360 deg.
        """
        config = self.config
        if config['center'] is None:
            raise ValueError('Center is not set.')
        center = config['center']

        self.info['ra'] = np.linspace(0, config['ra_max'], config['ra_bin'])
        self.info['az'] = np.linspace(0, 360, config['az_bin'])
        ra, az = np.meshgrid(self.info['ra'], self.info['az'], indexing='ij')
        yy0 = ra*np.sin(np.deg2rad(az+90)) + config['center'][0]
        xx0 = ra*np.cos(np.deg2rad(az+90)) + config['center'][1]
        xi = np.array((yy0.flatten(), xx0.flatten())).T
        sz = self.im.shape
        points = np.arange(sz[0]), np.arange(sz[1])
        from scipy.interpolate import interpn
        polar1d = interpn(points, self.im, xi, method=config['method'],
                        bounds_error=False, fill_value=config['fill_value'])
        self.im_polar = polar1d.reshape((config['ra_bin'],config['az_bin']))

    def write(self, file, **kwargs):
        """Save data to a fits file

        **kwargs : keyword arguments accepted by `astropy.io.fits.HDUList.writeto()`
        """
        hdu0 = fits.PrimaryHDU(self.im)
        config = self.config
        if config['center'] is not None:
            hdu0.header['yc'] = config['center'][0]
            hdu0.header['xc'] = config['center'][1]
        for k in ['ra_bin', 'az_bin', 'ra_max', 'method']:
            if config[k] is not None:
                hdu0.header[k] = config[k]
        hdu0.header['fill_val'] = config['fill_value']
        hdu0.header['unwrap'] = False
        hdulist = fits.HDUList([hdu0])
        if self.im_polar is not None:
            hdulist[0].header['unwrap'] = True
            hdulist.append(fits.ImageHDU(self.im_polar, name='polar'))
            hdulist.append(fits.ImageHDU(self.info['ra'], name='radial'))
            hdulist.append(fits.ImageHDU(self.info['az'], name='azimuth'))
        hdulist.writeto(file, **kwargs)

    @classmethod
    def from_fits(cls, file):
        """Load class object from FITS file
        """
        with fits.open(file) as _f:
            keys = [x.lower() for x in _f[0].header.keys()]
            kwargs = {}
            if 'yc' in keys:
                kwargs['center'] = _f[0].header['yc'], _f[0].header['xc']
            for k in ['ra_bin', 'az_bin', 'ra_max', 'method']:
                if k in keys:
                    kwargs[k] = _f[0].header[k]
            kwargs['fill_value'] = _f[0].header['fill_val']
            im = _f[0].data
            if _f[0].header['unwrap']:
                unwrap = True
                im_polar = _f[1].data
                ra = _f[2].data
                az = _f[3].data
            else:
                unwrap = False
        obj = cls(im, **kwargs)
        if unwrap:
            obj.im_polar = im_polar
            obj.info['ra'] = ra
            obj.info['az'] = az
        return obj


class RadialProfile(PolarProjection):
    """Analyze the radial profile of an image
    """

    def radial_profile(self, az):
        """Returns radial profile

        Parameters
        ----------
        az : number or 1D array of shape (m,)
            Azimuthal angle (ccw from up) in degrees to extract radial
            profile(s)

        Returns
        -------
        dist : 1D array of shape (n,)
            Distance to center in pixels
        profile : 1D array of shape (n,), or 2D array of shape (n, m)
            Radial profile at azimuthal angle `az`.
        """
        if self.im_polar is None:
            raise ValueError('Unwrapped image unavailalbe.  '
                             'Please run `.unwrap()` first.')
        sz = self.im_polar.shape
        az_ind = [int(np.round(x))
                for x in np.asarray(az)/360*(self.config['az_bin']-1)]
        return self.info['ra'], self.im_polar[:, az_ind]

    def radial_par(self, az, r1=5, r2=30, full=False):
        """Return parameters for radial profiles

        Radial profiles are fitted with model:
            y = a * x**b
        Fit is performed in with a linear fit in log space:
            log10(y) = log10(a) + b * log10(x)

        Parameters
        ----------
        az : number or 1D array of shape (m,)
            Azimuthal angle (ccw from up) in degrees to extract radial
            profile(s)
        r1, r2 : numbers, optional
            The inner and outer pixel distances within which a slope is fit
        full : bool, optional
            If `True`, then return the fitted parameters together with
            the `LinearFit` class object that contains more information
            about the fit.  Note, however, that the information stored
            in the `LinearFit` corresponding to the fit in log space,
            including the parameter errors in `.info['sigma']`.

        Returns
        -------
        b, a : arrays of shape (m)
            The fitted slope and interception parameters
        """
        r, pf = self.radial_profile(az)
        idx = (r > r1) & (r < r2)
        r = np.log10(r[idx])
        pf = np.log10(pf[idx])

        lf = LinearFit(r, pf)
        p = lf(full=full)
        p[1] = 10**p[1]
        if full:
            return p, lf
        else:
            return p

    def azimuth_profile(self, r):
        """Returns azimuth profile

        Parameters
        ----------
        r : number or 1D array of shape (m,)
            Distance in pixels from image center

        Returns
        -------
        az : 1D array of shape (n,)
            Azimuth angles in degrees
        profile : 1D array of shape (n,), or 2D array of shape (n, m)
            Azimuth profile at distance `r`
        """
        if self.im_polar is None:
            raise ValueError('Unwrapped image unavailalbe.  Please run'
                ' `.unwrap()` first.')
        sz = self.im_polar.shape
        r_ind = [int(np.round(x)) for x in np.asarray(r)]
