# tools to study the tail

import warnings
import numpy as np
import astropy.units as u
from ...saoimage import getds9
from .core import BrightnessProfile, BrightnessProfileSet, AzimuthalProfile


class FeatureModel(BrightnessProfile):
    """Model a feature in the image

    Attributes
    ----------
    .image : 2d array or u.Quantity
        Image to be processed
    .unwrapped : 2d array or u.Quantity
        Unwrapped image in (r, theta)
    .center : 2-element sequence of int
        Centroid pixel coordinate [y, x]
    .info : dict or any object that has the form of info['key'] = value
        Information data
    .azprofs : `BrightnessProfileSet` object
        Azimuthal profiles.  Added by `.extract_profs()`.
    .xprofs : `BrightnessProfileSet` object
        Profiles in x-direction.  Added by `.extract_profs()`.
    .yprofs : `BrightnessProfileSet` object
        Profiles in y-direction.  Added by `.extract_profs()`.
    .par : dict
        Model parameters.  Added by `.parameterize()`.
    """

    def __init__(self, image=None, unwrapped=None, center=None, info=None):
        """
        image : 2d array
            Image to be processed
        unwrapped : 2d array
            Unwrapped image in (r, theta)
        center : 2-element sequence of int
            Centroid pixel coordinate [y, x].  Default is image center.
        info : dict, or any object that has the form of info['key'] = value
            Information data
        """
        self.image = image
        self.unwrapped = unwrapped
        self.center = center if (center is not None) or (image is None) \
                      else u.Quantity(image.shape, u.pix) / 2
        self.info = info

    def imdisp(self, ds9=None, unwrapped=False, ds9par=[]):
        """Display images"""
        if ds9 is None:
            ds9 = getds9('tail')
        if unwrapped:
            if self.unwrapped is None:
                raise ValueError('unwrapped image not available.')
            ds9.imdisp(self.unwrapped)
        else:
            if self.image is None:
                raise ValueError('image not available.')
            ds9.imdisp(self.image)
        ds9.set('scale log')
        ds9.set('cmap cool')
        for p in ds9par:
            ds9.set(p)

    def extract_profs(self, dist=None, width=1, kind='az'):
        """Extract profiles

        dist : iterable of int
            Distances at which the profiles will be extracted.  This can
            be the distance along radial direction, or x or y directions
            of the image.  Default is from 0 to the largest possible in
            the image, with an incremental of 1.
        width : odd int
            Width along the `dist` direction to be averaged.  If not odd,
            then the next odd number is used, and a warning is issued.
        kind : str, ['az', 'x', 'y']
            The type of profiles to be extracted.

        The extracted profiles will be stored in attribute `._azprofs`,
        `.xprofs`, or `.yprofs`, depending on the value of `type`, and
        then returned.
        """
        if width % 2 == 0:
            warnings.warn('width is even number: {}, use an odd number: {}'.
                format(width, width + 1))
            width += 1
        w2 = width // 2
        if kind in ['az']:
            if not hasattr(self, 'unwrapped'):
                raise ValueError('unwrapped image unavailable.')
            if dist is None:
                nr = self.unwrapped.shape[0]  # number of radial pixels
                dist = np.arange(nr)
            naz = self.unwrapped.shape[1]  # number of azimuthal angles
            daz = 360 / naz  # azimuthal step size
            az = np.linspace(0, (naz - 1) * daz, naz)
            meta = {'dist': dist,
                    'width': width}
            self._azprofs = BrightnessProfileSet(
                [AzimuthalProfile(
                    self.unwrapped[i-w2:i+w2+1].mean(axis=0),
                    x=u.Quantity(az, u.deg),
                    info=self.info,
                    )
                for i in dist],
                meta=meta)
            return self._azprofs
        elif kind in ['x']:
            if not hasattr(self, 'image'):
                raise ValueError('image unavailable.')
            if dist is None:
                dist = range(self.image.shape[0])
            meta = {'dist': dist,
                    'width': width}
            self._xprofs = BrightnessProfileSet(
                [BrightnessProfile(self.image[y-w2:y+w2+1].mean(axis=0),
                    x=u.Quantity(range(self.image.shape[1]), u.pix),
                    info=self.info)
                for y in dist],
                meta=meta)
            return self._xprofs
        elif kind in ['y']:
            if not hasattr(self, 'image'):
                raise ValueError('image unavailable.')
            if dist is None:
                dist = range(self.image.shape[1])
            meta = {'dist': dist,
                    'width': width}
            self._yprofs = BrightnessProfileSet(
                [BrightnessProfile(self.image[:, x-w2:x+w2+1].mean(axis=1),
                    x=u.Quantity(range(self.image.shape[0]), u.pix),
                    info=self.info)
                for x in dist],
                meta=meta)
            return self._yprofs
        else:
            raise ValueError('unrecognized profile type: {}'.format(kind))

    @property
    def azprofs(self):
        """Azimuthal profiles"""
        return getattr(self, '_azprofs', None)

    @property
    def yprofs(self):
        return getattr(self, '_yprofs', None)

    @property
    def xprofs(self):
        return getattr(self, '_xprofs', None)

    def parameterize(self, p0, width, kind='az', fit_index=slice(None),
            **kwargs):
        """Parameterize the feature from modeling.

        This method will fit the extracted profiles to characterize the
        feature by its position angle, full-width-half-max, and peak
        brightness.

        It calls `BrightnessProfile.peak()` to parameterize the model.
         for *args and **kwargs.

        Parameters
        ----------
        p0 : number, u.Quantity or sequence of them
            Initial peak positions
        width : number, u.Quantity or sequence of them
            Width along the profile to search for peak
        kind : str, ('az', 'x', 'y')
            Use which profiles to parameterize
        fit_index : int or slice
            The indexes of profiles to be fitted
        kwargs : dict
            Keyword parameters for `BrightnessProfile.peak()`
        """
        # initial process: `kind`
        if kind in ['az']:
            profs = self.azprofs
        elif kind in ['x']:
            profs = self.xprofs
        elif kind in ['y']:
            profs = self.yprofs
        else:
            raise ValueError('unrecognized profile type.')

        # initial process: `p0`
        nprofs = len(profs[fit_index])
        if isinstance(p0, u.Quantity):
            if len(p0.shape) == 0:
                p0 = u.Quantity([p0] * nprofs)
        else:
            if not hasattr(p0, '__iter__'):
                p0 = np.full(nprofs, p0)
        if len(p0) != nprofs:
            raise ValueError('the length of initial peak position p0 '
                '{} is differnet from the length of profiles {}'.format(
                    len(p0), nprofs))

        # initial process: `width`
        if isinstance(width, u.Quantity):
            if len(width.shape) == 0:
                width = u.Quantity([width] * nprofs)
        else:
            if not hasattr(width, '__iter__'):
                width = np.full(nprofs, width)
        if len(width) != nprofs:
            raise ValueError('the length of initial peak position p0 '
                '{} is differnet from the length of profiles {}'.format(
                    len(width), nprofs))

        # prepare output
        keys = ['peak', 'amplitude', 'fwhm']
        par = {k: [] for k in keys}
        # fit peaks for all profiles
        for i, p in enumerate(profs[fit_index]):
            p.peak(p0[i], width[i], **kwargs)
            for k, v in p.par.items():
                par[k].append(v)
        # post processing
        for k, v in par.items():
            par[k] = u.Quantity(v)
        if kind in ['az']:
            # add peak position pixel coordinate
            par['peak_pa'] = par.pop('peak')
            par['peak_x'] = (profs.meta['dist'][fit_index]
                            * np.cos(par['peak_pa']) + self.center[1])
            par['peak_y'] = (profs.meta['dist'][fit_index]
                            * np.sin(par['peak_pa']) + self.center[0])
        else:
            if kind in ['y']:
                par['peak_y'] = par.pop('peak')
                par['peak_x'] = profs.meta['dist'][fit_index] * u.pix
            elif kind in ['x']:
                par['peak_x'] = par.pop('peak')
                par['peak_y'] = profs.meta['dist'][fit_index] * u.pix
            else:
                raise ValueError('unrecognized profile type {}.'.format(kind))
            y_ = par['peak_y'] - self.center[0]
            x_ = par['peak_x'] - self.center[1]
            par['peak_pa'] = (np.arctan2(y_, x_) + 1.5 * np.pi * u.rad) % (
                                2 * np.pi * u.rad)

        self.par = par

    def plot_feature(self):
        """Plot the parameters of feature
        """
        pass

    def overlay(self):
        """Show an overlay of image and feature model
        """
        pass
