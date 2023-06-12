# tools to study the tail

import warnings
import numpy as np
import astropy.units as u
from ...saoimage import getds9
from .core import BrightnessProfile, BrightnessProfileSet, AzimuthalProfile


class FeatureModel(BrightnessProfile):
    """Model a feature in the image"""

    def __init__(self, image=None, unwrapped=None, center=None):
        """
        image : 2d array
            Image to be processed
        unwrapped : 2d array
            Unwrapped image in (r, theta)
        center : 2-element sequence of int
            Centroid pixel coordinate [y, x].  Default is image center.
        """
        self.image = image
        self.unwrapped = unwrapped
        self.center = center

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
                    x=u.Quantity(az, u.deg)
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
                [BrightnessProfile(self.image[y-w2:y+w2+1].mean(axis=0))
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
                [BrightnessProfile(self.image[:, x-w2:x+w2+1].mean(axis=1))
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

    def parameterize(self):
        """Parameterize the feature from modeling.

        This method will fit the extracted profiles to characterize the
        feature by its position angle, full-width-half-max, and peak
        brightness.
        """
        pass

    def plot_feature(self):
        """Plot the parameters of feature
        """
        pass

    def overlay(self):
        """Show an overlay of image and feature model
        """
        pass
