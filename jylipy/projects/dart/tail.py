# tools to study the tail

from ...saoimage import getds9
from .core import BrightnessProfile, BrightnessProfileSet


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

    def extract_profs(self, dist=None, width=1, type='az'):
        """Extract profiles

        dist : iterable of int
            Distances at which the profiles will be extracted.  This can
            be the distance along radial direction, or x or y directions
            of the image.  Default is from 0 to the largest possible in
            the image, with an incremental of 1.
        width : odd int
            Width along the `dist` direction to be averaged.  If not odd,
            then the next odd number is used, and a warning is issued.
        type : str, ['az', 'x', 'y']
            The type of profiles to be extracted.

        The extracted profiles will be stored in attribute `._azprofs`,
        `.xprofs`, or `.yprofs`, depending on the value of `type`, and
        then returned.
        """
        pass

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
