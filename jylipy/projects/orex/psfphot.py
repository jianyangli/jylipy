# PSF photometry submodule

import numpy as np
from copy import copy
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.special import erf
from ...apext import Table, Column


_sqrt2recip = 1/np.sqrt(2)


class RoundGaussian2D(Fittable2DModel):
    """Round 2D Gaussian model

    Model parameters
    ----------------
    amplitude : Amplitude of the Gaussian, A
    sigma : Standard deviation of the Gaussian, S
    x0, y0 : Center position
    background : Background level, C

    Model formula
    -------------
        G(x, y) = A * exp(- 0.5 * ((x-x0)**2 + (y-y0)**2) / S**2 ) + C
    """
    amplitude = Parameter(default=1., min=0.)
    sigma = Parameter(default=1., min=0.)
    x0 = Parameter(default=0.)
    y0 = Parameter(default=0.)
    background = Parameter(default=0.)

    @staticmethod
    def evaluate(x, y, amplitude, sigma, x0, y0, background):
        xx = x - x0
        yy = y - y0
        zx = np.exp(-0.5* (xx/sigma)**2)
        zy = np.exp(-0.5* (yy/sigma)**2)
        return amplitude * zx * zy + background

    @property
    def flux(self):
        return 2*np.pi*self.sigma**2*self.amplitude


# Allowing smearing in arbitrary orientation
class SmearedGaussian2D(Fittable2DModel):
    """
    Round 2D Gaussian with a 1-D linear smearing

    Model parameters
    ----------------
    amplitude : Amplitude of the Gaussian, A
    sigma : Standard deviation of the Gaussian, S
    smear : Smearing length, M
    x0, y0 : Center position
    position_angle : Position angle (deg) of the smearing direction, measured
            ccw from up, T
    background : Background level, C

    Model formula
    -------------
        G(x, y) = A * exp(-0.5 * (x' / S)**2) * Y + C
        Y(y) = (erf((M/2 - y') / (sqrt(2)*S)) + erf((M/2 + y') / (sqrt(2)*S)))
                    / norm
        norm = 2 * erf(M / (2 * S * sqrt(2)))
        x' = dx * cos(A) + dy * sin(A)
        y' = -dx * sin(A) + dy * cos(A)
        dx = x - x0
        dy = y - y0
    """
    amplitude = Parameter(default=1., min=0.)
    sigma = Parameter(default=1., min=0.)
    smear = Parameter(default=0., min=0.)
    x0 = Parameter(default=0.)
    y0 = Parameter(default=0.)
    position_angle = Parameter(default=0., min=0., max=360.)
    background = Parameter(default=0.)

    @staticmethod
    def evaluate(x, y, amplitude, sigma, smear, x0, y0, angle, background):
        # formula form verified on 11/14/2019
        dx = x - x0
        dy = y - y0
        angle1 = angle+np.pi/2
        xx = dx*np.cos(angle1) + dy*np.sin(angle1)
        yy = -dx*np.sin(angle1) + dy*np.cos(angle1)
        if smear == 0:
            zx = np.exp(-0.5*(xx/sigma)**2)
        else:
            d = smear / 2
            norm = 2 * erf(d/sigma*_sqrt2recip)
            zx = (erf((d-xx)/sigma*_sqrt2recip)
                    + erf((d+xx)/sigma*_sqrt2recip))/norm
        zy = np.exp(-0.5* (yy/sigma)**2)
        return amplitude * zx * zy + background

    @property
    def flux(self):
        # flux derived and verified on 11/14/2019
        if self.smear == 0:
            return self.amplitude * 2 * np.pi * self.sigma**2
        else:
            return self.amplitude * np.sqrt(2*np.pi) * self.sigma \
                * self.smear / erf(0.5 * self.smear / self.sigma * _sqrt2recip)



class PSFPhot():
    """Class to perform PSF photometry for given locations
    """

    def __init__(self, image, locations, box=11, fitter=None):
        self.image = copy(image)
        self.locations = copy(locations)
        self.box = box
        if fitter is None:
            fitter = LevMarLSQFitter()
        self.fitter = fitter
        self.nloc = len(self.locations)

    def __call__(self, m0):
        """
        model : astropy.modelig.Model instance
        """
        if self.image is None:
            raise ValueError('No image is specified.')
        if self.locations is None:
            raise ValueError('No locations are specified.')

        width = self.box//2
        m0.x0 = width
        m0.y0 = width

        sz = self.image.shape
        subims = np.zeros(self.nloc, dtype=object)
        models = np.zeros(self.nloc, dtype=object)
        regions = np.zeros((self.nloc, 4))
        flux = np.zeros(self.nloc)
        ct = np.zeros((self.nloc, 2))
        for i, loc in enumerate(self.locations):
            # extract sub-image
            yc, xc = [int(round(x)) for x in loc]
            x1 = np.clip(xc - width, 0, sz[0])
            x2 = np.clip(xc + width + 1, 0, sz[0])
            y1 = np.clip(yc - width, 0, sz[1])
            y2 = np.clip(yc + width + 1, 0, sz[1])
            subim = self.image[y1:y2,x1:x2].copy()
            regions[i] = np.array([x1, y1, x2, y2])
            subims[i] = subim
            subsz = subim.shape
            xx, yy = np.meshgrid(range(subsz[0]), range(subsz[1]))

            # fit PSF to sub-image
            m0.amplitude = subim.max()
            m = self.fitter(m0, xx, yy, subim)
            xc = xc - width + m.x0
            yc = yc - width + m.y0

            models[i] = m
            flux[i] = m.flux
            ct[i] = np.array([xc, yc])

        parms = [m.parameters for m in models]
        parm_tbl = Table(np.array(parms).T.tolist(),
                names=models[0].param_names)
        parm_tbl.add_column(Column(flux, name='flux'))
        self.phot = parm_tbl
        self.sub_images = subims
        return parm_tbl
