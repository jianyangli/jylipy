# PSF photometry submodule

import numpy as np
from copy import copy
from astropy.modeling import Fittable2DModel, Parameter, FittableModel
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.special import erf
import astropy.units as u
from ...apext import Table, Column
from ...core import ascii_read
from sbpy.bib import cite
from sbpy import photometry
from sbpy.calib import solar_fluxd
import sbpy.units as sbu


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

    @staticmethod
    def evaluate(x, y, amplitude, sigma, x0, y0):
        xx = x - x0
        yy = y - y0
        zx = np.exp(-0.5* (xx/sigma)**2)
        zy = np.exp(-0.5* (yy/sigma)**2)
        return amplitude * zx * zy

    @property
    def flux(self):
        return 2*np.pi*self.sigma**2*self.amplitude


class SmearedGaussian2D(Fittable2DModel):
    """
    Round 2D Gaussian with a 1-D linear smearing

    Model parameters
    ----------------
    amplitude : Amplitude of the Gaussian, A
    sigma : Standard deviation of the Gaussian, S
    smear : Smearing length, M
    x0, y0 : Center position
    position_angle : Position angle (rad) of the smearing direction, measured
            ccw from up, T

    Model formula
    -------------
        G(x, y) = A * exp(-0.5 * (x' / S)**2) * Y
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
    position_angle = Parameter(default=0., min=0., max=np.pi)

    @staticmethod
    def evaluate(x, y, amplitude, sigma, smear, x0, y0, angle):
        # formula form verified on 11/14/2019
        dx = x - x0
        dy = y - y0
        angle1 = (angle + np.pi/2) % np.pi
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
        return amplitude * zx * zy

    @property
    def flux(self):
        # flux derived and verified on 11/14/2019
        if self.smear == 0:
            return self.amplitude * 2 * np.pi * self.sigma**2
        else:
            return self.amplitude * np.sqrt(2*np.pi) * self.sigma \
                * self.smear / erf(0.5 * self.smear / self.sigma * _sqrt2recip)

    def BGFree(self):
        """Return a background-free version of the model
        """
        return SmearedGaussian2D(self.amplitude, self.sigma, self.smear,
                self.x0, self.y0, self.position_angle)


class SmearedGaussian2D_ConstantBG(SmearedGaussian2D):
    """
    Round 2D Gaussian with a 1-D linear smearing with a constant background

    Model parameters
    ----------------
    amplitude : Amplitude of the Gaussian, A
    sigma : Standard deviation of the Gaussian, S
    smear : Smearing length, M
    x0, y0 : Center position
    position_angle : Position angle (rad) of the smearing direction, measured
            ccw from up, T
    background : Background level, C

    Model formula
    -------------
        G(x, y) = G0(x, y) + C
        where G0(x, y) is background-free SmearedGaussian2D model
    """
    amplitude = Parameter(default=1., min=0.)
    sigma = Parameter(default=1., min=0.)
    smear = Parameter(default=0., min=0.)
    x0 = Parameter(default=0.)
    y0 = Parameter(default=0.)
    position_angle = Parameter(default=0., min=0., max=np.pi)
    background = Parameter(default=0.)

    @staticmethod
    def evaluate(x, y, amplitude, sigma, smear, x0, y0, angle, background):
        return SmearedGaussian2D.evaluate(x, y, amplitude, sigma, smear, x0,
                y0, angle) + background


class SmearedGaussian2D_LinearBG(SmearedGaussian2D):
    """
    Round 2D Gaussian with a 1-D linear smearing with a constant background

    Model parameters
    ----------------
    amplitude : Amplitude of the Gaussian, A
    sigma : Standard deviation of the Gaussian, S
    smear : Smearing length, M
    x0, y0 : Center position
    position_angle : Position angle (rad) of the smearing direction, measured
            ccw from up, T
    a : Background slope along x
    b : Background slope along y
    c : Background constant c

    Model formula
    -------------
        G(x, y) = G0(x, y) + BG(x, y)
        BG(x, y) = a*x + b*y + c
        where G0(x, y) is background-free SmearedGaussian2D model
    """
    amplitude = Parameter(default=1., min=0.)
    sigma = Parameter(default=1., min=0.)
    smear = Parameter(default=0., min=0.)
    x0 = Parameter(default=0.)
    y0 = Parameter(default=0.)
    position_angle = Parameter(default=0., min=0., max=np.pi)
    a = Parameter(default=0.)
    b = Parameter(default=0.)
    c = Parameter(default=0.)

    @staticmethod
    def evaluate(x, y, amplitude, sigma, smear, x0, y0, angle, a, b, c):
        return SmearedGaussian2D.evaluate(x, y, amplitude, sigma, smear, x0,
                y0, angle) + a*x + b*y + c


class PSFPhot():
    """Class to perform PSF photometry for given locations
    """

    def __init__(self, image, locations, box=11, fitter=None, mask=None):
        self.image = copy(image)
        self.locations = copy(locations)
        self.box = box
        if fitter is None:
            fitter = LevMarLSQFitter()
        self.fitter = fitter
        self.nloc = len(self.locations)
        self.mask = mask

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
        mod_full = np.zeros_like(self.image)
        xx0, yy0 = np.meshgrid(range(sz[1]), range(sz[0]))

        subims = np.zeros(self.nloc, dtype=object)  # sub images
        models = np.zeros(self.nloc, dtype=object)  # model objects
        modims = np.zeros(self.nloc, dtype=object)  # model images
        resims = np.zeros(self.nloc, dtype=object)  # residual images
        regions = np.zeros((self.nloc, 4))
        flux = np.zeros(self.nloc)
        pos = np.zeros((self.nloc, 2))  # position in original image
        for i, loc in enumerate(self.locations):
            # extract sub-image
            yc, xc = [int(round(x)) for x in loc]
            x1 = np.clip(xc - width, 0, sz[1])
            x2 = np.clip(xc + width + 1, 0, sz[1])
            y1 = np.clip(yc - width, 0, sz[0])
            y2 = np.clip(yc + width + 1, 0, sz[0])
            subim = self.image[y1:y2,x1:x2].copy()
            regions[i] = np.array([x1, y1, x2, y2])
            subims[i] = subim
            subsz = subim.shape
            xx, yy = np.meshgrid(range(subsz[1]), range(subsz[0]))

            # fit PSF to sub-image
            m0.amplitude = subim.max()
            if self.mask is not None:
                gdpix = ~self.mask[y1:y2,x1:x2]
                xx = xx[gdpix]
                yy = yy[gdpix]
                subim = subim[gdpix]
            m = self.fitter(m0, xx, yy, subim)

            # position in original image
            xc = xc - width + m.x0
            yc = yc - width + m.y0
            pos[i] = np.array([xc, yc])

            # record model results
            models[i] = m
            flux[i] = m.flux
            if self.mask is None:
                modims[i] = m.BGFree()(xx, yy)
                resims[i] = subim - modims[i]
            else:
                modims[i] = np.zeros(subsz)
                modims[i][gdpix] = m.BGFree()(xx, yy)
                resims[i] = np.zeros(subsz)
                resims[i][gdpix] = subim - modims[i][gdpix]

            # calculate full frame model
            m1 = m.BGFree()
            m1.x0, m1.y0 = pos[i]
            mod_full += m1(xx0, yy0)

        parms = [m.parameters for m in models]
        parm_tbl = Table(np.array(parms).T.tolist(),
                names=models[0].param_names)
        parm_tbl.add_column(Column(flux, name='flux'))
        self.phot = parm_tbl
        self.subims = subims
        self.models = models
        self.submod = modims
        self.subres = resims
        self.spos = pos
        self.regions = regions
        self.residual = self.image - mod_full

        return parm_tbl


class Geometry():
    """Observing geometry class"""
    def __init__(self, rh, delta, phase):
        """
        rh: heliocentric distance, in au or Quantity
        delta: observer distance, in km or Quantity
        phase: phase angle, in deg or Quantity
        """
        if not isinstance(rh, u.Quantity):
            rh = rh * u.au
        if not isinstance(delta, u.Quantity):
            delta = delta * u.km
        if not isinstance(phase, u.Quantity):
            phase = phase * u.deg
        self.rh = rh
        self.delta = delta
        self.phase = phase


class BennuPhaseFunc():
    """Bennu phase function class

    Default is the v-band phase function derived from approach data as
    published in Hergenrother et al. (2019)
    """
    @cite({'Default Bennu V-band phase function': '2019NatCo..10.1291H',
           'Bennu radius': '2019Natur.568...55L'})
    def __init__(self, model=None):
        """
        model : str
            File name, ASCII file store the phase function model
        """
        if model is None:
            self.model = '/Users/jyli/Work/OSIRIS-REx/Publications/201903_Nature/AWG/model_phasefunc.txt'
        else:
            self.model = model

        # equivalent radius of Bennu in km (Lauretta et al. 2019)
        r_bennu = 0.24503 * u.km

        phase_model = ascii_read(self.model)
        with solar_fluxd.set({'V': -26.77 * u.mag}):
            iof = (phase_model['HG12'] * u.mag).to('1/sr', sbu.reflectance('V',
                    cross_section=np.pi * r_bennu**2)).value * np.pi
        from scipy.interpolate import interp1d
        self.func = interp1d(phase_model['phase'], iof)

    def __call__(self, phase):
        return self.func(phase)


class Dust():
    """Dust class"""
    def __init__(self, radius=None, phasefunc=None):
        """
        radius: the radius of dust
        phasefunc: the phase function of dust, where the <I/F> at phase angle
            a is `phasefunc(a)`
        """
        self.radius = radius
        self.phasefunc = phasefunc

    @property
    def diameter(self):
        return 2 * self.radius

    @classmethod
    def from_counts(cls, counts, geom, phasefunc=BennuPhaseFunc()):
        """Calculate dust radius from count rate

        counts: The count rate in DN/s
        geom: Geometry class object, observing geometry for `counts`
        phasefunc: The phase function of dust, where the <I/F> at phase angle
            a is `phasefunc(a)`
        """
        pixscl = 2.8e-4  # NAVCAM1 pixel scale in rad number from Hergenrother
        iofcal = 2.12e-8   # I/F calibration constant at 1 au
        iof = counts * iofcal * geom.rh.to('au').value**2
        fill_fac = iof / phasefunc(geom.phase.to('deg').value)
        radius = pixscl * geom.delta * np.sqrt(fill_fac / np.pi)
        return cls(radius=radius, phasefunc=phasefunc)


class OpenCVDistortion(FittableModel):
    """Open CV distortion model

    Model inputs
    ------------
    (x, y) : numbers or iterables of numbers
        The coordinate of scene in the undistorted frame, in angular units
        radians.

    Model outputs
    -------------
    (x1, y1) : numbers or iterables of numbers
        The pixel coordinates in the distorted frame, in pixels.

    Model parameters
    ----------------
    k1, k2, k3 : numbers
        Radial distortion parameters, dimensionless
    p1, p2 : numbers
        Tangential distortion parameters, dimensionless
    fx, fy : numbers
        Focal lengths in x and y direction, in pixels
    cx, cy : numbers
        Pixel coordinates of origin, or boresight on the detector, in pixels

    Model description
    -----------------
    x_radial = x * (1 + k1*r**2 + k2*r**4 + k3*r**6)
    y_radial = y * (1 + k1*r**2 + k2*r**4 + k3*r**6)
    x_tang = x + 2*p1*x*y + p2*(r**2 + 2*x**2)
    y_tang = y + p1*(r**2 + 2*y**2) + 2*p2*x*y
    x1 = (x_radial + x_tang) * fx + cx
    y1 = (y_radial + y_rang) * fy + cy

    where (x, y) are inputs, (x1, y1) are outputs, r**2 = x**2 + y**2.
    """
    inputs = ('x', 'y')
    outputs = ('x1', 'y1')

    k1 = Parameter()
    k2 = Parameter()
    k3 = Parameter()
    p1 = Parameter()
    p2 = Parameter()
    fx = Parameter()
    fy = Parameter()
    cx = Parameter()
    cy = Parameter()

    @staticmethod
    def evaluate(x, y, k1, k2, k3, p1, p2, fx, fy, cx, cy):
        x2 = x*x
        y2 = y*y
        xy = x*y
        r2 = x2 + y2
        r4 = r2 * r2
        radial = 1 + k1*r2 + k2*r4 + k3*r2*r4
        x1 = radial*x + 2*p1*xy + p2*(r2 + 2*x2)
        y1 = radial*y + p1*(r2+2*y2) + 2*p2*xy
        x1 = x1*fx + cx
        y1 = y1*fy + cy
        return x1, y1

    def ifov(self, x, y):
        """Calculate pixel scale iFOV.

        Parameters
        ----------
        (x, y) : numbers or interables of numbers
            The coordinate of scene in the undistorted frame, in angular units
        radians.

        Returns
        -------
        (dx/dx1, dy/dy1) : numpy arrays
            The ifov along x and y direction in unit of radian/pixel.
        """
        x2 = x*x
        y2 = y*y
        xy = x*y
        r2 = x2 + y2
        r4 = r2 * r2
        radial = 1 + self.k1*r2 + self.k2*r4 + self.k3*r2*r4
        radial1 = 2*(self.k1 + 2*self.k2*r2 + 3*self.k3*r4)
        tang = 2*(self.p1*y + self.p2*x)
        sumterm = radial + tang
        dx = (sumterm + x2*radial1 + 4*self.p2*x) * self.fx
        dy = (sumterm + y2*radial1 + 4*self.p1*y) * self.fy
        return 1/dx, 1/dy

