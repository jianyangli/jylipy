# PSF photometry submodule

import numpy as np
from copy import copy
from astropy.modeling import Fittable2DModel, Parameter, FittableModel
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.special import erf
import astropy.units as u
from ...apext import Table, Column, table
from ...core import ascii_read, findfile, readfits
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


class PSFSource():
    """PSF source class

    Attributes
    ----------
    ID : whatever
        The ID of the source
    center : (y, x), numbers
        The center pixel coordinate
    model : `astropy.modeling.Model` instance
        The model describe the source
    flux : number
        The total flux of source
    image : 2D array
        The image of the source
    """
    def __init__(self, center, image, mask=None, ID=None, model=None,
                flux=None, meta=None):
        self.center = center
        self.image = image
        self._mask = mask
        self.ID = ID
        self.model = model
        self._flux = flux
        self.meta = None

    @property
    def mask(self):
        if self._mask is None:
            return np.zeros_like(self.image).astype(bool)
        else:
            return self._mask

    @mask.setter
    def mask(self, msk):
        self._mask = msk

    def model_image(self, nobg=True):
        """Return model image"""
        self._check_model()
        if np.allclose(self.model.parameters, 0):
            return np.zeros_like(self.image)
        else:
            imsz = self.image.shape
            xx0, yy0 = np.meshgrid(range(imsz[1]), range(imsz[0]))
            if nobg and hasattr(self.model, 'BGFree'):
                return self.model.BGFree()(xx0, yy0)
            else:
                return self.model(xx0, yy0)

    def residual(self, nobg=True):
        """Return residual image"""
        self._check_model()
        res = self.image - self.model_image(nobg=nobg)
        if np.any(self.mask):
            res[self.mask] = 0
        return res

    @property
    def flux(self):
        if (self.model is None) or np.allclose(self.model.parameters, 0):
            return self._flux
        else:
            if not hasattr(self.model, 'flux'):
                raise AttributeError('`flux` attribute is not available in'
                    ' model.')
            return self.model.flux

    def _check_model(self):
        if self.model is None:
            raise ValueError('Model not specified for fitting.')

    def fit(self, fitter=None):
        """Fit the source to a PSF model
        """
        self._check_model()

        if fitter is None:
            fitter = LevMarLSQFitter()

        imsz = self.image.shape
        xx0, yy0 = np.meshgrid(range(imsz[1]), range(imsz[0]))
        if np.all(self.mask):
            self.model.parameters[:] = 0.
        else:
            xx = xx0[~self.mask].astype('float32')
            yy = yy0[~self.mask].astype('float32')
            data = self.image[~self.mask].astype('float32')
            self.model.amplitude = data.max()
            self.model = fitter(self.model, xx, yy, data)
            self.center = self.model.y0, self.model.x0


class PSFSourceGroup():
    """A group of PSF sources that are from the same source image.

    Attributes
    ----------
    image_file : str
        The name of image file
    image : 2d array
        The image
    catalog : `astropy.table.Table` instance
        The catalog of PSF sources.  Columns are:
        cx, cy : center pixel coordinates
        bx, by : box size in pixels
        flux : total flux
        model : str, name of model
    """
    def __init__(self, image, catalog, bbox=None, mask=None):
        """
        Parameters
        ----------
        image : 2d array
            The image that contains all the sources
        catalog : `astropy.table.Table` instance
            The catalog of PSF sources.  It must contain at least two columns
            `cx` and `cy` listing the center coordinates of sources.  Other
            useful columns are:
            'ID' : str, the ID of each source
            'flux' : number, the flux of each source
            More columns will be simply copied over to `.meta` attribute of
            each particle object.
        bbox : (by, bx), number
            The default box size in pixels for all sources, if `catalog` table
            does not contain `bx` and `by` columns.
        mask : 2d bool array
            Image mask
        """
        self.image = image
        self.catalog = catalog.copy()
        self.bbox = bbox
        self.mask = mask
        if ('cx' not in catalog.keys()) or ('cy' not in catalog.keys()):
            raise ValueError('`catalog` must contain columns `cx` and `cy`.')
        self._populate_sources()

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, *args, **kwargs):
        return self.sources.__getitem__(*args, **kwargs)

    def _region(self, i):
        """Return the region slice for the ith source"""
        cat_keys = self.catalog.keys()
        if 'by' in cat_keys:
            by = self.catalog['by'][i]
        else:
            if self.bbox is None:
                raise ValueError('`bbox` not specified.')
            by = self.bbox[0]
        if 'bx' in cat_keys:
            bx = self.catalog['bx'][i]
        else:
            if self.bbox is None:
                raise ValueError('`bbox` not specified.')
            bx = self.bbox[1]
        cy, cx = self.catalog['cy', 'cx'][i]
        imsz = self.image.shape
        x1 = int(round(np.clip(cx-bx//2, 0, imsz[1])))
        x2 = int(round(np.clip(cx+bx//2+1, 0, imsz[1])))
        y1 = int(round(np.clip(cy-by//2, 0, imsz[0])))
        y2 = int(round(np.clip(cy+by//2+1, 0, imsz[0])))
        return slice(y1, y2), slice(x1, x2)

    def _extract_subim(self, i, image=None, mask=False):
        """Extract and return the image snippet that contains the ith source

        i : int
            The index of source
        image : 2d image, optional
            The image to extract the sub-image for source.  Default is
            `self.image`.
        mask : bool, optional
            If `True`, then the mask of sub-image will be returned
        """
        if image is None:
            image = self.image
        region = self._region(i)
        if mask:
            if self.mask is None:
                ys = region[0].stop - region[0].start
                xs = region[1].stop - region[1].start
                return np.zeros((ys, xs), dtype=bool)
            else:
                return self.mask[region].copy()
        else:
            return image[region].copy()

    def _populate_sources(self):
        """Populate all the source objects
        """
        cat_keys = self.catalog.keys()
        self.sources = []
        if 'flux' in cat_keys:
            has_flux = True
        else:
            has_flux = False
        if 'ID' in cat_keys:
            has_id = True
        else:
            has_id = False
        for i, row in enumerate(self.catalog):
            center = row['cy', 'cx']
            subim = self._extract_subim(i).copy()
            submsk = self._extract_subim(i, mask=True).copy()
            if has_flux:
                flux = row['flux']
            else:
                flux = None
            if has_id:
                ID = row['ID']
            else:
                ID = None
            self.sources.append(PSFSource(row['cy','cx'], subim, mask=submsk,
                    ID=ID, flux=flux, meta=row))

    def _update_catalog(self):
        """Update catalog with model parameters, including the columns
        `cx`, `cy`, and `flux`"""
        if 'flux' not in self.catalog.keys():
            self.catalog.add_column(Column(np.repeat(0., len(self)),
                    name='flux'))
        for i,s in enumerate(self.sources):
            region = self._region(i)
            self.catalog[i]['cx'] = s.model.x0.value + region[1].start
            self.catalog[i]['cy'] = s.model.y0.value + region[0].start
            self.catalog[i]['flux'] = s.flux

    def fit(self, fitter=None, model=None, niter=1):
        """Fit PSF to all sources
        """
        for j in range(niter):
            fluxes = [x.flux for x in self.sources]
            ordered = np.argsort(fluxes)[::-1]
            residual = self.image.copy()
            model_image = np.zeros_like(self.image)
            for i in ordered:
                self.sources[i].image = self._extract_subim(i, residual)
                self.sources[i].mask = self._extract_subim(i, mask=True)
                if model is not None:
                    self.sources[i].model = model
                self.sources[i].fit(fitter=fitter)
                submodel = self.sources[i].model_image()
                region = self._region(i)
                residual[region] -= submodel
                model_image[region] += submodel
            self.residual = residual
            self.model = model_image
        self._update_catalog()

    def mark_source(self, ds9, radius=3, color=None):
        """Mark the source location in DS9
        """
        from ...saoimage import CircularRegion
        print('showing sources')
        for x, y in self.catalog['cx','cy']:
            r = CircularRegion(x,y,radius,color=color)
            r.show(ds9)

    def display(self, image=True, model=True, residual=True, ds9=None,
            source=True, **kwargs):
        """Display images, model, and residual

        image, model, residual : bool
            Switch to display the original image, the model, and the residual
        ds9 : `saoimage.DS9`
            The DS9 to display images.  If `None`, then images will be
            displayed in matplotlib plot
        source : bool
            Show source locations
        radius : number
            The radius of circles to mark source locations in DS9
        color : str
            The color of circles to mark source locations in DS9
        **kwargs : other keywords accepted by `matplotlib.pyplot.figure`.
        """
        radius = kwargs.pop('radius', 3)
        color = kwargs.pop('color', 'green')
        if ds9 is None:
            import matplotlib.pyplot as plt
            plt.figure(**kwargs)
            if image:
                plt.imshow(self.image)
            if model and hasattr(self, 'model'):
                plt.imshow(self.model)
            if residual and hasattr(self, 'residual'):
                plt.imshow(self.residual)
        else:
            if image:
                ds9.imdisp(self.image)
                if source:
                    self.mark_source(ds9, radius=radius, color=color)
            if model and hasattr(self, 'model'):
                ds9.imdisp(self.model)
                if source:
                    self.mark_source(ds9, radius=radius, color=color)
            if residual and hasattr(self, 'residual'):
                ds9.imdisp(self.residual)
                if source:
                    self.mark_source(ds9, radius=radius, color=color)


class PSFPhot():
    """Class to perform PSF photometry for given locations
    """

    def __init__(self, catalog, datadir='', box=11, fitter=None, mask=None):
        """
        Parameters
        ----------
        image : 2d array
            Image to be processed
        catalog : astropy.table.Table instance
            The catalog of sources.  It has to contain at least three columns,
            `x0`, `y0`: the approximate centroid (x, y) of the source.
            `Image` : str, the unique ID that can be used to identify image of
                the source.
            If `catalog` contains a column `flux`, then this column will be
            used to sort the source from the brightest to the faintest for PSF
            fitting.
        box : number
            The box size within which PSF fitting is performed
        fitter : astropy.modeing.fitting.Fitter class object
            Fitter used to perform PSF fitting.  Default is
            `astropy.modeling.fitting.LevMarLSQFitter`
        mask : 2d array of bool
            Image mask with True to mask out bad pixels that will not be used
            to fit PSF.
        """
        self.catalog = catalog.copy()
        self.box = box
        if fitter is None:
            fitter = LevMarLSQFitter()
        self.fitter = fitter
        self.len = len(self.catalog)
        self.mask = mask
        self._filelist(datadir)

    def _filelist(self, datadir):
        """Compile a list of file contained in the data directory, with a
        uniform indexing that does not include level and version number.
        """
        from os.path import basename
        self._file_list = findfile(datadir, '.fits', recursive=True)
        self._file_id = ['_'.join(basename(x).split('.')[0].split('_')[:2])
                for x in self._file_list]

    def _load_image(self, img_id):
        """Return the image based on the input id
        """
        try:
            match = self._file_id.index('_'.join(img_id.split('_')[:2]))
        except ValueError:
            return None
        return readfits(self._file_list[match],
                verbose=False).astype('float32')

    def psffit(self, m0, cat, image, mask=None, flux0=None, niter=1):
        """Fit PSF model to all sources listed in a catalog in an image

        Parameters
        ----------
        m0 : astropy.modelig.Model instance
            The PSF model to be fitted.  The model has to have three
            parameters: `xc`, `yc` for the center position, and `amplitude` as
            an overall scaling factor.
        cat : astropy.table.Table instance
            The catalog of sources in the same image.  See `__init__`.
        image : 2D array
            Image that contains all sources.
        flux0 : array
            The initial estimate of fluxes for sources, used to sort the PSF
            fitting process from the brightest source to the faintest source.
        """
        ordered = None
        if flux0 is not None:
            ordered = np.asanyarray(flux0).argsort()
            if 'flux' not in cat.keys():
                cat.add_column(Column(flux0, name='flux'))
        elif 'flux' in cat.keys():
            ordered = cat.argsort('flux')
        if ordered is not None:
            cat = cat[ordered[::-1]]

        cat_len = len(cat)

        for n in range(niter):

            if 'psf_flux' in cat.keys():
                cat.sort('psf_flux')
                cat.reverse()

            width = self.box//2
            m0.x0 = width
            m0.y0 = width

            sz = image.shape
            mod_full = np.zeros_like(image)
            xx0, yy0 = np.meshgrid(range(sz[1]), range(sz[0]))

            subims = np.zeros(cat_len, dtype=object)  # sub images
            models = np.zeros(cat_len, dtype=object)  # model objects
            modims = np.zeros(cat_len, dtype=object)  # model images
            resims = np.zeros(cat_len, dtype=object)  # residual images
            regions = np.zeros((cat_len, 4))
            flux = np.zeros(cat_len)
            pos = np.zeros((cat_len, 2))  # position in original image
            residual = image.copy()
            for i, loc in enumerate(cat['xc', 'yc']):
                # extract sub-image
                xc, yc = [int(round(x)) for x in loc]
                x1 = np.clip(xc - width, 0, sz[1])
                x2 = np.clip(xc + width + 1, 0, sz[1])
                y1 = np.clip(yc - width, 0, sz[0])
                y2 = np.clip(yc + width + 1, 0, sz[0])
                subim = residual[y1:y2,x1:x2].copy()
                regions[i] = np.array([x1, y1, x2, y2])
                subims[i] = subim

                # fit PSF to sub-image
                subsz = subim.shape
                xx, yy = np.meshgrid(range(subsz[1]), range(subsz[0]))
                m0.amplitude = subim.max()
                if mask is not None:
                    gdpix = ~mask[y1:y2,x1:x2]
                    xx = xx[gdpix]
                    yy = yy[gdpix]
                    subim = subim[gdpix]
                if len(xx) < 9:
                    m = m0.copy()
                    m.amplitude = -999.
                else:
                    m = self.fitter(m0, xx, yy, subim)

                # position in original image
                xc = xc - width + m.x0
                yc = yc - width + m.y0
                pos[i] = np.array([xc, yc])

                # record model results
                models[i] = m
                flux[i] = m.flux
                if mask is None:
                    modims[i] = m.BGFree()(xx, yy)
                    resims[i] = residual[y1:y2,x1:x2].copy() - modims[i]
                else:
                    modims[i] = np.zeros(subsz)
                    modims[i][gdpix] = m.BGFree()(xx, yy)
                    resims[i] = np.zeros(subsz)
                    resims[i][gdpix] = residual[y1:y2,x1:x2].copy()[gdpix] \
                            - modims[i][gdpix]

                # calculate full frame model
                m1 = m.BGFree()
                m1.x0, m1.y0 = pos[i]
                residual -= m1(xx0, yy0)

            if 'psf_flux' in cat.keys():
                cat['psf_flux'] = flux
            else:
                cat.add_column(Column(flux, name='psf_flux'))

        parms = [m.parameters for m in models]
        parm_tbl = Table(np.array(parms).T.tolist(),
                names=models[0].param_names)
        parm_tbl['x0'] = pos[:,0]
        parm_tbl['y0'] = pos[:,1]
        parm_tbl.add_column(Column(flux, name='flux'))

        return cat, models, parm_tbl, regions, subims, modims, resims, residual

    def __call__(self, m0, flux0=None, niter=1):
        """
        m0 : astropy.modelig.Model instance
            The PSF model to be fitted.  The model has to have three
            parameters: `xc`, `yc` for the center position, and `amplitude` as
            an overall scaling factor.
        flux0 : array
            The initial estimate of fluxes for sources, used to sort the PSF
            fitting process from the brightest source to the faintest source
        niter : number
            Number of iteration
        """
        self.catalog.sort(['Image', 'ID'])
        imgs = np.unique(self.catalog['Image'])
        cats = []
        self.phot = []
        self.subims = []
        self.models = []
        self.submod = []
        self.subres = []
        self.regions = []
        self.residual = []
        self.image = []
        self.imname = []

        for f in imgs:
            im = self._load_image(f)
            if im is None:
                continue
            index = self.catalog['Image'] == f
            cat = self.catalog[index].copy()
            if flux0 is not None:
                flx = flux0[index]
            else:
                flx = None
            cat, models, parm_tbl, regions, subims, modims, resims, residual \
                    =  self.psffit(m0, cat, im, mask=im>4094, flux0=flx, \
                        niter=niter)
            cats.append(cat)
            self.phot.append(parm_tbl)
            self.subims.append(subims)
            self.models.append(models)
            self.submod.append(modims)
            self.subres.append(resims)
            self.regions.append(regions)
            self.residual.append(residual)
            self.imname.append(f)
            self.image.append(im)

        # post-processing
        self.catalog = table.vstack(cats)
        self.phot = table.vstack(self.phot)
        self.subims = np.concatenate(self.subims)
        self.models = np.concatenate(self.models)
        self.submod = np.concatenate(self.submod)
        self.subres = np.concatenate(self.subres)
        self.regions = np.concatenate(self.regions)

        sorting = self.catalog.argsort('ID')
        self.catalog = self.catalog[sorting]
        self.phot = self.phot[sorting]
        self.subims = self.subims[sorting]
        self.models = self.models[sorting]
        self.submod = self.submod[sorting]
        self.subres = self.subres[sorting]
        self.regions = self.regions[sorting]


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


class PSF_Corr():
    """Correction to PSF photometry
    """
    def __init__(self, corr, xs=2752, ys=2004, kind='cubic'):
        """
        corr : 2D array
            Correction factor array, assuming uniformly distributed across the
            FOV.
        xs, ys : int
            The size of image in x and y directions
        """
        self.corr = corr
        self.shape = xs, ys
        self.kind = kind
        sz = corr.shape
        xx, yy = np.linspace(0, xs, sz[1]*2+1)[1::2], \
                 np.linspace(0, ys, sz[0]*2+1)[1::2]
        from scipy.interpolate import interp2d
        self.func = interp2d(xx, yy, corr, kind=kind)

    def __call__(self, x, y, grid=False):
        out = self.func(x, y)
        if (not grid) and out.ndim > 1:
            out = out.diagonal()
        return out
