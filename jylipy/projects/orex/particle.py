# PSF photometry submodule

import numpy as np
from copy import copy
from astropy.modeling import Fittable2DModel, Parameter, FittableModel
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.special import erf
import astropy.units as u
from ...apext import Table, Column, table
from ...core import ascii_read, findfile, readfits
from ...saoimage import DS9
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
        ID of the source
    center : (y, x), numbers
        Center pixel coordinate
    model : `astropy.modeling.Model`
        Model that describes the source
    flux : number
        Total flux of source
    image : 2D array
        Image of the source
    mask : 2D bool array of the same size as `image`
        Image mask
    corr : number
        Correction factor for actual PSF
    """
    def __init__(self, image, mask=None, ID=None, model=None,
                flux=None, meta=None, corr=None):
        self.image = image
        self._mask = mask
        self.ID = ID
        self.model = model
        self._flux = flux
        self.meta = meta
        self.corr = corr

    @property
    def mask(self):
        """Mask of image"""
        if self._mask is None:
            return np.zeros_like(self.image).astype(bool)
        else:
            return self._mask

    @mask.setter
    def mask(self, msk):
        self._mask = msk

    @property
    def center(self):
        if hasattr(self, 'model'):
            return self.model.y0.value, self.model.x0.value
        else:
            return self.image.shape[0]/2, self.image.shape[1]/2

    def model_image(self, nobg=True):
        """Return model image

        Parameters
        ----------
        nobg : bool, optional
            If true, then the returned model image will contain no background.
            This parameter requires that the `.model` class has a method
            `.BGFree()` that returns the background free model.  Otherwise
            returns the model image with modeled background.
        """
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
        """Return residual image.  See `.model_image` for parameter `nobg`."""
        self._check_model()
        res = self.image - self.model_image(nobg=nobg)
        if np.any(self.mask):
            res[self.mask] = 0
        return res

    @property
    def flux(self):
        """Total flux of source"""
        if (self.model is None) or np.allclose(self.model.parameters, 0):
            return self._flux
        else:
            if not hasattr(self.model, 'flux'):
                raise AttributeError('`flux` attribute is not available in'
                    ' model.')
            if self.corr is None:
                return self.model.flux
            else:
                return self.model.flux * self.corr

    @flux.setter
    def flux(self, v):
        self._flux = v

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

    def display(self, ds9=None, num=None, subplots_kw={}, imshow_kw={},
                nobg=True):
        """Display the image, model, and residual.

        Parameter
        ---------
        ds9 : `...saoimage.DS9`, optional
            DS9 window for display.  If not set, then use `matplotlib` for
            display.
        num : int or str, optional
            A `.pyplot.figure` keyword that sets the figure number or label.
        subplots_kw : dict, optional
            Dict with keywords passed to the `~matplotlib.pyplot.subplots` call
            to create axes.
        imshow_kw : dict, optional
            Dict with keywords passed to the `~matplotlib.pyplot.imshow` call.

        Return
        ------
        fig : `~matplotlib.pyplot.figure.Figure` if matplotlib display is used.
        """
        if isinstance(ds9, DS9):
            ds9.imdisp(self.image)
            ds9.imdisp(self.model_image(nobg=nobg))
            ds9.imdisp(self.residual(nobg=nobg))
        else:
            from matplotlib import pyplot as plt
            _ = subplots_kw.pop('nrows', None)
            _ = subplots_kw.pop('ncols', None)
            _ = subplots_kw.pop('sharex', None)
            _ = subplots_kw.pop('sharey', None)
            if num is not None:
                subplots_kw['num'] = num
            f, ax = plt.subplots(1, 3, sharey=True, **subplots_kw)
            ax[0].imshow(self.image, **imshow_kw)
            ax[0].set_axis_off()
            ax[1].imshow(self.model_image(nobg=nobg), **imshow_kw)
            ax[1].set_axis_off()
            ax[2].imshow(self.residual(nobg=nobg), **imshow_kw)
            ax[2].set_axis_off()
            return f

    @classmethod
    def from_row(cls, image, row, bbox=None, mask=None):
        """Generate `PSFSource` object from a row in the catalog

        Parameter
        ---------
        image : 2D array
            Whole image that contains the source
        row : `astropy.table.row.Row`, record array
            A row in the catalog that contains at least 'cx' and 'cy'.  Other
            useable fields include 'cx_fit', 'cy_fit', 'flux', 'flux_fit',
            'ID', 'bx', 'by'.
        bbox : [by, bx], numbers
            The bounding box size in pixels
        """
        keys = row.dtype.names
        if (('by' not in keys) or ('bx' not in keys)) and bbox is None:
            raise ValueError('bounding box not defined.')
        if 'by' in keys:
            by, bx = row['by', 'bx']
        else:
            by, bx = bbox
        if ('cy_fit' in keys) and ('cx_fit' in keys):
            cy, cx = row['cy_fit', 'cx_fit']
        else:
            cy, cx = row['cy', 'cx']
        if 'flux_fit' in keys:
            flux = row['flux_fit']
        elif 'flux' in keys:
            flux = row['flux']
        else:
            flux = None
        if 'ID' in keys:
            ID = row['ID']
        else:
            ID = None
        imsz = image.shape
        x1 = int(round(np.clip(cx-bx//2, 0, imsz[1])))
        x2 = int(round(np.clip(cx+bx//2+1, 0, imsz[1])))
        y1 = int(round(np.clip(cy-by//2, 0, imsz[0])))
        y2 = int(round(np.clip(cy+by//2+1, 0, imsz[0])))
        subim = image[y1:y2, x1:x2]
        if mask is None:
            submsk = np.zeros((y2-y1, x2-x1), dtype=bool)
        else:
            submsk = mask[y1:y2, x1:x2]
        return cls(subim, mask=submsk, ID=ID, flux=flux, meta=row)


class PSFSourceGroup():
    """A group of PSF sources that are from the same source image.

    Attributes
    ----------
    image : 2d array
        Image
    catalog : `astropy.table.Table` instance
        Catalog of PSF sources.  Columns are:
        cx, cy : center pixel coordinates
        bx, by : bounding box size (pixels) of sources
        flux : total flux
        model : str, name of model
    sources : list of `PSFSource`
        List of sources
    mask : 2d bool array of the same shape as `image`
        Image mask
    model_parm : `astropy.table.Table` or list
        Model parameters.  If the models of all sources are identical, then
        this attribute is a table.  Otherwise it is a list of tables for each
        model types.
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
        keys = catalog.keys()
        if ('cx' not in keys) or ('cy' not in keys):
            raise ValueError('`catalog` must contain columns `cx` and `cy`.')
        if (('bx' not in keys) or ('by' not in keys)) and (bbox is None):
            raise ValueError('`bbox` is not specified anywhere.')
        ns = len(catalog)
        if 'bx' not in keys:
            catalog.add_column(Column(np.repeat(bbox[1], ns), name='bx'))
        if 'by' not in keys:
            catalog.add_column(Column(np.repeat(bbox[0], ns), name='by'))
        self.image = image
        self.mask = mask

        self.catalog = catalog.copy()
        self.sources = [PSFSource.from_row(self.image, row, bbox=bbox,
                                           mask=mask) for row in self.catalog]

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, *args, **kwargs):
        return self.sources.__getitem__(*args, **kwargs)

    def _center(self, i):
        """Return (yc, xc) of the ith source"""
        keys = self.catalog.keys()
        if ('cy_fit' in keys) and ('cx_fit' in keys):
            return self.catalog[i]['cy_fit', 'cx_fit']
        else:
            return self.catalog[i]['cy', 'cx']

    def _region(self, i):
        """Return the region slice for the ith source"""
        by, bx = self.catalog[i]['by', 'bx']
        cy, cx = self._center(i)
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

    def _update_sources(self, i=None):
        """Update sources based on `.catalog`

        If optional parameter `i` is None, then all sources will be updated.
        Otherwise source[i] is updated.
        """
        if not hasattr(self, 'sources'):
            return
        if i is None:
            for i, s in enumerate(self.sources):
                self._update_sources(i=i)
        else:
            s = self.sources[i]
            s.image = self._extract_subim(i).copy()
            s.mask = self._extract_subim(i, mask=True).copy()
            c = self.catalog[i]
            keys = self.catalog.keys()
            if 'flux_fit' in keys:
                s.flux = c['flux_fit']
            elif 'flux' in keys:
                s.flux = c['flux']
            else:
                s.flux = None

    def _update_catalog(self):
        """Update catalog with model parameters, including the columns
        `cx_fit`, `cy_fit`, and `flux_fit`"""
        model_name = []
        flux_fit = np.zeros(len(self))
        cx_fit = np.zeros(len(self))
        cy_fit = np.zeros(len(self))
        for i, s in enumerate(self.sources):
            region = self._region(i)
            ct = s.center
            cx_fit[i] = ct[1] + region[1].start
            cy_fit[i] = ct[0] + region[0].start
            flux_fit[i] = s.flux
            if s.model.name is None:
                model_name.append(s.model.__class__.__name__)
            else:
                model_name.append(s.model.name)
        model_name = Column(model_name, name='model')
        flux_fit = Column(flux_fit, name='flux_fit')
        cx_fit = Column(cx_fit, name='cx_fit')
        cy_fit = Column(cy_fit, name='cy_fit')
        keys = self.catalog.keys()
        if 'model' in keys:
            self.catalog.replace_column('model', model_name)
        else:
            self.catalog.add_column(model_name)
        if 'flux_fit' in keys:
            self.catalog.replace_column('flux_fit', flux_fit)
        else:
            self.catalog.add_column(flux_fit)
        if 'cx_fit' in keys:
            self.catalog.replace_column('cx_fit', cx_fit)
        else:
            self.catalog.add_column(cx_fit)
        if 'cy_fit' in keys:
            self.catalog.replace_column('cy_fit', cy_fit)
        else:
            self.catalog.add_column(cy_fit)

    @property
    def model_parm(self):
        if 'model' not in self.catalog.keys():
            return None
        parm_tbl = []
        for m in np.unique(self.catalog['model']):
            index = self.catalog.index('model', m)[0]
            parm = {}
            for k in self.sources[index[0]].model.param_names:
                parm[k] = [getattr(self.sources[x].model, k).value \
                           for x in index]
            parm = Table(parm)
            if 'ID' in self.catalog.keys():
                parm.add_column(self.catalog[index]['ID'], index=0)
            else:
                parm.add_column(self.catalog[index]['cy'], index=0)
                parm.add_column(self.catalog[index]['cx'], index=0)
            parm_tbl.append(Table(parm))
        if len(parm_tbl) == 1:
            parm_tbl = parm_tbl[0]
        return parm_tbl

    def fit(self, model=None, fitter=None, niter=1):
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
                    self.sources[i].model = model.copy()
                self.sources[i].fit(fitter=fitter)
                submodel = self.sources[i].model_image()
                region = self._region(i)
                residual[region] -= submodel
                model_image[region] += submodel
            self.residual = residual
            self.model = model_image
            self._update_catalog()
            self._update_sources()

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

    Attributes
    ----------
    sgroup : list
        Source groups.  A source group is defined as all the sources that are
        identified in one single image.
    catalog : list of `astropy.table.Table`
        Source catalogs from all source groups.
    fitter : `astropy.modeling.fitting.Fitter`
        Default fitter.
    model_parm : list of list or `astropy.table.Table`
        Model parameters from all source groups.
    """

    def __init__(self, catalog, datadir='', bbox=None, fitter=None):
        """
        Parameters
        ----------
        image : 2d array
            Image to be processed
        catalog : astropy.table.Table instance
            The catalog of sources.  It has to contain at least three columns,
            `cx`, `cy`: the approximate centroid (x, y) of the source.
            `image` : str, the unique ID that can be used to identify image of
                the source.
            If `catalog` contains a column `flux`, then this column will be
            used to sort the source from the brightest to the faintest for PSF
            fitting.
        bbox : (by, bx), number
            The default box size in pixels for all sources, if `catalog` table
            does not contain `bx` and `by` columns.
        fitter : astropy.modeing.fitting.Fitter class object
            Fitter used to perform PSF fitting.  Default is
            `astropy.modeling.fitting.LevMarLSQFitter`
        """
        if fitter is None:
            fitter = LevMarLSQFitter()
        self.bbox = bbox
        self.fitter = fitter
        self._generate_filelist(datadir)
        self._initialize_source_set(catalog)

    @property
    def catalog(self):
        return [x.catalog for x in self.sgroup]

    @property
    def model_parm(self):
        return [x.model_parm for x in self.sgroup]

    def _initialize_source_set(self, catalog):
        """Initialize source set that contains groups of particles
        corresponding to different images"""
        from os.path import basename
        self.sgroup = []
        imnames = np.unique(catalog['image'])
        for nn in imnames:
            indices = catalog.index('image', nn)
            im = self._load_image(nn)
            if im is None:
                raise IOError('Could not load image {}.'.format(nn))
            sg = PSFSourceGroup(im, catalog[indices], bbox=self.bbox,
                    mask=im>4094)
            self.sgroup.append(sg)

    def _generate_filelist(self, datadir):
        """Compile a list of file contained in the data directory, with a
        uniform indexing that does not include level and version number.
        """
        from os.path import basename
        self._file_list = findfile(datadir, '.fits', recursive=True)
        self._file_id = [basename(x).split('.')[0].split('_')[0].split('S')[0]
                         for x in self._file_list]

    def _load_image(self, img_id):
        """Return the image based on the input id
        """
        from os.path import basename
        try:
            match = self._file_id.index(basename(img_id).split('.')[0].split(
                                        '_')[0].split('S')[0])
        except ValueError:
            return None
        return readfits(self._file_list[match],
                verbose=False).astype('float32')

    def fit(self,  model=None, fitter=None, niter=1):
        """Fit PSF model to all sources.  See `PSFSourceGroup.fit`
        """
        for sg in self.sgroup:
            sg.fit(model=model, fitter=fitter, niter=niter)

    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)


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
