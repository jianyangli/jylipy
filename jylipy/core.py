'''
Comet data analysis package

Module dependency
-----------------
numpy, scipy.interpolate.interp1d

History
-------
8/23/2013, started by JYL @PSI
'''

import collections
import warnings
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ccdproc
from .saoimage import getds9
from .apext import *
import contextlib

class jylipyDeprecationWarning(DeprecationWarning):
    pass


def is_iterable(v):
    """Check whether a variable is iterable"""
    if isinstance(v, (str, bytes)):
        return False
    elif hasattr(v, '__iter__'):
        return True
    else:
        return False


def quadeq(a, b, c):
    '''Solving quadratic equation
      a * x**2 + b * x + c = 0
    Returns a `None` for no solution, a numpy array of length 1 for
    one solution or length 2 for two solutions

    v1.0.0 : JYL @PSI, 2/23/2016
    '''
    d = b**2-4*a*c
    if d < 0:
        return None
    elif d == 0:
        return np.array([-b/(2*a)])
    else:
        d = np.sqrt(d)
        return (-b+d*np.array([-1, 1]))/(2*a)


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class CaseInsensitiveMapping(collections.MutableMapping):
    '''
    A case-insensitive ``dict``-like object.
    Implements all methods and operations of
    ``collections.MutableMapping`` as well as dict's ``copy``. Also
    provides ``lower_items``.
    All keys are expected to be strings. The structure remembers the
    case of the last key to be set, and ``iter(instance)``,
    ``keys()``, ``items()``, ``iterkeys()``, and ``iteritems()``
    will contain case-sensitive keys. However, querying and contains
    testing is case insensitive:
        cid = CaseInsensitiveDict()
        cid['Accept'] = 'application/json'
        cid['aCCEPT'] == 'application/json'  # True
        list(cid) == ['Accept']  # True
    For example, ``headers['content-encoding']`` will return the
    value of a ``'Content-Encoding'`` response header, regardless
    of how the header name was originally stored.
    If the constructor, ``.update``, or equality comparison
    operations are given keys that have equal ``.lower()``s, the
    behavior is undefined.

    v1.0.0 : JYL @PSI, 1/20/2015.
      Following https://github.com/kennethreitz/requests/blob/v1.2.3/requests/structures.py#L37
    '''
    def __init__(self, cls, data=None, **kwargs):
        self._store = cls()
        if data is None:
            data = cls()
        self.update(data, **kwargs)

    def __setitem__(self, key, value):
        # Use the lowercased key for lookups, but store the actual
        # key alongside the value.
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._store[key.lower()][1]

    def __delitem__(self, key):
        del self._store[key.lower()]

    def __iter__(self):
        return (casedkey for casedkey, mappedvalue in list(self._store.values()))

    def __len__(self):
        return len(self._store)

    def lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return (
            (lowerkey, keyval[1])
            for (lowerkey, keyval)
            in list(self._store.items())
        )

    def __eq__(self, other):
        if isinstance(other, collections.Mapping):
            other = type(self)(type(self._store),other)
        else:
            return NotImplemented
        # Compare insensitively
        return type(self._store)(self.lower_items()) == type(self._store)(other.lower_items())

    # Copy is required
    def copy(self):
        return type(self)(list(self._store.values()))

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, type(self._store)(list(self.items())))


class CaseInsensitiveDict(CaseInsensitiveMapping):
    '''Case insensitive dict class'''
    def __init__(self, data=None, **kwargs):
        super(CaseInsensitiveDict, self).__init__(dict, data, **kwargs)


class CaseInsensitiveOrderedDict(CaseInsensitiveMapping):
    '''Case insensitive ordered dict class'''
    def __init__(self, data=None, **kwargs):
        super(CaseInsensitiveOrderedDict, self).__init__(collections.OrderedDict, data, **kwargs)


def ascii_read(*args, **kwargs):
    '''astropy.io.ascii.read wrapper

    Same API and functionalities as ascii.read
    Returns the extended Table class'''
    return Table(ascii.read(*args, **kwargs))


def imdisp(im, ds9=None, newwindow=False, **kwargs):
    '''Display an image in DS9.

 Parameters
 ----------
 im : string or string sequence, 2-D or 3-D array-like numbers
   File name, sequence of file names, image, or stack of images.  For
   3-D array-like input, the first dimension is the dimension of stack
 ds9 : ds9 instance, or None, optional
   The target DS9 window.  If `None`, then the first opened DS9
   window will be used, or a new window will be opened if none exists.
 newwindow : bool, optional
   Same as the `new` keyword for getds9()
 Accept all keywords for DS9.imdisp()

 Returns
 -------
 DS9 instance that display the image

 v1.0.0 : JYL @PSI, Nov 17, 2013
 v1.0.1 : JYL @PSI, October 31, 2014
   Added keywords `newframe`, `status`, `silent`.
   Major restructuring, changing recursive call to loop
 v1.1.0 : JYL @PSI, 2/13/2015
   Accept astropy.nddata.NDData type and inherited classes
   Major reconstruction
   Removed keyword `status`.
   Add keyword `newwindow`.
 v2.0.0 : JYL @PSI, 2/14/2015
   Move all the major function into DS9.imdisp()
 v2.0.1 : JYL @PSI 2/19/2015
   Change keyword `silent` to `verbose`
    '''
    d = getds9(ds9, newwindow)
    d.imdisp(im, **kwargs)
    return d


class CCDData(ccdproc.CCDData):
    '''Subclass of ccdproc.CCDData

    Removed the required keyword `unit` from ccdproc.CCDData, assigning
    unit `dimensionless_unscaled` if `unit` is not specified.

    Add field .flags

    v1.0.0 : JYL @PSI, 2/19/2015.
    '''

    def __init__(self, *args, **kwargs):
        super(CCDData, self).__init__(*args, **kwargs)

        if len(args) == 0:
            flags = None
        else:
            if isinstance(args[0], CCDData):
                flags = getattr(args[0], 'flags', None)
            else:
                flags = kwargs.pop('flags', None)

        if flags is None:
            self.flags = None
        elif isinstance(flags, np.ndarray):
            if flags.shape != self.shape:
                raise ValueError('flags must have the same shape as data')
            self.flags = flags
        elif isinstance(flags, nddata.FlagCollection):
            for f in flags:
                if f.shape != self.shape:
                    raise ValueError('all flags must have the same shape as data')
            self.flags = flags
        else:
            raise TypeError('flags type not recoganized')


class Image(CCDData):
    '''Image class'''

    def __init__(self, *args, **kwargs):
        super(Image, self).__init__(*args, **kwargs)
        if self.data.ndim != 2:
            raise TypeError('Image can only be 2-dimensional, {0} dimensional data received'.format(self.data.ndim))


def aperture_photometry(data, apertures, **kwargs):
    '''Wrapper for photutils.aperture_photometry

    Fixes a bug in the original program (see below).
    Returns the extended Table class.

    ...bug description:
    The original program that puts the `xcenter` and `ycenter` in a
    shape (1,1) array when input aperture contains only one position.

    v1.0.0 : JYL @PSI, Feb 2015'''

    from photutils import aperture_photometry
    ap = Table(aperture_photometry(data, apertures, **kwargs))
    if apertures.positions.shape[0] == 1:
        xc = ap['xcenter'].data[0]
        yc = ap['ycenter'].data[0]
        ap.remove_column('xcenter')
        ap.remove_column('ycenter')
        ap.add_column(Column([xc],name='xcenter'))
        ap.add_column(Column([yc],name='ycenter'))
    if ap['xcenter'].unit is None:
        ap['xcenter'].unit = units.pix
    if ap['ycenter'].unit is None:
        ap['ycenter'].unit = units.pix
    return ap


def apphot(im=None, aperture=None, ds9=None, radius=3, newwindow=False, **kwargs):
    '''Measure aperture photometry

    If no image is given, then the image will be extracted from the
    active frame in a DS9 window specified by `ds9.`

    If no DS9 window is specified, then the first-openned DS9 window
    will be used.

    If no aperture is given, then the aperture(s) will be extracted
    from the circular and annulus regions in the DS9 window.  If no
    regions is defined, then the aperture center will be specified
    interactively by mouse click in DS9, and the radius of apertures
    will be specified by keyword `radius`.

    Photometry will be returned in a Table.

    v1.0.0 : JYL @PSI, Feb 2015
    v1.0.1 : JYL @PSI, Jan 18, 2017
      Bug fix
    '''
    if im is None and ds9 is None:
        raise ValueError('either `im` or `ds9` must be specified')

    if im is None:  # Get image from a ds9 window
        im = getds9(ds9).get_arr2np()

    if aperture is None:  # If no aperture specified, then get it from DS9
        ds9 = getds9(ds9, newwindow)
        ds9.imdisp(im)
        aperture = ds9.aperture()
        if not aperture:
            centroid = kwargs.pop('centroid', False)
            aperture = ds9.define_aperture(radius=radius, centroid=centroid)
    if aperture == []:  # If still not specified, do nothing
        return None

    # Measure photometry
    pho, r, r1 = [], [], []
    if not isinstance(aperture,list):
        aperture = [aperture]
    for apt in aperture:
        napt = apt.positions.shape[0]
        if hasattr(im,'uncertainty'):
            error = im.uncertainty.array
        else:
            error = None
        error = kwargs.pop('error', error)
        pho.append(aperture_photometry(im, apt, error=error, **kwargs))
        if napt == 1:
            r.append(getattr(apt, 'r', getattr(apt, 'r_in', None)))
            r1.append(getattr(apt, 'r_out', None))
        else:
            r = r+[getattr(apt, 'r', getattr(apt, 'r_in', None))]*napt
            r1 = r1+[getattr(apt, 'r_out', None)]*napt
    if pho == []:
        return None
    from .apext import table
    pho = Table(table.vstack(pho))
    pho.add_column(Column(r, name='r', unit='pix'))
    if len(np.nonzero(r1)[0]) > 0:
        pho.add_column(Column(r1, name='r_out', unit='pix'))
    return pho


def num(v):
    '''Try to convert a string to int or float, if not succssful,
    silently returns the original string'''
    try:
        return int(v)
    except:
        try:
            return float(v)
        except:
            return v


def geometric_center(image, threshold, mask=None):
    '''Returns the geometric center of an image for values above a
    threshold

    image : array_like
      The 2D data array.

    threshold : number
      The threshold of image above background.

    mask : array_like(bool), optional
      A boolean mask, with the same shape as `data`, where `True` value
      indicates the corresponding element of `data` is masked.

    Returns: `center`: array of length 2, the `x, y` coordinates
    '''
    sz = image.shape
    y, x = makenxy(0, sz[0]-1, sz[0], 0, sz[1]-1, sz[1])
    im_mask = np.asanyarray(image) > threshold
    if mask is not None:
        im_mask |= mask
    yin, xin = y[im_mask], x[im_mask]
    return xin.mean(), yin.mean()


def size2mag(radi, alb=0.1, rh=1.0, delta=1.0, phase=0.0, beta=0.04, magsun=-26.74):
    '''Calculate magnitude from radius.

 Parameters
 ----------
 radi : number, array-like, or astropy Quantity
     Radius of object [km]
 alb  : number, array-like, optional
     Geometric albedo
 rh   : number, array-like, or astropy Quantity, optional
     Heliocentric distance [AU]
 delta : number, array-like, or astropy Quantity, optional
     Observer distance [AU]
 phase : number, array-like, or astropy Quantity, optional
     Phase angles in [deg]
 beta : number, array-like, optional
     Phase coefficient [mag/deg]
 magsun : number, or astropy Quantity, optional
   Apparent magnitude of the Sun at 1 AU

 Returns
 -------
 Number or numpy array
   Visual magnitude

 v1.0.0 : JYL @PSI, Jul 31, 2013.
 v1.0.1 : JYL @PSI, Nov 18, 2013.
   Added acceptance of astropy Quantity as input
 v1.0.2 : JYL @PSI, December 22, 2013.
   Add keyword `magsun` to allow calculations in wavelengths other
     than V-band
    '''

    q = False
    if isinstance(radi, units.Quantity):
        radi = radi.to(units.km).value
        q = True
    if isinstance(rh, units.Quantity):
        rh = rh.to(units.au).value
        q = True
    if isinstance(delta, units.Quantity):
        delta = delta.to(units.au).value
        q = True
    if isinstance(phase, units.Quantity):
        phase = phase.to(units.deg).value
        q = True
    if isinstance(beta, units.Quantity):
        beta = beta.to(units.Unit('mag/deg')).value
        q = True
    if isinstance(magsun, units.Quantity):
        magsun = magsun.to(units.mag).value
        q = True

    redmag = -2.5*np.log10(alb*radi*radi/units.au.to(units.km)**2)+magsun
    mag = redmag+5*np.log10(rh*delta)+beta*phase

    if q:
        return mag*units.mag
    else:
        return mag


def mag2size(mag, alb=0.1, rh=1.0, delta=1.0, phase=0.0, beta=0.04, magsun=-26.74):
    '''Calculate radius from magnitude

 Parameters
 ----------
 mag : number, or array-like, or astropy Quantity
   Magnitude of object
 alb : number, array-like, optional
   Geometric albedo
 rh  : number, array-like, or astropy Quantity, optional
   Heliocentric distance in AU
 delta : number, array-like, or astropy Quantity, optional
   Observer distance in AU
 phase : number, array-like, or astropy Quantity, optional
   Phase angles in degrees
 beta : number, array-like, or astropy Quantity, optional
   Phase coefficient in mag/deg.
 magsun : number, or astropy Quantity, optional
   Magnitude of the Sun

 Returns
 -------
 Returns the radius [km]

 v1.0.0 : JYL @PSI, Jul 31, 2013.
 v1.0.1 : JYL @PSI, Nov 18, 2013.
   Added acceptance of astropy Quantity as input.
 v1.0.2 : JYL @PSI, December 22, 2013.
   Add keyword `magsun` to allow calculations in wavelengths other
     than V-band
    '''

    q = False
    if isinstance(mag, units.Quantity):
        mag = mag.to(units.mag).value
        q = True
    if isinstance(rh, units.Quantity):
        rh = rh.to(units.au).value
        q = True
    if isinstance(delta, units.Quantity):
        delta = delta.to(units.au).value
        q = True
    if isinstance(phase, units.Quantity):
        phase = phase.to(units.deg).value
        q = True
    if isinstance(beta, units.Quantity):
        beta = beta.to(units.Unit('mag/deg')).value
        q = True
    if isinstance(magsun, units.Quantity):
        magsun = magsun.to(units.mag).value
        q = True

    redmag = mag-5*np.log10(rh*delta)-beta*phase
    radi = units.au.to(units.km)/np.sqrt(alb)*10**(-0.2*(redmag-magsun))

    if q:
        return radi*units.km
    else:
        return radi


def mag2alb(mag, radi, rh=1.0, delta=1.0, phase=0.0, beta=0.04, magsun=-26.74):
    '''Calculate albedo from magnitude and size

 Parameters
 ----------
 mag : number, or array-like, or astropy Quantity
   Magnitude of object
 radi : number, array-like, or astropy Quantity
   Radius of object [km]
 rh  : number, array-like, or astropy Quantity, optional
   Heliocentric distance in AU
 delta : number, array-like, or astropy Quantity, optional
   Observer distance in AU
 phase : number, array-like, or astropy Quantity, optional
   Phase angles in degrees
 beta : number, array-like, or astropy Quantity, optional
   Phase coefficient in mag/deg.
 magsun : number, or astropy Quantity, optional
   Magnitude of the Sun

 Returns
 -------
 Returns the albedo

 v1.0.0 : JYL @PSI, December 22, 2013
    '''

    q = False
    if isinstance(mag, units.Quantity):
        mag = mag.to(units.mag).value
        q = True
    if isinstance(radi, units.Quantity):
        radi = radi.to(units.km).value
        q = True
    if isinstance(rh, units.Quantity):
        rh = rh.to(units.au).value
        q = True
    if isinstance(delta, units.Quantity):
        delta = delta.to(units.au).value
        q = True
    if isinstance(phase, units.Quantity):
        phase = phase.to(units.deg).value
        q = True
    if isinstance(beta, units.Quantity):
        beta = beta.to(units.Unit('mag/deg')).value
        q = True
    if isinstance(magsun, units.Quantity):
        magsun = magsun.to(units.mag).value
        q = True

    redmag = mag-5*np.log10(rh*delta)-beta*phase
    alb = (units.au.to(units.km)/radi)**2*10**(-0.4*(redmag-magsun))

    return alb


def afrho(fc, rho, fs=None, rh=1.0, delta=1.0, phase=0., magnitude=False):
    '''Calculate Afrho from comet flux or magnitude

 Parameters
 ----------
 fc  : number, array-like numbers
   Comet flux, either in flux unit or in magnitude
 rho : number, array-like numbers
   Aperture radius in arcsec
 fs  : number, array-like numbers, optional
   Solar flux, either in the same flux unit as the comet flux, or in
   magnitude.  Default is the standard V-band value
 rh, delta : number, array-like numbers, optional
   Heliocentric distance and geocentric distance [AU]
 phase : number, array-like numbers, optional
   Phase angle [deg]
 magnitude : bool, optional
   Specify whether the input comet and solar flux is in flux or in
   magnitude units

 Returns
 -------
 number(s): Afrho in [cm]

 Notes
 -----
 Program follows the definition in A'Hearn et al., (1984, AJ 89,
 579-591).

    Afrho (cm) = F_c/F_s * (2 rh[AU] delta[km] / rho[km])**2 * rho(cm)
               = 1.234e19 * F_c / F_s * rh[AU]**2 * delta[AU] / rho["]

 The correction for phase angle uses the dust phase function by
 Schleicher http://asteroid.lowell.edu/comet/dustphase.html

 History
 -------
 v1.0.0 : JYL @PSI, Aug 23, 2013
 v1.0.1 : JYL @PSI, Nov 19, 2013
   Added acceptance of astropy quantity as input.
 v1.0.2 : JYL @PSI, Apr 30, 2015
   Add phase function correction
    '''

    if fs is None:
        if magnitude:
            fs = -26.74
        else:
            fs = 1.84e3

    q = False
    if isinstance(rho, units.Quantity):
        rho = rho.to(units.arcsec).value
        q = True
    if isinstance(rh, units.Quantity):
        rh = rh.to(units.au).value
        q = True
    if isinstance(delta, units.Quantity):
        delta = delta.to(units.au).value
        q = True
    if isinstance(phase, units.Quantity):
        phase = phase.to(units.deg).value
        q = True
    if isinstance(fc, units.Quantity) and isinstance(fs, units.Quantity):
        fc = fc.to(fs.unit).value
        fs = fs.value
        q = True
    elif isinstance(fc, units.Quantity):
        if magnitude:
            fc = fc.to(units.mag).value
        else:
            fc = fc.to(units.Unit('W/(m2 um)')).value
        q = True
    elif isinstance(fs, units.Quantity):
        if magnitude:
            fs = fs.to(units.mag).value
        else:
            fs = fs.to(units.Unit('W/(m2 um)')).value
        q = True

    if magnitude:
        fcfs = 10**(0.4*(fs-fc))
    else:
        fcfs = fc/fs
    afr = 1.234e19*fcfs*rh**2*delta/rho

    if hasattr(phase, '__iter__'):
        if (np.asarray(phase) != 0).any():
            phcorr = dust_phasefunc()(phase)
            afr /= phcorr
    else:
        if phase != 0:
            phcorr = dust_phasefunc()(phase)
            afr /= phcorr

    if q:
        return afr*units.cm
    else:
        return afr


class dust_phasefunc(object):
    '''Coma dust phase function
 From http://asteroid.lowell.edu/comet/dustphase.html

 v1.0.0 : 4/30/2015, JYL @PSI
    '''
    datafile = '/Users/jyli/work/references/dust_phasefunc/dustphaseHM_table.txt'

    def __init__(self):
        from scipy.interpolate import interp1d
        self.table = ascii_read(self.datafile,data_start=5)
        self.table.rename_column('col1','phase')
        self.table.rename_column('col2','func0')
        self.table.rename_column('col3','func90')
        self.phasefunc = interp1d(self.table['phase'],self.table['func0'],kind='cubic')

    def __call__(self, ph, normalized=0.):
        if hasattr(ph,'__iter__'):
            if (ph<0).any() or (ph>180).any():
                raise ValueError('phase angle must be within [0, 180]')
        else:
            if ph<0 or ph>180:
                raise ValueError('phase angle must be within [0, 180]')
        func = self.phasefunc(ph)
        if normalized != 0.:
            func /= self.phasefunc(normalized)
        return func


class linspec(object):
    '''Linear spectral slope class
 v1.0.0: 4/30/2015, JYL @PSI
    '''
    def __init__(self, slp, wvs):
        '''
        slp : linear slope in %/100 nm
        wvs : (wv1, wv2) in nm, at which the slope is measured
        '''
        slp /= 100
        self.slope = slp
        self.wv1, self.wv2 = wvs
        x2 = slp*(self.wv2-self.wv1)/100.
        self.func = lambda wv: x2/(self.wv2-self.wv1)*(np.asarray(wv)-self.wv1)+1-x2/2

    def __call__(self, wv):
        return self.func(wv)

    def to(self, wvs):
        '''Translate the slope to between new pairs of wavelengths'''
        wv1, wv2 = wvs
        sp1, sp2 = self(wvs)
        return (sp2-sp1)/(sp2+sp1)*2/(wv2-wv1)*100*100


def coma(azind, azfth, size=(300,300), center=None, ns=200, core=(2,2), r=None, theta=None):
    '''Generate a coma dust model f(theta)/r^n(theta)

 Parameters
 ----------
 azind  : number or array-like
   Power-law index (indices) of coma model.  If it is a single
   number, then the coma model uses one single power-law index.
   If it is an array, then the values correspond to the indices
   along all azimuthal angles starting from 0.
 azfth  : number of array-like
   Scaling factor of the model.  Similar to azind.
 size : tuple, optional
   The size of output image, in pixels in python convention.
 center : tuple, optional
   The center of coma model, in pixels in python convention.  Default
   is (size-1)/2.
 ns : integer
   Oversampling factor for the central (core) region, where each
   pixel is oversampled to ns pixels in the calculation.
 core : tuple, optional
   The size of the core region to be oversampled in calculation,
   in pixels in horizontal and verticle direction, respectively.
 r, theta : variable
   Return the distance array and azimuthal angle array of output
   model, in [pix] and [deg], respectively.

 Output
 ------
 Returns numpy array, the dust coma model

 Notes
 -----
 1. ns should always be a positive number.  Negative number is accepted
 but is for internal flow control.  User should never pass a negative
 value to this parameter.
 2.  The accuracy of peak pixel value depends on the oversampling
 parameter 'ns', which in turn depends on the slope.  For a slope of
 1., ns>=20 will achieve 95\% accuracy for the peak pixel.  But for
 a slope of 1.5, ns needs to be >=200.  The routine is not stable
 for slope >=2.0.

 History
 -------
 8/23/2013, created by JYL @PSI
 10/20/2014, JYL @PSI
   Changed the order of (x,y) in `size` and `center` to follow the
   python convention rather than (horizontal, vertical).
    '''

    # use Y. Fernandez's Numerical results for central pixel integral
    # if the central pixel has a distance of 0.  This is just a way to
    # avoid infinity with a reasonable accuracy.  The final results
    # depends little on these values.
    centpixval = [1.113,1.242,1.391,1.564,1.764,1.997, 2.273, 2.600, 2.990, \
                  3.462,4.036,4.743,5.622,6.729,8.136,9.949,12.311,15.427, \
                  19.585]
    indexval = 0.10 + np.arange(19)*0.1
    #-----------------------------------------------------------------------


    ys, xs = size
    if center is None:
        yc, xc = (ys-1.)/2., (xs-1.)/2.
    else:
        yc, xc = center

    # initialize variables
    yy, xx = np.mgrid[0:ys,0:xs].astype(float)
    xx -= xc
    yy -= yc
    r = np.sqrt(xx*xx+yy*yy)    # distance array
    if ns < 0:
        r /= (-ns)
    theta = np.arctan2(yy, xx)/np.pi*180
    theta = (theta+360) % 360   # azimuthal angle array

    # generate a preliminary model
    from scipy.interpolate import interp1d
    if hasattr(azind,'__iter__') and np.size(azind) > 1:
        az = np.linspace(0,360,len(azind)+1)
        inds = interp1d(az, np.append(azind,azind[0]), bounds_error=False)(theta)
        fths = interp1d(az, np.append(azfth,azfth[0]), bounds_error=False)(theta)
        dust = fths/r**inds
        if r.min() == 0:
            dust[np.unravel_index(r.argmin(),r.shape)] = \
                interp1d(indexval, centpixval, bounds_error=False)(azind).mean()
    else:
        dust = azfth/r**azind
        if r.min() == 0:
            dust[np.unravel_index(r.argmin(),r.shape)] = \
                interp1d(indexval, centpixval, bounds_error=False)(azind)

    # revise core region
    if ns > 1:
        x1, x2 = int(round(xc)-core[0]), int(round(xc)+core[0]+1)
        y1, y2 = int(round(yc)-core[1]), int(round(yc)+core[1]+1)
        coresize = (core[1]*2+1)*ns, (core[0]*2+1)*ns
        center = ns*(core[1]+0.5+xc-round(xc))-0.5, ns*(core[0]+0.5+yc-round(yc))-0.5
        coma_center = coma(azind,azfth,size=coresize, center=center, ns=-ns)
        dust[y1:y2,x1:x2] = rebin(coma_center, (ns, ns), mean=True)

    return dust


def comafit(im, center=None, azbin=10, rrange=(5.,30), snmap=None, sncutoff=3.):
    '''
 Fit a coma model with azimuthal 1/rho**k power-law

 Input
 -----
 im : array-like, numbers
     Input image
 center : tuple of two numbers, optional
     The center of input image in horizontal and verticle directions
     [pix].  If not provided, then the center will be fitted using
     using improc.centroid around the maximum pixel
 azbin : number, optional
     Azimuthal bin number
 rrange : tuple of two numbers, optional
     The range in radial direction within which the model is fitted
     [pix]
 snmap : array-like, numbers, same shape as im, optional
     Signal-to-noise map
 sncutoff : number, optional
     S/N cut off.  If snmap is not provided, then it is ignored

 Output
 ------
 Returns a tuple of two arrays, containing the best-fit indices and
 scaling factors of the coma model

 Notes
 -----
 1. The input image will be unwrapped to polar projection before fit.
 The number of bins in azimuthal direction is always 360 (1 deg bins).
 The bin size in radial direction is always 0.5 pixel.
 2. snmap is unwrapped in the same manner as the original image.

 History
 -------
 8/25/2013, created by JYL @PSI
    '''

    if type(im) != np.ndarray:
        im0 = np.array(im)
    else:
        im0 = im

    if snmap != None:
        if type(snmap) != nd.array:
            snmap0 = np.array(snmap)
        else:
            snmap0 = snmap

    # find the center
    if center == None:
        asdf
    else:
        xc, yc = center

    # unwrap image to polar coordinate
    im1 = xy2rt(im0, center, ramax=rrange[1],rabin=rrange[1]*2+1, azbin=360)
    im1 = rebin(im1,360/azbin,1)*azbin/360.
    if snmap != None:
        snmap1 = xy2rt(snmap0, center, ramax=rrange[1],rabin=rrange[1]*2+1, azbin=360)
        snmap1 = rebin(snmap0, 360/azbin, 1)*azbin/360.

    # fit azimuthal cuts
    azind = np.empty(azbin)
    azfth = np.empty(azbin)
    ra = np.linspace(0, rrange[1], rrange[1]*2+1)
    for i in range(azbin):
        ind = ra >= rrange[0]
        if snmap != None:
            ind = ind and (snmap1[:,i] >= sncutoff)
        azind[i], azfth[i] = fitpowerlaw(ra[ind], im1[ind,i], method='log')

    return azind, azfth


def flux2ref(flux, wv, radius, error=None, helio=1., geo=1., phasecorr=1., bin=1., solar=None):
    '''
 Convert flux spectrum to reflectance spectrum

 Input
 -----
 flux, wv: array-like, float
   Input flux [W/m**2/um] and corresponding wavelength [um]
 error: array-like, float
   Error bars for input flux, same unit as flux
 radius: number
   Radius of the object
 helio, geo, phasecorr: number, optional
   Heliocentric distance [AU], geocentric distance [AU], and phase
   angle correction factor
 bin: number, >=1, optional
   Bin size for input spectrum, in unit of number of points

 Output
 ------
 (ref, wv), or (ref, wv, err): Tuple, float
 ref, wv: array-like, number
   Reflectance spectrum and corresponding wavelengths
 err: array, number
   Error bar.  Present only if keyword `error` is set to not None.

 v1.0.0, JYL @PSI, June 2, 2014
    '''

    from scipy.interpolate import interp1d

    flux, wv = np.asarray(flux), np.asarray(wv)

    if solar is None:
        # read in solar spectrum
        sun = ascii.read('/Users/jyli/work/archives/Sun/E490_00a_AM0.tab')
        # interpolation
        ww = (sun['col1']<=wv.max()*1.1) & (sun['col1']>=wv.min()*0.9)
        sunspec = interp1d(sun['col1'][ww],sun['col2'][ww],kind='cubic')
    else:
        ww = (solar[0]<=wv.max()*1.1) & (solar[0]>=wv.min()*0.9)
        sunspec = interp1d(solar[0][ww],solar[1][ww],kind='cubic')

    # convert to reflectance
    factor = helio*helio*geo*geo*phasecorr/((radius/1.496e8)**2*sunspec(wv))
    ref = flux*factor
    wvf = wv.copy()

    # propagate error and binning
    if error is not None:
        error = np.asarray(error)
        referr = error*factor
        weight = 1/(referr*referr)
        if bin>1:
            wvf = rebin(wv,bin,mean=True)
            ref = rebin(ref*weight,bin)/rebin(weight,bin)
            referr = 1/np.sqrt(rebin(weight,bin))
        return ref, wvf, referr
    else:
        if bin>1:
            wvf = rebin(wv,bin,mean=True)
            ref = rebin(ref,bin,mean=True)
        return ref, wvf


def syncsynd(comet, utc, beta, dt, observer='Earth', frame='J2000', kernel=None, three=False, vinit=None):
    ''' Calculate synchrones and syndynes for a comet

 Parameters
 ----------
 comet : str
   Name or JPL Horizons record number of a comet
 utc : str
   UTC time of the calculation, in a format recoganizable by SPICE
 beta : sequence of float
   The beta's of dust to be calculated
 dt : sequence of time
   The time of synchrones to be calculated, in days before `utc`
 observer : str, optional
   The observer.  Default is the geometry from Earth.
 frame : str, optional
   Specify the coordinate frame of output vectors.
 kernel : str, optional
   The SPICE kernel file.  Default is `None`, in which case the
   kernels need to be loaded beforehand.
 three : bool, optional
   If `True`, then the 3-D position (x,y,z) of synchrones and syndynes
   in `observer`-centered, J2000 coordinates are returned.  Otherwise
   the apparent angular position (RA, Dec) from `observer` is
   returned.
 vinit : 3-element sequence, float, optional
   Adding an intial velocity to the synchrones and syndynes.  It has
   to be a 3-element vector, in km/s, and in `frame`.

 Returns
 -------
 (syncsynd_array, comet_pos)
 syncsynd_array : 3-D array of dimension (beta.shape, dt.shape, 2 or
   3), float The `observer`-centered coordinate of synchrones and
   syndynes.  The first dimension is syndynes, and the second
   dimension the synchrones of the corresponding time and beta,
   respectively, and the third dimension is the coordinate dimension.
   The last dimension has a length of either 2 or 3, depending on the
   keyword `three`.  In units of either deg or km, respectively.
 comet_pos : A 2- or 3-element array containing the
   `observer`-centered position of the comet.

 v1.0.0 : JYL @PSI, 6/9/2014
    '''

    if beta.max() > 1:
        raise ValueError('beta > 1 cannot be calculated.')

    gm = (constants.G*constants.M_sun).to('km3/s2').value

    if kernel is not None:
        spice.furnsh(kernel)

    et = spice.str2et(utc)
    # state and lighttime of comet from `observer` in output frame
    st_c, lt_c = spice.spkezr(comet, et, frame, 'lt+s', observer)
    # state and lighttime of comet in eclipj2000 w/r to the Sun
    st, lt = spice.spkezr(comet, et-lt_c, 'eclipj2000', 'none', 'sun')
    # position of observer in eclipj2000 frame w/r to the Sun
    posobs, lt = spice.spkpos(observer, et-lt_c, 'eclipj2000', 'none', 'sun')
    # transformation matrix from eclipj2000 to output frame
    m = np.array(spice.pxform('eclipj2000', frame, et-lt_c))
    # initial velocity
    if vinit is not None:
        vinit = np.concatenate((np.zeros(3),m.T.dot(vinit)))
    else:
        vinit = np.zeros(6)

    # construct synchrons & syndynes
    syn = np.empty((beta.shape[0], dt.shape[0], 3))
    for j in range(dt.shape[0]):
        # propagate comet backward
        st0 = np.array(spice.prop2b(gm, st, -dt[j]*86400.))+vinit
        for i in range(beta.shape[0]):
            # propagate partical forward
            if beta[i] == 1:
                syn[i,j] = st0[:3]+dt[j]*86400.*st0[3:]
            else:
                syn[i,j] = np.array(spice.prop2b(gm*(1-beta[i]), st0, dt[j]*86400.)[:3])
            # convert to output `frame`
            syn[i,j] = m.dot(syn[i,j]-posobs)

    if kernel is not None:
        spice.unload(kernel)

    if three is True:
        return syn, st_c[:3]
    else:
        return xyz2sph(syn.reshape(beta.shape[0]*dt.shape[0],3),row=True).T[:,1:].reshape(beta.shape[0],dt.shape[0],2), xyz2sph(st_c[:3])[1:]


def sflux(wv, th, spec=None):
    '''
 Calculate solar flux and magnitude through a filter

 Parameters
 ----------
 wv, th : number arrays
 Wavelength and throughput of filters.  `wv` can be numbers or astropy
   quantities.  If `wv` contains numbers, then the unit is 'um', or
   the same unit as in `spec`.
 spec : (2, n) list of numbers of quantities
 The spectrum of light source.

 Returns
 -------
 Equivalent flux (number or quantity) through the input filter.  The
 default unit is 'W/m-2 um-2'.

 v1.0.0 : JYL @PSI, November 4, 2014
    '''

    from scipy.interpolate import interp1d

    if spec is None:
        fsun = ascii.read('/Users/jyli/work/references/Sun/E490_00a_AM0.tab')
        fs = interp1d(fsun['col1'], fsun['col2'])
    else:
        if (type(spec[0]) is units.Quantity) and (type(spec[1]) is units.Quantity):
            swv = spec[0].to(units.Unit('um'))
            sfx = spec[1].to(units.Unit('W m-2 um-1'))
        elif type(spec[0]) is units.Quantity:
            swv = spec[0].value
            sfx = spec[1]
        elif type(spec[1]) is units.Quantity:
            swv = spec[0]
            sfx = spec[1].value
        else:
            swv, sfx = spec
        fs = interp1d(swv, sfx)

    if type(wv) is units.Quantity:
        wv_value = wv.to(units.Unit('um'))
    else:
        wv_value = wv

    d_wv = np.gradient(wv_value)
    result = (fs(wv_value)*th*d_wv*wv_value).sum()/(th*d_wv*wv_value).sum()
    if type(wv) is units.Quantity:
        result = result*units.Unit('W m-2 um-1')
    return result


def enhance_1overrho(im, ext=0, center=None, centroid=False, div=True):
    '''
 1/rho enhancement tool to study cometary coma morphology.

 Parameters
 ----------
 im : str, list of str, file, list of files, 2-D image, list of 2-D
 images, 3-D image cube
   Input images to be processed.  If 'str' or file types, it contains
   the name of image(s).  If 3-D image cube, then the first dimension
   of the array is assumed to specify image layers.
 ext : number, list of numbers, optional
   FITS extension number that stores the images to be processed
 center : [ycenter, xcenter], or list of [ycenter, xcenter], optional
   The optocenter of the comet in image.  Default will be at the
   geometric center of input image(s).
 centroid : bool, optional
   If `True`, then interactive centroiding will be performed, and the
   `center` keyword will be ignored.
 div : bool, optional
   If `True`, the enhancement will be performed with a division of
   the real image to 1/rho model.  Otherwise a subtraction will be
   performed.

 Returns
 -------
 2-D image or list of 2-D images of the same size(s) as the input
 image(s)

 v1.0.0 : JYL @PSI, November 5, 2014
 v1.0.1 : JYL @PSI, December 1, 2014
   Removed keyword `size`
    '''

    # Pre-process for the case of a single image
    if type(im) is str:
        im = [im]
    elif type(im) is np.ndarray:
        if im.ndim == 2:
            im = [im]
    nimgs = len(im)
    if center is not None:
        center = np.asarray(center)
        if center.ndim == 1:
            center = [center]*nimgs
    if not hasattr(ext, '__iter__'):
        ext = [ext]*nimgs

    # Loop through all input images
    ens = []
    for i in range(nimgs):

        # Load image
        if isinstance(im[i], (str, bytes)):
            img = readfits(im[i],ext=ext[i],verbose=False)
        else:
            img = im[i]

        # Centroiding
        if centroid:
            ct = centroiding(img, refine=True, newframe=False, verbose=False)
        else:
            ct = center[i]
        ct = int(ct[0]), int(ct[1])

        # Generate 1/rho model
        sz = img.shape
        comamodel = coma(1.,1.,size=sz, center=ct)
        scl = comamodel[ct[0]-1:ct[0]+2,ct[1]-1:ct[1]+2].sum()/img[ct[0]-1:ct[0]+2,ct[1]-1:ct[1]+2].sum()

        # Enhance image
        if div:
            en = img/comamodel*scl
        else:
            en = img-comamodel/scl

        ens.append(en)

    if len(ens) == 1:
        ens = ens[0]

    return ens


def azavg(im, ext=0, center=None, centroid=False):
    '''
 Generate an azimuthally averaged image based on the input image(s)

 Parameters
 ----------
 im : str, list of str, file, list of files, 2-D image, list of 2-D
 images, 3-D image cube
   Input images to be processed.  If 'str' or file types, it contains
   the name of image(s).  If 3-D image cube, then the first dimension
   of the array is assumed to specify image layers.
 ext : number, list of numbers, optional
   FITS extension number that stores the images to be processed
 center : [ycenter, xcenter], or list of [ycenter, xcenter], optional
   The optocenter of the comet in image.  Default will be at the
   geometric center of input image(s).
 centroid : bool, optional
   If `True`, then interactive centroiding will be performed, and the
   `center` keyword will be ignored.
 div : bool, optional
   If `True`, the enhancement will be performed with a division of
   the real image to 1/rho model.  Otherwise a subtraction will be
   performed.

 Returns
 -------
 2-D image or list of 2-D images of the same size as input images

 v1.0.0 : JYL @PSI, November 5, 2014
    '''

    from scipy.interpolate import interp1d

    # Pre-process for the case of a single image
    if type(im) is str:
        im = [im]
    elif type(im) is np.ndarray:
        if im.ndim == 2:
            im = [im]
    nimgs = len(im)
    if center is not None:
        center = np.asarray(center)
        if center.ndim == 1:
            center = [center]*nimgs
    if not hasattr(ext, '__iter__'):
        ext = [ext]*nimgs

    # Loop through all input images
    avgimgs = []
    for i in range(nimgs):

        # Load image
        if isinstance(im[i], (str, bytes)):
            img = readfits(im[i], ext=ext[i], verbose=False)
        else:
            img = im[i]

        # Centroiding if needed
        if centroid:
            ct = centroiding(img, refine=True, newframe=False)
        else:
            ct = center[i]

        # Generate azimuthally models
        sz = img.shape
        ct = np.asarray(ct)
        rmax = int(np.ceil(np.linalg.norm([np.asarray([0,0])-ct, np.asarray([0,sz[1]-1])-ct, np.asarray([sz[0]-1,0])-ct, np.asarray([sz[0]-1,sz[1]-1])-ct],axis=1).max()))
        azimg = xy2rt(img, center=ct, ramax=rmax, rastep=rmax*2+1, azstep=720, method='splinef2d')
        radprof = interp1d(np.linspace(0,rmax,rmax*2+1), np.median(azimg,axis=1))
        dst = dist(-ct[0], sz[0]-ct[0]-1, sz[0], -ct[1], sz[1]-ct[1]-1, sz[1])
        avgim = radprof(dst.reshape(-1)).reshape(sz)

        avgimgs.append(avgim)

    if len(avgimgs) == 1:
        avgimgs = avgimgs[0]

    return avgimgs


def enhance_azavg(im, ext=0, center=None, centroid=False, div=True):
    '''
 Azimuthal average removal enhancement tool to study cometary coma morphology.

 Similar to `enhance_1overrho`, but uses azimuthally averaged image as
 the coma model.

 Parameters
 ----------
 im : str, list of str, file, list of files, 2-D image, list of 2-D
 images, 3-D image cube
   Input images to be processed.  If 'str' or file types, it contains
   the name of image(s).  If 3-D image cube, then the first dimension
   of the array is assumed to specify image layers.
 ext : number, list of numbers, optional
   FITS extension number that stores the images to be processed
 size : [ysize, xsize], or list of [ysize, xsize], optional
   The size of output image(s).  Default is the same size as the
   input.
 center : [ycenter, xcenter], or list of [ycenter, xcenter], optional
   The optocenter of the comet in image.  Default will be at the
   geometric center of input image(s).
 centroid : bool, optional
   If `True`, then interactive centroiding will be performed, and the
   `center` keyword will be ignored.
 div : bool, optional
   If `True`, the enhancement will be performed with a division of
   the real image to 1/rho model.  Otherwise a subtraction will be
   performed.

 Returns
 -------
 2-D image or list of 2-D images

 v1.0.0 : JYL @PSI, November 5, 2014
    '''

    # Pre-process for the case of a single image
    if type(im) is str:
        im = [im]
    elif type(im) is np.ndarray:
        if im.ndim == 2:
            im = [im]
    nimgs = len(im)
    if center is not None:
        center = np.asarray(center)
        if center.ndim == 1:
            center = [center]*nimgs
    if not hasattr(ext, '__iter__'):
        ext = [ext]*nimgs

    # Loop through all input images
    ens = []
    for i in range(nimgs):

        # Load image
        if isinstance(im[i],(str,bytes)):
            img = readfits(im[i],ext=ext[i],verbose=False)
        else:
            img = im[i]

        # Centroiding
        if centroid:
            ct = centroiding(img, refine=True, newframe=False, verbose=False)
        else:
            ct = center[i]

        # Generate azimuthally averaged model
        sz = img.shape
        comamodel = azavg(img, center=ct)

        # Enhance image
        if div:
            en = img/comamodel
        else:
            en = img-comamodel

        ens.append(en)

    if len(ens) == 1:
        ens = ens[0]

    return ens


class Filter(object):
    ''' '''

    def __init__(self, wv, th, name=None, interp_kind='linear'):
        ''' '''
        from scipy.interpolate import interp1d
        assert np.asarray(wv).shape == np.asarray(th).shape, 'Initialization Error: `wv` and `th` must have the same shape.'
        wv *= units.dimensionless_unscaled
        th *= units.dimensionless_unscaled
        self._xunit = wv.unit
        self._data = Spectrum(wv, th, bounded=False, fill_value=0., kind=interp_kind)
        self.name = name
        self._interp_kind = interp_kind
        self._d_wv = np.gradient(wv)*wv.unit
        self._FWHM, self._center = self._calc_FWHM()
        self._rectw = self._calc_rectw()
        self._equvw = self._calc_equvw()
        self._peak = self._data.x[self._data.y.argmax()]*self._xunit
        self._bandw = self._calc_bandw()

    def __call__(self, wv, spec):
        '''Return filtered spectrum.'''
        return self.passthrough(wv, spec)

    def __str__(self):
        keys = (['Name'.rjust(18), self.name],
                #['Average wavelength'.rjust(18), self._avgwv],
                #['Pivot wavelength'.rjust(18), self._pivwv],
                #['Mean log wavelength'.rjust(18), self._barlam],
                ['RMS bandwidth'.rjust(18), self._bandw],
                ['Equivalent wv'.rjust(18), self._equvw],
                ['Peak'.rjust(18), self._peak],
                ['Center'.rjust(18), self._center],
                ['FWHM'.rjust(18), self._FWHM],
                ['Rectangular width'.rjust(18), self._rectw])
        part = ['{0}: {1}'.format(k,v) for k, v in keys]
        return '\n'.join(part)

    def _calc_bandw(self):
        barlam = self.BarLam(self._data.x, np.ones_like(self._data.x))
        return barlam*np.sqrt((self._data.y*np.log(self._data.x/barlam)**2/self._data.x*self._d_wv).sum()/(self._data.y/self._data.x*self._d_wv).sum())*self._xunit

    def _calc_FWHM(self):
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        t = self._data.y
        w = self._data.x
        f = interp1d(w, t/t.max()-0.5)
        w1 = brentq(f, w.min(), w[t.argmax()])
        w2 = brentq(f, w[t.argmax()], w.max())
        return (w2-w1)*self._unit, (w2+w1)/2*self._unit

    def _calc_rectw(self):
        return (self._data.y*self._d_wv).sum()/self._data.y.max()

    def _calc_equvw(self):
        return (self._data.y*self._d_wv).sum()

    @property
    def unit(self):
        return self._unit
    @unit.setter
    def unit(self, value):
        value = units.Unit(value)
        assert value.is_equivalent(self._unit), 'Wrong unit.'
        self._data.x = (self._data.x*self._unit).to(value).value
        self._bandw = self._bandw.to(value)
        self._FWHM = self._FWHM.to(value)
        self._rectw = self._rectw.to(value)
        self._equvw = self._equvw.to(value)
        self._peak = self._peak.to(value)
        self._center = self._center.to(value)
        self._unit = value

    @property
    def FWHM(self):
        return self._FWHM

    @property
    def rectw(self):
        return self._rectw

    @property
    def center(self):
        return self._center

    @property
    def peak(self):
        return self._peak

    @property
    def data(self):
        return self._throughput

    def Plot(self, **kwarg):
        '''Plot the filter throughput curve.'''
        self._data.Plot(**kwarg)

    def Filt(self, wv, flux):
        '''Return filtered spectrum.'''

        if isinstance(wv, units.Quantity):
            wv = wv.to(self._unit).value
        else:
            wv = np.asarray(wv)
        outflux = np.empty_like(flux)
        out = (wv < self._data.x.min()) | (wv > self._data.x.max())
        outflux[out] = 0
        outflux[~out] = self._data(wv[~out])*flux[~out]
        return outflux

    def AvgWv(self, wv, flux):
        flux1 = self.Pass(wv, flux)
        d_wv = np.gradient(wv)
        return (flux1*wv*d_wv).sum()/(flux1*d_wv).sum()

    def PivWv(self, wv, flux):
        flux1 = self.Pass(wv, flux)
        d_wv = np.gradient(wv)
        return np.sqrt((flux1*wv*d_wv).sum()/(flux1/wv*d_wv).sum())

    def BarLam(self, wv, flux):
        flux1 = self.Pass(wv, flux)
        d_wv = np.gradient(wv)
        return np.exp((flux1*np.log(wv)/wv*d_wv).sum()/(flux1/wv*d_wv).sum())

    def EqFlux(self, wv, flux):
        d_wv = np.gradient(wv)
        return (self.Pass(wv, flux)*d_wv).sum()/(self.Pass(wv, np.ones_like(wv))*d_wv).sum()


#######################################

'''
Image processing package.

Module dependency
-----------------
numpy, scipy.interpolate, scipy.ndimage

History
-------
8/23/2013, started by JYL @PSI
'''

def shift(im, dx, **kwarg):
    '''
 Shift an array.  Uses numpy.roll for integer unit shift for speed,
 and uses scipy.ndimage.shift for arbitrary unit shift.

 Parameters
 ----------
 im : array-like, numbers
   Input array, dimention >= 1.
 dx : number or sequence of numbers
   Number of units to be shifted.  Positive values mean shifting
   towards the direction of increasing index.  See scipy.ndimage.shift
 mode : str, optional, default 'wrap'
   See scipy.ndimage.shift
 **kwarg : other keywords accepted by scipy.ndimage.shift

 Returns
 -------
 ndarray of the same kind and same type of input array
   The shifted array.  For integer unit shift, the elements shifted
   out of the bounds will be rotated back to the other side.  For
   fractional units shift, see scipy.ndimage.shift

 History
 -------
 v1.0.0 : JYL @PSI, Dec 22, 2013
    '''

    im = np.asarray(im)

    if not hasattr(dx,'__iter__'):
        sh = np.repeat(dx,im.ndim)
    else:
        sh = np.asarray(dx)

    if sh.shape[0] != im.ndim:
        raise RuntimeError('shift must have length equal to input rank')

    # integer shifts
    if (sh % 1 == 0).all():
        out = im.copy()
        for i in range(im.ndim):
            out = np.roll(out, int(sh[i]), axis=i)
        return out

    # fractional shift
    mode = kwarg.pop('mode','wrap')
    import scipy.ndimage
    return scipy.ndimage.shift(im, sh, mode=mode, **kwarg)


def rebin(arr, bin, axis=0, mean=False, median=False, weight=None):
    '''Rebin an array

 Parameters
 ----------
 arr  : array-like
   Array to be binned, can be of any dimention greater than 1
 bin : positive integer, or sequence of them
   Bin size (number of elements).
   If the size of the array along a dimention is not integer times of
   the bin size, then the last pixel of output array will be the sum
   of remaining pixels.  If more elements in `bin` are available, the
   redundant elements will be ignored.
 axis : integer, or sequence of them, optional
   The axis or axes corresponding to the elements in bin.  If
   len(bin) > len(axis), the bins in `bin` without corresponding
   `axis` will be continuously following the last element in `axis`.
   If len(bin) < len(axis), the redundant elements in `axis` will
   simply be ignored.
 mean : bool, optional
   If True, then the elements in output array will be the average of
   binned elements in the input.
 weight : same as arr, optional
   Weight of binning.

 Returns
 -------
 Return a binned numpy array.
 If `weight` is not `None`, then the error array of binned array will also
 be returned.

 v1.0.0 : JYL @PSI, Sept 2013.
 v1.0.1 : JYL @PSI, Nov 18, 2013.
   Add parameter `axis` to allow bin along arbitrary axis
   Slight adjustment of program structure
 v1.0.2 : JYL @PSI, December 1, 2014.
   Added keyword `weight`.
 v1.0.3 : JYL @PSI, May 18, 2016
   Added keyword 'median'
   Corrected a bug for weighted binning
    '''

    if weight is None:

        arr = np.asanyarray(arr)
        if arr.ndim<1:
            raise IOError('Input array needs to have at least one dimension.')

        bi = np.asarray(bin).flatten()
        nb = len(bi)
        if (bi<1).any():
            raise ValueError('Bin sizes need to be positive numbers.')

        if hasattr(axis,'__iter__'):
            ax = axis[:]
        else:
            ax = [axis]
        na = len(ax)
        if nb > na:
            ax.extend([ax[-1]+i+1 for i in range(nb-na)])

        # loop through all axes to be binned.
        inarr = arr.copy()
        for b, a in zip(bi[0:arr.ndim], ax[0:arr.ndim]):
            if b != 1:
                sh = np.asarray(inarr.shape)
                i0s = list(range(0,sh[a],b))
                sh[a] = len(i0s)
                i1s = list(range(sh[a]))
                newarr = np.empty(sh,dtype=inarr.dtype).view(type(inarr))
                arr_v, new_v = np.rollaxis(inarr,a), np.rollaxis(newarr,a)

                if mean:
                    for i0, i1 in zip(i0s[:-1], i1s[:-1]):
                        new_v[i1] = arr_v[i0:i0+b].mean(0)
                    new_v[-1] = arr_v[i0s[-1]:].mean(0)
                elif median:
                    for i0, i1 in zip(i0s[:-1], i1s[:-1]):
                        if hasattr(arr_v, 'median'):
                            new_v[i1] = arr_v[i0:i0+b].median(0)
                        else:
                            new_v[i1] = np.median(arr_v[i0:i0+b],0)
                    if hasattr(arr_v, 'median'):
                        new_v[-1] = arr_v[i0s[-1]:].median(0)
                    else:
                        new_v[-1] = np.median(arr_v[i0s[-1]:],0)
                else:
                    for i0, i1 in zip(i0s[:-1], i1s[:-1]):
                        new_v[i1] = arr_v[i0:i0+b].sum(0)
                    new_v[-1] = arr_v[i0s[-1]:].sum(0)

                #for i0, i1 in zip(i0s[:-1], i1s[:-1]):
                #   new_v[i1] = arr_v[i0:i0+b].sum(0)
                #if mean:
                #   new_v /= b
                #   new_v[-1] = arr_v[i0s[-1]:].mean(0)
                #else:
                #   new_v[-1] = arr_v[i0s[-1]:].sum(0)
            else:
                newarr = inarr
            inarr = newarr

        return newarr

    else:

        y = rebin(arr*weight, bin, axis=axis, mean=mean, median=median)
        e = rebin(weight, bin, axis=axis, mean=mean, median=median)
        if mean:
            return y/e, e
        else:
            return y, e


def rot(im, ang, mag=1.0, center=None, missing=0., pivot=False, order=3, method='linear', version='1.1'):
    '''
 Rotate an image.  This function is similar to IDL rot

 Parameters
 ----------
 im   : array-like, numbers
   Input image
 ang  : number
   Angle to be rotated counter-clockwise [deg]
 mag  : number, optional
   Magnification factor
 center : tuple of two numbers, optional
   Pixel center position for rotation (y, x).  Default is the center of
   image, i.e., index (size-1)/2.
 missing : number, optional
   If provided, the value to use for points outside of the
   interpolation domain.  If set to NONE, values outside the domain
   are extrapolated.  Default is to fill with 0.
 pivot : bool, optional
   If True, then the (x0,y0) rotation center will be mapped to the
   same location in the rotated image.  Oterwise it is mappted to
   the center of rotated image.
 order : int in [0,6), optional
   The order of spline interpolation, in the range of [0,5].  Only
   relevant to v1.0.  See
   scipy.ndimage.interpolation.geometric_transform
 method : str: 'linear', 'nearest', 'splinef2d', optional
   Interpolation method.  Only relevant to v1.1.  See interpn
 version : str, '1.0' or others, optional
   Select program version.  This is mostly used for benchmark during
   development.

 Returns
 -------
 Rotated image in a numpy array

 Notes
 -----
 1. This is a wrapper for scipy.ndimage.interpolation.geometric_transform
 to rotate input image and simulate the interface of IDL rot.
 2. This routine runs much slower (~80x)
 scipy.ndimage.interpolation.rotate, which in turn seems to be slower
 than IDL rot.

 v1.0.0 : JYL @PSI, Aug 23, 2013.
 v1.1.0 : JYL @PSI, June 5, 2014.
   Use scipy.interpolate.interpn to perform 2-D interpolation.  This
   function is 30x faster than
   scipy.ndimage.interpolation.geometric_transform
    '''

    im0 = np.asarray(im)
    sz = np.asarray(im0.shape)
    if center == None:
        center = (sz-1.)/2.

    def rot_xform(coords, ang, mag, size, center, pivot):
        yy, xx = coords
        ys, xs = size
        y0, x0 = center

        if pivot:
            xx, yy = xx-x0, yy-y0
        else:
            xx, yy = xx-(xs-1.)/2., yy-(ys-1.)/2.

        cosa = np.cos(ang*np.pi/180)
        sina = np.sin(ang*np.pi/180)
        xx0, yy0 = (xx*cosa+yy*sina)/mag, (-xx*sina+yy*cosa)/mag

        xx0, yy0 = xx0+x0, yy0+y0

        return (yy0, xx0)

    if method == 'old':

        from scipy.ndimage.interpolation import geometric_transform
        out = geometric_transform(im0, rot_xform, \
            extra_arguments=(ang, mag, sz, center, pivot), order=order, \
            cval=missing)

        return out

    else:

        points = (np.arange(sz[0]),np.arange(sz[1]))

        coord = makenxy(0,sz[0]-1,sz[0],0,sz[1]-1,sz[1])
        yy0, xx0 = rot_xform(coord, ang, mag, sz, center, pivot)
        xi = np.array((yy0.flatten(),xx0.flatten())).T

        from scipy.interpolate import interpn
        return interpn(points, im, xi, method=method, bounds_error=False, fill_value=np.asarray(missing).astype(im.dtype)).reshape(sz)


def xy2rt(im, center, rastep=1., ramax=None, azstep=1., order=3, missing=0., method='linear', version='1.1'):
    '''
 Reproject the input image from rectangular coordinates to polar
 coordinates (theta, r)

 Parameters
 ----------
 im : array-like, number
   Input image
 center : sequence of two numbers
   The center of input image, in vertical and horizontal directions
 rastep, azstep : number, optional
   The step sizes in radial and azimuthal directions.  Default is 1
   pixel and 1 degree, respectively.
 ramax : number, optional
   The maximum radial distance from the center in output image.
   Default is the maximum integer pixel distance from the center
   in the image
 missing : number, optional
   The number used to fill in missing pixels in output.  See the
   cval keyword of geometric_transform.
 order : integer [0,5], optional
   The order of spline interpolation.  Only relevant to v1.0.  See
   geometric_transform.
 method : str: 'linear', 'nearest', or 'splinef2d'.  Only relevant to
   v1.1.  See interpn
 version : str, '1.0' or others
   Select version.  This is mostly used for benchmark during
   development.

 Returns
 -------
 Reprojected image in polar coordinate, with horizontal direction as
 azimuthal axis, and verticle direction as radial axis.  The azimuthal
 direction starts from PA=0, i.e., up direction, increases ccw
 (towards left).

 v1.0.0 : JYL @PSI, Aug 25, 2013
 v1.1.0 : JYL @PSI, June 5, 2014
   Use scipy.interpolate.interpn to perform 2-D interpolation.  This
   function is 13x faster than
   scipy.ndimage.interpolation.geometric_transform
 v1.1.1 : JYL @PSI, 5/6/2016
   Removed keywords `rabin' and `azbin', replace by `rastep' and `azstep'.
    '''
    (yc, xc) = center
    im0 = np.asarray(im)
    sz = im0.shape

    if ramax == None:
        cor = np.array([[0, 0], [sz[0], 0], [sz[0], sz[1]], [0, sz[1]]])
        ramax = np.floor(np.sqrt(((cor-(yc,xc))**2).sum(axis=1)).max())

    rabin = int(round(ramax/rastep+1))
    azbin = int(round(360./azstep))

    def xy2rt_xform(xxx_todo_changeme, xxx_todo_changeme1, rastep, azstep):
        (ra, az) = xxx_todo_changeme
        (yc, xc) = xxx_todo_changeme1
        return ra*rastep*np.sin(np.deg2rad(az*azstep+90))+yc, ra*rastep*np.cos(np.deg2rad(az*azstep+90))+xc

    if version == '1.0':
        from scipy.ndimage.interpolation import geometric_transform
        return geometric_transform(im0, xy2rt_xform, extra_arguments=((yc, xc), rastep, azstep), order=order, cval=missing, output_shape=(rabin,azbin))

    else:
        points = (np.arange(sz[0]),np.arange(sz[1]))

        coord = makenxy(0,rabin+1,rabin,0,azbin,azbin)
        yy0, xx0 = xy2rt_xform(coord, (yc, xc), rastep, azstep)
        xi = np.array((yy0.flatten(), xx0.flatten())).T

        from scipy.interpolate import interpn
        return interpn(points, im, xi, method=method, bounds_error=False, fill_value=np.asarray(missing).astype(im.dtype)).reshape((rabin,azbin))


def feature_pa(im, xxx_todo_changeme3, pa0, r_lim=[5,100], r_step=1, pa_res=1., pa_lim=50., **kwargs):
    '''Measure the position angle of a coma feature from an image of a comet

    v1.0.0 : 5/6/2016, JYL @PSI
    '''
    (yc, xc) = xxx_todo_changeme3
    kwargs.pop('azstep', None)
    kwargs.pop('rastep', None)
    kwargs['rastep'] = r_step
    imr = xy2rt(np.asarray(im), (yc, xc), azstep=pa_res, **kwargs)
    if im.dtype.names is not None:
        if 'error' in im.dtype.names:
            ime = xy2rt(im['error'], (yc, xc), azstep=pa_res, **kwargs)
    elif hasattr(im, 'uncertainty'):
        ime = xy2rt(im.uncertainty.array, (yc, xc), azstep=pa_res, **kwargs)
    else:
        ime = np.ones_like(imr)
    if im.dtype.names is not None:
        if 'mask' in im.dtype.names:
            msk = xy2rt(im['mask'], (yc, xc), azstep=pa_res, **kwargs)
    elif hasattr(im, 'mask'):
        msk = xy2rt(im.mask, (yc, xc), azstep=pa_res, **kwargs)
    else:
        msk = np.zeros_like(imr,dtype=bool)

    r1 = int(np.round(r_lim[0]/float(r_step)-1))
    r2 = int(np.round(r_lim[1]/float(r_step)-1))

    pas = []
    methods = []
    for r in range(r1, r2):
        pa = pa0
        for pal in [160., pa_lim]:
            i1 = np.round((pa-pal/2.)/pa_res)
            i2 = np.round((pa+pal/2.)/pa_res)+1
            if i1<0:
                di = i1
                i1, i2 = 0, i2-i1
            elif i2*pa_res>360:
                di = (i2*pa_res-360)/pa_res
                i1, i2 = 360/pa_res-(i2-i1)-1, 360/pa_res-1
            else:
                di = 0
            i1 = int(i1)
            i2 = int(i2)
            di = int(di)
            ts = np.arange(i1, i2)*pa_res
            if di != 0:
                imr1 = np.roll(imr, -di, axis=1)
                ime1 = np.roll(ime, -di, axis=1)
                msk1 = np.roll(msk, -di, axis=1)
            else:
                imr1 = imr
                ime1 = ime
                msk1 = msk

            v = imr1[r,i1:i2]
            e = ime1[r,i1:i2]
            m = msk1[r,i1:i2]

            try:
                p = gaussfit(ts, v, e, 5)[0]
                f = lambda x: -p[0]/p[2]**2*(x-p[1])*np.exp(-(x-p[1])**2/(2*p[2]**2))+p[4]
                import scipy
                rpk0 = ts[gauss(ts, *p).argmax()]
                rpk = scipy.optimize.newton(f, rpk0)
                mod = gauss(ts, *p)
                m = 0
            except:
                p = np.polyfit(ts, v, 4, w=1/e)
                rpk = np.roots(np.poly1d(p).deriv())
                rpk = rpk.real[rpk.imag==0]
                rpk0 = ts[np.poly1d(p)(ts).argmax()]
                rpk = rpk[abs(rpk-rpk0).argmin()]
                mod = np.poly1d(p)(ts)
                m = 1

#           from matplotlib.pyplot import errorbar, plot
#           errorbar(ts, v, e)
#           plot(ts, mod,'--')

            if abs(rpk-rpk0)>(pa_res*2):
                rpk = np.nan
            if np.isnan(rpk):
                break
            rpk += di*pa_res
            pa = rpk
#           print rpk
        pas.append(rpk)
        methods.append(m)

    pas = np.ma.masked_array(pas, np.isnan(pas))

    return np.stack([pas,methods])


def background(im, ext=0, region=None, std=False, method='mean', plot=False):
    '''Measure the background of an image.

 Parameters
 ----------
 im : string or array-like
   Input image.  If string, then it is taken as the name of a FITS
   image.
 ext : integer, optional
   If the image is in a FITS file, then this keyword specifies the
   index of extension of the image.
 region: sequence of numbers with four elements, (y1, x1, y2, x2)
   The region in the image to be measured.
 std : bool, optional
   If `True`, then returns a tuple of (background, standard deviation)
   If `method` is 'median', then `std` is ignored
 method : str in ['mean', 'gauss','median'], optional
   Choose the method to measure background.  'mean' method uses
   resistant mean, rather than simple mean.  'gauss' uses a Gaussian
   fit to the histogram of image (region) to estimate the background
   and standard deviation
 plot : bool, optional
   Plot the histogram of images within range.  Only applicable when
   `method` is 'gauss'.

 Returns
 -------
 Background value, or tuple of (background, stddev) if std is set True

 v1.0.0 : JYL @PSI, Nov 15, 2013
 v1.0.1 : JYL @PSI, Nov 19, 2013
   Corrected a bug causing the measurement of input array to fail.
   Removed parameter `range`.
   Replaced parameter `gauss` with `method`.
 v1.0.2 : JYL @PSI, Feb 8, 2017
   Bug fix for method='median'
    '''

    if not isinstance(im, (str,bytes)):
        if hasattr(im, '__iter__') and (isinstance(im[0],(str,bytes)) or (np.asarray(im).ndim is 3)):
            # recursively calculate background for each image
            if not std:
                return [background(i, ext=ext, region=region, method=method) for i in im]
            else:
                bg, st = [], []
                for i in im:
                    tmp = background(i, ext=ext, region=region, std=True, method=method)
                    if method is 'median':
                        bg.append(tmp)
                    else:
                        bg.append(tmp[0])
                        st.append(tmp[1])
                if method is 'median':
                    return bg
                else:
                    return bg, st
        else:
            raise ValueError('string type or string iterable or 3-D array expected, {0} received'.format(type(im)))

    if isinstance(im, (str,bytes)):
        img = fits.getdata(im,ext).astype(np.float32)
    else:
        img = np.asarray(im)

    if region is not None:
        img = img[region[0]:region[2],region[1]:region[3]]

    # resistent mean method
    if method is 'mean':
        if std:
            return resmean(img.flatten(), std=True)
        else:
            return resmean(img.flatten())

    # median method
    if method is 'median':
        return np.median(img.flatten())

    # gaussian fit method
    hist, bin = np.histogram(img.flatten(), bins=100, range=[res[0]-10*res[1],res[0]+10*res[1]])
    par0 = list(res)
    par0.insert(0,max(hist))
    x = (bin[0:-1]+bin[1:])/2
    par, sigma, chisq, yfit = gaussfit(x, hist,par0=par0)
    if plot:
        fig = plt.gcf()
        ax = fig.add_subplot(111)
        ax.plot(x,hist)
        ax.vlines(par[1],0,hist.max())
        ax.hlines(hist.max()/2,par[1]-par[2],par[1]+par[2])
        if range is not None:
            ax.set_xlim(range)
        plt.draw()
    if not std:
        return par[1]
    else:
        return par[1], par[2]


def readfits(imfile, ext=0, verbose=True, header=False):
    '''IDL readfits emulator.

 Parameters
 ----------
 imfile : string, or list of strings
   FITS file name(s)
 ext : non-negative integer, optional
   The extension to be read in
 verbose : bool, optional
   Suppress the screen print of FITS information if False
 header : bool, optional
   If `True`, then (image, header) tuple will be returned

 Returns
 -------
 image, or tuple : (image, header)
   image : ndarray or list of ndarray of float32
   header : astropy.io.fits.header.Header instance, or list of it

 v1.0.0 : JYL @PSI, Nov 17, 2013
 v1.0.1 : JYL @PSI, 5/26/2015
   Accept extension name for `ext`.
   Return the actual header instead of `None` even if extension
     contains no data
   Returned data retains the original data type in fits.
    '''

    if isinstance(imfile, (str,bytes)):
        fitsfile = fits.open(imfile)
        if verbose:
            fitsfile.info()

        try:
            extindex = fitsfile.index_of(ext)
        except KeyError:
            print()
            print(('Error: Extension {0} not found'.format(ext)))
            if header:
                return None, None
            else:
                return None

        if extindex >= len(fitsfile):
            print()
            print(('Error: Requested extension number {0} does not exist'.format(extindex)))
            img, hdr = None, None
        else:
            hdr = fitsfile[extindex].header
            if fitsfile[extindex].data is None:
                print()
                print(('Error: Extension {0} contains no image'.format(ext)))
                img = None
            else:
                img = fitsfile[extindex].data

        fitsfile.close()
        if header:
            return img, hdr
        else:
            return img
    elif hasattr(imfile,'__iter__'):

        img = [readfits(f, ext=ext, verbose=verbose) for f in imfile]
        if header:
            return img, headfits(imfile, ext=ext, verbose=verbose)
        else:
            return img

    else:
        raise TypeError('str or list of str expected, {0} received'.format(type(imfile)))


def writefits(imfile, data=None, header=None, name=None, append=False, clobber=False, overwrite=False):
    '''IDL writefits emulator'''
    if clobber:
        warnings.warn('"clobber" was deprecated and will be removed in the future. Use argument "overwrite" instead.', jylipyDeprecationWarning,stacklevel=2)
        overwrite=True
    if append:
        hdu = fits.ImageHDU(data, header=header, name=name)
        hdulist = fits.open(imfile)
        hdulist.append(hdu)
        hdulist.writeto(imfile, overwrite=True)
    else:
        hdu = fits.PrimaryHDU(data, header=header)
        hdu.writeto(imfile, overwrite=overwrite)


def headfits(imfile, ext=0, verbose=True):
    '''IDL headfits emulator.

 Parameters
 ----------
 imfile : string, or list of string
   FITS file name(s)
 ext : non-negative integer, optional
   The extension to be read in
 verbose : bool, optional
   Suppress the screen print of FITS information if False

 Returns
 -------
 astropy.io.fits.header.Header instance, or list of it

 v1.0.0 : JYL @PSI, Nov 17, 2013
    '''

    if is_iterable(imfile):
        return [headfits(f,ext=ext,verbose=verbose) for f in imfile]
    elif not isinstance(imfile, (str, bytes)):
        raise ValueError('string types or iterable expected, {0} received'.format(type(imfile)))

    fitsfile = fits.open(imfile)
    if verbose:
        fitsfile.info()

    if ext >= len(fitsfile):
        print()
        print(('Error: Extension '+repr(ext)+' does not exist!'))
        return None

    if fitsfile[ext].data is None:
        print()
        print(('Error: Extension '+repr(ext)+' contains no image!'))
        return None

    return fitsfile[ext].header


def crop(im, sz, ct, ext=0, out=None, padmode='constant', fill_value=0., **kwarg):
    '''Crop input image or images for given center and size

 Parameters
 ----------
 im : array-like, 2-D image, 3-D image cube, string(s), file(s)
   Input images to be cropped
 sz : 2-element array-like
   Size of output images (y-size, x-size).  NOTE: the size of output
   images can be off by 1 pixel
 ct : (ycenter, ycenter)
   Centers of images
 ext : Int
   In case of FITS images as input, this keyword specifies the
   FITS extension contianing the input image(s)
 out : array-like string or file type
   Output file names
 padmode : str, optional
   See numpy.pad
 fill_value : number, optional
   See numpy.pad keyword `constant_values`.  Note that the same value
   will be used here for all padding areas.
 **kwarg : other optional keywords of numpy.pad

 Returns
 -------
 Image or list of images, containing the cropped images

 v1.0.0 : JYL @PSI, April 22, 2014
 v1.0.1 : JYL @PSI, June 5, 2014
   Corrected a bug resulting in an error in handling single image in
   an array
 v1.1.0 : JYL @PSI, December 2, 2014
   * Added the capacity to process the case when padding to the input
   image is needed.  This update increases the robustness.
   * Restructured the program
 NOTE: Realized that the program is badly conceived.  Need to do major
 revision to remove ambiguities.
    '''

    sz = np.asarray(sz)
    ct = np.asarray(ct)
    y1, x1 = np.round(ct-sz/2.)
    y2, x2 = np.asarray([y1, x1])+sz
    by = max([-y1, 0])
    bx = max([-x1, 0])

    def padhdr(hdr):
        hdr['y1'] = (y1, 'Start Y index (0-based)')
        hdr['y2'] = (y2-1, 'End Y index (0-based)')
        hdr['x1'] = (x1, 'Start X index (0-based)')
        hdr['x2'] = (x2-1, 'End X index (0-based)')
        return hdr

    def padcrop(im):
        if max([bx, by, ax, ay]) > 0:
            kwarg['mode'] = padmode
            if padmode == 'constant':
                fill = ((fill_value, fill_value), (fill_value, fill_value))
                kwarg['mode'] = 'constant'
                kwarg['constant_values'] = fill
            imc = np.pad(im, ((by, ay), (bx, ax)), **kwarg)
            yy1 = np.where(by==0, y1, 0)
            xx1 = np.where(bx==0, x1, 0)
            yy2, xx2 = np.array([yy1, xx1])+sz
            imc = imc[yy1:yy2, xx1:xx2]
        else:
            imc = im[y1:y2, x1:x2]
        return imc

    if not hasattr(im, '__iter__'):
        if isinstance(im, (str,bytes)):
            img = readfits(im, ext=ext, verbose=False)
        elif hasattr(im, '__array__'):
            img = np.array(im)
        else:
            raise TypeError('Input image error')
        imsz = img.shape
        ay = max([y2-imsz[0], 0])
        ax = max([x2-imsz[1], 0])
        imc = padcrop(img)
        if out is not None:
            hdr = padhdr(fits.Header())
            fits.writeto(out, imc, hdr)
        return imc
    else:
        crp = []
        if out is not None:
            j = 0
        if np.asarray(im).ndim == 2:
            img = np.asarray(im)
            imsz = img.shape
            ay = max([y2-imsz[0], 0])
            ax = max([x2-imsz[1], 0])
            imc = padcrop(img)
            if out is not None:
                hdr = padhdr(fits.Header())
                fits.writeto(out[j], imc, hdr)
            return imc
        else:
            for i in im:
                if isinstance(i, str):
                    img = readfits(i, ext=ext, verbose=False)
                else:
                    img = np.asarray(i)
                imsz = img.shape
                ay = max([y2-imsz[0], 0])
                ax = max([x2-imsz[1], 0])
                imc = padcrop(img)
                crp.append(imc)
                if out is not None:
                    hdr = padhdr(fits.Header())
                    fits.writeto(out[j], imc, hdr)
                    j += 1

        return crp


def makenxy(y1, y2, ny, x1, x2, nx, rot=None):
    '''Make 2-d y and x coordinate arrays of specified dimensions
  (Like IDL JHU/APL makenxy.pro)

 Parameters
 ----------
 y1, y2 : float
   Min and max coordinate of the first dimension in output array
 ny : float
   Number of steps in the first dimension
 x1, x2 : float
   Min and max coordinate of the second dimension in output array
 nx : float
   Number of steps in the second dimension
 rot : float
   Rotation of arrays

 Returns
 -------
 yarray, xarray : 2-D arrays

 v1.0.0 : JYL @PSI, June 2, 2014
    '''

    y, x = np.indices((ny,nx), float)
    y = y*(y2-y1)/(ny-1)+y1
    x = x*(x2-x1)/(nx-1)+x1

    if rot is not None:
        m = rotm(rot)
        x, y = m[0,0]*x+m[0,1]*y, m[1,0]*x+m[1,1]*y

    return np.array((y, x))


def dist(y1, y2, ny, x1, x2, nx, rot=None):
    '''
 Generate a distance array

 See `makenxy`.

 v1.0.0 : JYL @PSI, November 5, 2014
    '''

    xy = makenxy(y1, y2, ny, x1, x2, nx, rot=rot)
    return np.sqrt(xy[0]*xy[0]+xy[1]*xy[1])


def makenrt(shape, center=None):
    '''Make 2-d r and theta (polar) coordinate arrays of specified
  dimensions

 Parameters
 ----------
 shape : tuple of float
   The shape of output array
 center : tuple of float
   The center of output array

 Returns
 -------
 r-array, theta-array : 2-D arrays

 v1.0.0 : JYL @PSI, June 2, 2014
    '''

    y, x = makenxy(0, shape[0]-1, shape[0], 0, shape[1]-1, shape[1])
    if center is None:
        center = (np.asarray(shape)-1.)/2.
    y, x = y-center[0], x-center[1]

    r = np.sqrt(x*x+y*y)
    t = (np.rad2deg(np.arctan2(y, x))+360) % 360

    return np.array((r, t))


def centroiding(im, ext=0, ds9=None, newframe=True, coord='image', refine=False, box=5, verbose=True):
    '''
 Interactively centroiding tool

 Parameters
 ----------
 im : string or string sequence, 2-D or 3-D array-like numbers
   File name, sequence of file names, image, or stack of images.  For
   3-D array-like input, the first dimension is the dimension of stack
 ext : non-negative integer, optional
   The extension to be displayed
 ds9 : ds9 instance, or None, optional
   The target DS9 window.  If `None`, then the first opened DS9
   window will be used, or a new window will be opened if none exists.
 newframe : bool, optional
   If set `False`, then the image will be displayed in the currently
   active frame in DS9, and the previous image will be overwritten.
   By default, a new frame will be created to display the image.
 coord : str in ['image', 'physical', 'fk4', 'fk5', 'icrs', 'galactic',
   'ecliptic']
   Coordinate system used by DS9.  If `refine`=`True`, then this
   keyword is ignored, and the default 'image' is used.
 refine : bool, optional
   If `True`, then the centroid will be refined by mskpy.gcentroid().
 box : number, optional
   Box size for centroid refining.  See mskpy.gcentroid()
 verbose : bool, optional
   If `False`, then all screen output is suppressed.  Note that this
   only suppresses information output.  All error or warning messages
   will still be output to screen.

 Returns
 -------
 [yc, xc] or list of [yc, xc]
 If the centroid is not measured for some images in the list, then
 the corresponding center will be set to `None`.

 v1.0.0 : JYL @PSI, October 31, 2014
    '''

    # Pre-process image list
    if isinstance(im, str):
        ims = [im]
    elif isinstance(im, np.ndarray):
        if im.ndim == 2:
            ims = [im]
        else:
            ims = im
    else:
        ims = im
    if refine:
        coord = 'image'

    # Loop through all images
    nimgs = len(ims)
    if verbose:
        print(('Centroiding %i images from input' % nimgs))
        print()
    cts = [None]*nimgs
    i, j = 0, 0
    retry = False
    while i < nimgs:
        if isinstance(ims[i], str):
            if verbose:
                print(('Image %i in the list: %s.' % (i, ims[i])))
        else:
            if verbose:
                print(('Image %i in the list.' % i))
        if not retry:
            pass
            d = imdisp(ims[i], ext=ext, ds9=ds9, newframe=newframe, verbose=verbose)
        else:
            if verbose:
                print('Retry clicking near the center.')
        retry = False
        ct = d.get('imexam coordinate '+coord).split()
        if len(ct) != 0:
            ct = [float(ct[1]), float(ct[0])]
            if refine:
                if isinstance(ims[i], str):
                    img = readfits(ims[i], ext=ext, verbose=verbose)
                else:
                    img = ims[i]
                cts[i] = centroid(img, ct, box=box, verbose=verbose)
            else:
                cts[i] = ct
            if verbose:
                print(('Centroid at (%.6f, %.6f)' % (cts[i][0], cts[i][1])))
            j += 1
        else:
            # Enter interactive session
            key = eval(input('Center not measured.  Try again? (y/n/q) '))
            if key.lower() in ['y', 'yes']:
                retry = True
                continue
            elif key.lower() in ['n', 'no']:
                i += 1
                continue
            elif key.lower() in ['q', 'quit']:
                break
        print()
        i += 1

    if len(cts) == 1:
        cts = cts[0]
    if verbose:
        print(('%i images measured out of a total of %i' % (j, nimgs)))

    return cts


def centroid(im, center=None, error=None, mask=None, method=0, box=6, tol=0.01, maxiter=50, threshold=None, verbose=False):
    '''Wrapper for photutils.centroiding functions

    Parameters
    ----------
    im : array-like, astropy.nddata.NDData or subclass
      Input image
    center : (y, x), optional
      Preliminary center to start the search
    error : array-like, optional
      Error of the input image.  If `im` is NDData type, then `error` will
      be extracted from NDData.uncertainty.  This keyword overrides the
      uncertainty in NDData.
    mask : array-like bool, optional
      Mask of input image.  If `im` is NDData type, then `mask` will be
      extracted from NDData.mask.  This keyword overrides the mask in NDData.
    method : int or str, optional
      Method of centroiding:
      [0, '2dg', 'gaussian'] - 2-D Gaussian
      [1, 'com'] - Center of mass
      [2, 'geom', 'geometric'] - Geometric center
    box : int, optional
      Box size for the search
    tol : float, optional
      The tolerance in pixels of the center.  Program exits iteration when
      new center differs from the previous iteration less than `tol` or number
      of iteration reaches `maxiter`.
    maxiter : int, optional
      The maximum number of iterations in the search
    threshold : number, optional
      Threshold, only used for method=2
    verbose : bool, optional
      Print out information

    Returns
    -------
    (y, x) as a numpy array

    This program uses photutils.centroids.centroid_2dg() or .centroid_com()

    v1.0.0 : JYL @PSI, Feb 19, 2015
    '''
    from photutils.centroids import centroid_2dg, centroid_com
    if isinstance(im, nddata.NDData):
        if error is None:
            if im.uncertainty is not None:
                error = im.uncertainty.array
        if mask is None:
            if im.mask is not None:
                mask = im.mask
        im = im.data

    if center is None:
        center = np.asarray(im.shape)/2.
    else:
        center = np.asarray(center)
    b2 = box/2
    if (method in [2, 'geom', 'geometric']) and (threshold is None):
        raise ValueError('threshold is not specified')
    if verbose:
        print(('Image provided as a '+str(type(im))+', shape = ', im.shape))
        print(('Centroiding image in {0}x{0} box around ({1},{2})'.format(box,center[0],center[1])))
        print(('Error array '+condition(error is None, 'not ', ' ')+'provided'))
        print(('Mask array '+condition(mask is None, 'not ', ' ')+'provided'))
    i = 0
    delta_center = np.array([1e5,1e5])
    while (i < maxiter) and (delta_center.max() > tol):
        if verbose:
            print(('  iteration {0}, center = ({1},{2})'.format(i, center[0], center[1])))
        p1, p2 = np.floor(center-b2).astype('int'), np.ceil(center+b2).astype('int')
        subim = np.asarray(im[p1[0]:p2[0],p1[1]:p2[1]])
        if error is None:
            suberr = None
        else:
            suberr = np.asarray(error[p1[0]:p2[0],p1[1]:p2[1]])
        if mask is None:
            submask = None
        else:
            submask = np.asarray(mask[p1[0]:p2[0],p1[1]:p2[1]])
        if method in [0, '2dg', 'gaussian']:
            xc, yc = centroid_2dg(subim, error=suberr, mask=submask)
        elif method in [1, 'com']:
            xc, yc = centroid_com(subim, mask=submask)
        elif method in [2, 'geom', 'geometric']:
            xc, yc = geometric_center(subim, threshold, mask=submask)
        else:
            raise ValueError("unrecognized `method` {0} received.  Should be [0, '2dg', 'gaussian'] or [1, 'com']".format(method))
        center1 = np.asarray([yc+p1[0], xc+p1[1]])
        delta_center = abs(center1-center)
        center = center1
        i += 1

    if verbose:
        print(('centroid = ({0},{1})'.format(center[0],center[1])))
    return center


def stack(ims, ext=0, size=None, center=None, mode='mean', verbose=False):
    '''
 Stack input images

 Parameters
 ----------
 ims : list of str or 2-D arrays
   Input image file names or 2-D images
 ext : integer or list of integers
   FITS extension number
 size : (ysize, xsize)
   Size of output image.  Default is the size of the first image in
   the list
 center : (ycenter,xcenter) or list of (ycenter,xcenter)
   Center(s) of input images.  Default is the geometric center of all
   images
 mode : str, optional
   Mode of stacking, either 'mean' or 'median'.
 verbose : bool, optional

 Return
 ------
 image
 The stacked image.

 v1.0.0 : JYL @PSI, December 1, 2014
    '''

    assert mode in 'mean median'.split(), '`mode` must be in [''mean'', ''median''].'

    if isinstance(ims, str):
        return readfits(ims,ext=ext)
    if isinstance(ims, np.ndarray):
        if ims.ndim == 2:
            return im
        else:
            nimgs = ims.shape[0]
    else:
        nimgs = len(ims)

    if ~hasattr(ext, '__iter__'):
        ext = duplicate(ext, nimgs)

    if center is not None:
        if np.array(center).ndim == 1:
            center = [center]*nimgs
    else:
        center = [None]*nimgs

    if size is None:
        if isinstance(ims[0], (str,bytes)):
            sz0 = np.array(fits.open(ims[0])[ext[0]].data.shape)
        else:
            sz0 = np.array(ims[0].shape)
    else:
        sz0 = np.asarray(size)

    if mode == 'mean':
        out = np.zeros(sz0)
    else:
        out = []

    if center[0] is None:
        ct0 = (np.array(sz)-1)/2
    else:
        ct0 = np.array(center[0])

    for i in range(nimgs):
        # Load image
        if isinstance(ims[i], (str,bytes)):
            img = readfits(ims[i], ext=ext[i], verbose=False)
        else:
            img = ims[i]
        sz = np.array(img.shape)

        # Shift image
        if center[i] is None:
            ct = (np.array(img.shape)-1)/2
        else:
            ct = np.asarray(center[i])

        img = shift(img,ct0%1-ct%1)
        img = crop(img, sz0, np.floor(ct)+(ct0%1))

        if mode == 'mean':
            out += img
        else:
            out.append(img)

    if mode == 'mean':
        return out/nimgs
    else:
        #return out
        return np.median(np.array(out), axis=0)



########################################


def linfit(x, y=None, yerr=None, xerr=None, intercept=True, return_all=False):
    '''Linear fit to input data points: y = a + b*x

 Parameters
 ----------
 x, y : array-like, float; y is optional
   Data points to be fitted.  If both supplied, then they need to have
   the same length.  If only x supplied, it must have a dimention with
   length 2, and it will be split as (x, y) before the fit.
 yerr, xerr : scalor or array-like, float, optional
   Measurement errors.  If scalor, then all points have the same
   error bar.  If array, they need to have the same length as input
   data.  Only yerr is implemented.
 intercept : bool, optional
   If False, then the interception of linear fit is set to 0.  Only
   the slope is returned.
 return_all : bool, optional
   If `True`, then the probability q and coefficient of correlation cc
   will be returned.

 Returns
 -------
 Tuple : (par, sigma, chisq), or (par, sigma, chisq, q, cc)
   :par, sigma are 2-element arrays, (a, b) if `intercept`=True, and
     the 1-sigma errors, or float numbers if `intercept`=False
   :chisq, float, reduced chisq, defined as
        ((model - measure)**2).sum()/(n-2) if no yerr is given,
     or (((model - measure)/yerr)**2).sum()/(n-2)
     where n is the length of y
   :q, probability that a value of chisq as poor as the returned chisq
     should occur.  If `yerr`=None, then q is returned as None
   :cc, float, coefficient of correlation.  If `intercept`=False, then
     it is set to None.

 Notes
 -----
 Follow Numerical Recipes in C, \S 15.2 and 15.3

 v1.0.0 : JYL @PSI, Oct 21, 2013
    '''

    if y is None:
        x = np.asarray(x)
        if x.shape[0] == 2:
            x1, y1 = x
        if x.shape[1] == 2:
            x1, y1 = x.T
        else:
            msg = "If only `x` is given as input, it has to be of shape (2, N) or (N, 2), provided shape was %s" % str(x.shape)
            raise ValueError(msg)
    else:
        x1, y1 = np.asarray(x).flatten(), np.asarray(y).flatten()

    n = len(x1)
    if yerr is None:
        yerr1 = np.ones(n)
    else:
        yerr1 = np.asarray(yerr).astype(float).flatten()
    if len(yerr1) is 1:
        yerr1 = np.repeat(yerr1,n)

    if xerr is None:
        xerr1 = np.ones(n)
    else:
        xerr1 = np.asarray(xerr).astype(float).flatten()
    if len(xerr1) is 1:
        xerr1 = np.repeat(xerr1,n)

    yerr2 = yerr1*yerr1
    ss = (1/yerr2).sum()
    sx, sy = (x1/yerr2).sum(), (y1/yerr2).sum()
    sxx, sxy = (x1*x1/yerr2).sum(), (x1*y1/yerr2).sum()
    Delta = ss*sxx - sx*sx
    a, b = (sxx*sy - sx*sxy)/Delta, (ss*sxy - sx*sy)/Delta

    diff = (y1-(a+b*x1))/yerr1
    chisq = (diff*diff).sum()
    redchisq = chisq/(n-2)

    cc = -sx/np.sqrt(ss*sxx)

    siga, sigb = np.sqrt(sxx/Delta), np.sqrt(ss/Delta)
    if yerr is None:
        siga, sigb = (np.array([siga, sigb])*np.sqrt(chisq)).flat
        q = None
    else:
        from scipy.special import gammainc
        q = gammainc((n-2)/2., chisq/2)

    if not return_all:
        return np.array([a, b]), np.array([siga, sigb]), redchisq
    else:
        return np.array([a, b]), np.array([siga, sigb]), redchisq, q, cc


def power(x, par=[1.,-1.]):
    '''Calculate power law y = par[0] * x**par[1]

 Parameters
 ----------
 x : array-like, float
 par : array-like, scalor, float, optional
   If scalor, then the scaling parameter is set to 1.

 v1.0.0 : JYL @PSI, Oct, 2013
    '''
    if np.size(par) is 1:
        p = np.array([1., par])
    else:
        p = np.asarray(par)
    return p[0]*np.asarray(x)**p[1]


def powfit(x, y=None, yerr=None, xerr=None, scale=True, fast=True):
    '''Power law fit to input data points: y = a * x**b

 Parameters
 ----------
 x, y : array-like, float; y is optional
   Data points to be fitted.  If both supplied, then they need to have
   the same length.  If only x supplied, it must have a dimention with
   length 2, and it will be split as (x, y) before the fit.
 yerr, xerr : scalor or array-like, float, optional
   Measurement errors.  If scalor, then all points have the same
   error bar.  If array, they need to have the same length as input
   data.  Only yerr is implemented.
 scale : bool, optional
   If False, then the scaling of the power law is set to 1.  Only the
   power law index is returned.
 fast : bool, optional
   If True, then the fit is performed with linear fit in log-log
   space.  If False, then the fit will be performed with a curve_fit
   to power law.  The speed difference is ~8x.

 Returns
 -------
 Returns a tuple: (par, sigma, chisq)
   :par, sigma are 2-element arrays, (a, b) if `scale`=True, and the
     1-sigma errors, or float numbers if `scale`=False
   :chisq, float, reduced chisq, defined as
        ((model - measure)**2).sum()/(n-2) if no yerr is given,
     or (((model - measure)/yerr)**2).sum()/(n-2)
     where n is the length of y

 v1.0.0 : JYL @PSI, Oct 22, 2013
    '''

    if y is None:
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        if x.shape[1] == 2:
            x, y = x.T
        else:
            msg = "If only `x` is given as input, it has to be of shape (2, N) or (N, 2), provided shape was %s" % str(x.shape)
            raise ValueError(msg)
    else:
        x1, y1 = np.asarray(x).flatten(), np.asarray(y).flatten()

    n = len(x)
    if yerr is None:
        yerr1 = np.ones(n)
    else:
        yerr1 = np.asarray(yerr).astype(float).flatten()
    if len(yerr1) is 1:
        yerr1 = np.repeat(yerr1,n)

    if xerr is None:
        xerr1 = np.ones(n)
    else:
        xerr1 = np.asarray(xerr).astype(float).flatten()
    if len(xerr1) is 1:
        xerr1 = np.repeat(xerr1,n)

    if fast:
        par, sig, chisq = linfit(np.log10(x1), np.log10(y1), yerr1/y1, xerr1/x1, intercept=scale)
        par = np.array([10**par[0], par[1]])
        sig = np.array([par[0]*sig[0], sig[1]])
    else:
        def func(x, a, b):
            return power(x,[a,b])
        from scipy.optimize import curve_fit
        par, sig = curve_fit(func, x1, y1, sigma=yerr1)
        sig = np.sqrt(np.diag(sig))

    res = y1 - power(x,par)
    chisq = (res*res/(yerr1*yerr1)).sum()/(n-2)
    return par, sig, chisq


def gauss(x, *par):
    '''Gauss function with 3-6 parameters.  See `gaussfit''

    v1.0.0 : 5/6/2016, JYL @PSI, extracted from inside gaussfit for
      general use.

    '''
    z = (x-par[1])/par[2]
    g = par[0]*np.exp(-z*z/2)
    if len(par) >= 4:
        g += par[3]
    if len(par) >= 5:
        g += par[4]*x
    if len(par) == 6:
        g += par[5]*x*x
    return g


def gaussfit(x, y=None, yerr=None, nterms=3, par0=None):
    '''Perform Gaussian fit to input data.

 Parameters
 ----------
 x, y : array-like, numbers; y is optional
   Data points to be fitted.  If both supplied, then they need to have
   the same length.  If only x supplied, it must have a dimention with
   length 2, and it will be split as (x, y) before the fit.
 yerr : scalor or array-like, number, optional
   Measurement errors.  If scalor, then all points have the same
   error bar.  If array, they need to have the same length as input
   data.
 nterms : number from 3 to 6, optional
   The number of parameters.  The model is:
     y = A0 * exp(-(x-A1)**2/(2*A2**2)) + A3 + A4 * x + A5 * x**2
   nterms = i for i=3,4,5,6 will return (A0, A1, ..., A_i-1)
 par0 : array-like, number, optional
   Initial guess of parameters.  While this parameter is optional, if
   not specified, some times the fit results are completely wrong.
   This keyword override `nterms'.

 Returns
 -------
 Tuple : (par, sigma, chisq, yfit)

 v1.0.0 : JYL @PSI, Nov 14, 2013
 v1.0.0 : 5/6/2016, JYL @PSI
   Corrected a bug that disabled `nterms' keyword.
    '''

    if y is None:
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        if x.shape[1] == 2:
            x, y = x.T
        else:
            msg = "If only `x` is given as input, it has to be of shape (2, N) or (N, 2), provided shape was %s" % str(x.shape)
            raise ValueError(msg)
    else:
        x, y = np.asarray(x).flatten(), np.asarray(y).flatten()

    if yerr is not None and len(yerr) == 1:
        yerr = np.repeat(len(x1))

    if par0 is None:
        par0 = list(resmean(x,std=True))
        par0.insert(0,y.max())
        if nterms > 3:
            par0.append(0)
        if nterms > 4:
            par0.append(0)

    from scipy.optimize import curve_fit
    par, pcov = curve_fit(gauss, x, y, par0, yerr)
    sigma = np.diag(pcov)
    yfit = gauss(x, *par)
    diff = y-yfit
    if yerr is not None:
        diff /= yerr
    chisq = (diff*diff).sum()/(len(x)-len(par))

    return par, sigma, chisq, yfit


def resmean(x, threshold=3., std=False):
    '''Calculate resistant mean.

 Parameters
 ----------
 x : array-like, numbers
 cut : float, optional
   Threshold for resistant mean
 std : bool, optional
   If set True, then a tuple of (mean, std_dev) will be returned

 Returns
 -------
 float, or tuple of two float

 v1.0.0 : JYL @PSI, Oct 24, 2013
    '''

    x1 = np.asarray(x).flatten()

    while True:
        m, s = x1.mean(), x1.std()
        ww = (x1 <= m+threshold*s) & (x1 >= m-threshold*s)
        x1 = x1[ww]
        if np.abs(x1.std()-s) < 1e-13:
            break

    if std:
        return x1.mean(), x1.std()
    else:
        return x1.mean()


def sum(arr, err=None, axis=None):
    '''Sum array elements with error propagation.

 Parameters
 ----------
 arr : array-like, numbers
   Input array
 err : scalor number, or array-like numbers, optional
   Variance of input array
 axis : interger, or None, optional
   The axis along which the sum is taken.  `None` means sum over
   whole array

 Returns
 -------
 scalor number or numpy array, or tuple (sum, var) if `err` is not
 `None`.

 v1.0.0 : JYL @PSI, Nov 21, 2013
    '''

    if err is None:
        return np.asarray(arr).sum(axis=axis)

    if np.size(err) == 1:
        tmp = np.empty_like(arr)
        tmp[:] = err
        err = tmp

    return np.asarray(arr).sum(axis=axis), np.sqrt(np.asarray(err*err).sum(axis=axis))


def prod(arr, err=None, axis=None):
    '''Multiply array elements with error propagation.

 Parameters
 ----------
 arr : array-like, numbers
   Input array
 err : scalor number, or array-like numbers, optional
   Variance of input array
 axis : interger, or None, optional
   The axis along which the multiplication is taken.  `None` means
   multiply over whole array

 Returns
 -------
 scalor number or numpy array, or tuple (prod, var) if `err` is not
 `None`.

 v1.0.0 : JYL @PSI, Nov 21, 2013
    '''

    p = np.asarray(arr).prod(axis=axis)
    if err is None:
        return p

    if np.size(err) == 1:
        tmp = np.empty_like(arr)
        tmp[:] = err
        err = tmp

    e = err/arr
    return p, np.abs(p)*np.sqrt((e*e).sum(axis=axis))


def add(arr1, arr2, err1=None, err2=None):
    '''Add two arrays with error propagation

 Parameters
 ----------
 arr1, arr2 : array-like numbers
   Arrays to be summed up
 err1, err2 : array-like numbers, optional
   Errors associated with the input arrays

 Returns
 -------
 numpy array, or tuple of two arrays
 If neither `err1` nor `err2` is defined, then return the sum of two
 input arrays.
 If either `err1` or `err2` is defined, then return (sum, err), the
 sum of two arrays and the error that is defined.
 If both `err1` and `err2` are defined, then return (sum, err), the
 sum of two arrays and the error following the error propagation.

 v1.0.0 : JYL @PSI, December 28, 2013
 v1.0.0 : JYL @PSI, December 3, 2014
   * Change from calculating error only when both `err1` and `err2` are
   not `None` to either of them is not `None`.
   * Add conversion with asarray to increase the robustness
    '''

    s = np.asanyarray(arr1) + np.asanyarray(arr2)

    if err1 is None and err2 is None:
        return s

    if err1 is None or err2 is None:
        if err1 is None:
            return s, np.asanyarray(err2)
        if err2 is None:
            return s, np.asanyarray(err1)

    return s, np.sqrt(np.asanyarray(err1)**2+np.asanyarray(err2)**2)


def sub(arr1, arr2, err1=None, err2=None):
    '''Difference between two arrays arr1-arr2 with error propagation.

 Parameters
 ----------
 arr1, arr2 : numbers, array-like
   Two input arrays
 err1, err2 : numbers, array-like
   Error bars of input arrays

 Returns
 -------
 tuple (diff, var)
 Difference and variance of arr1 and arr2: diff = arr1-arr2

 v1.0.0 : JYL @PSI, Nov 21, 2013
 v1.0.0 : JYL @PSI, December 3, 2014
   * Change from calculating error only when both `err1` and `err2` are
   not `None` to either of them is not `None`.
   * Add conversion with asarray to increase the robustness
    '''

    d = np.asanyarray(arr1) - np.asanyarray(arr2)

    if err1 is None and err2 is None:
        return d

    if err1 is None or err2 is None:
        if err1 is None:
            return d, np.asanyarray(err2)
        if err2 is None:
            return d, np.asanyarray(err1)

    return d, np.sqrt(np.asanyarray(err1)**2+np.asanyarray(err2)**2)


def mul(arr1, arr2, err1=None, err2=None):
    '''Multiply two arrays with error propagation

 Parameters
 ----------
 arr1, arr2 : array-like numbers
   Arrays to be summed up
 err1, err2 : array-like numbers, optional
   Errors associated with the input arrays

 Returns
 -------
 numpy array, or tuple of two arrays
 The sum of two input arrays if `err1` or `err2` is `None`
 Or (sum, var) of two input arrays if both `err1` and `err2` are
 defined.

 Notes
 -----
 For zero elements in the product array, their variance is not
 defined, and will return as -1.

 v1.0.0 : JYL @PSI, December 28, 2013
 v1.0.0 : JYL @PSI, December 3, 2014
   * Change from calculating error only when both `err1` and `err2` are
   not `None` to either of them is not `None`.
   * Add conversion with asarray to increase the robustness
    '''

    p = np.asanyarray(arr1)*np.asanyarray(arr2)

    if err1 is None and err2 is None:
        return p

    if err1 is None or err2 is None:
        if err1 is None:
            return p, np.asanyarray(err2)*np.asanyarray(arr1)
        if err2 is None:
            return p, np.asanyarray(err1)*np.asanyarray(arr2)

    err1, err2 = np.asanyarray(err1), np.asanyarray(err2)
    s = -np.ones_like(p)
    nz = p != 0
    s[nz] = np.abs(p[nz])*np.sqrt((err1[nz]/arr1[nz])**2+(err2[nz]/arr2[nz])**2)
    return p, s


def div(arr1, arr2, err1=None, err2=None):
    '''Division between two arrays arr1/arr2 with error propagation.

 Parameters
 ----------
 arr1, arr2 : numbers, array-like
   Two input arrays
 err1, err2 : numbers, array-like
   Error bars of input arrays

 Returns
 -------
 tuple (div, var)
 Division and variance of arr1 and arr2: diff = arr1/arr2

 Notes
 -----
 For zero elements in the product array, their variance is not
 defined, and will return as -1.

 v1.0.0 : JYL @PSI, Nov 21, 2013
 v1.0.0 : JYL @PSI, December 3, 2014
   * Change from calculating error only when both `err1` and `err2` are
   not `None` to either of them is not `None`.
   * Add conversion with asarray to increase the robustness
    '''

    d = np.asarray(arr1)/np.asarray(arr2)

    if err1 is None and err2 is None:
        return d

    if err1 is None or err2 is None:
        if err2 is None:
            return d, np.asarray(err1)/np.asarray(arr2)
        if err1 is None:
            return d, np.asarray(err2)*np.asarray(arr1)/np.asarray(arr2)**2

    err1, err2 = np.asanyarray(err1), np.asanyarray(err2)
    s = -np.ones_like(d)
    nz = d != 0
    s[nz] = np.abs(d[nz])*np.sqrt((err1[nz]/arr1[nz])**2+(err2[nz]/arr2[nz])**2)
    return d, s


def avg(arr, err=None, axis=None, var=False):
    '''Weighted average of input array with error propagation.

 Parameters
 ----------
 arr : array-like numbers
   Input array
 err : scalor number, or array-like numbers, optional
   Variance of input array
 axis : interger, optional
   The axis along which the average is taken.  `None` means for whole
   array
 var : bool, optional
   If `True`, then the variance of average will be returned.  If `err`
   is not provided (unweighted average), then `var` will return the
   standard deviation

 Returns
 -------
 array-like numbers, or tuple (average, variance)

 v1.0.0 : JYL @PSI, Nov 21, 2013
    '''

    if err is None:
        if var:
            return arr.mean(axis=axis), arr.std(axis=axis)
        else:
            return arr.mean(axis=axis)

    if np.size(err) == 1:
        tmp = np.empty_like(arr)
        tmp[:] = err
        err = tmp

    w2 = 1/(err*err)
    if var:
        return (arr*w2).sum(axis=axis)/w2.sum(axis=axis), 1/np.sqrt(w2.sum(axis=axis))
    else:
        return (arr*w2).sum(axis=axis)/w2.sum(axis=axis)


def grc(norm,step=1.):
    '''Calculate the great circle for a normal vector

 Parameters
 ----------
 norm : 3-element or 2-element sequence, float
   The normal vector, can be in (x,y,z) or in (RA, Dec)
 step : float, optional
   The step size of output x and y sequences, in degrees

 Returns
 -------
 2-D array of float of dimension (2 or 3, 360/`step`)
   The (x, y, z) or (RA, Dec) coordinates of points along great
   circle.  The form of vector is the same as `norm`.  If 3-vector,
   then all vectors are normalized to unit length.

 v1.0.0 : JYL @PSI, 6/13/2014
    '''

    if len(norm) == 3:
        norm = xyz2sph(norm)

    theta = np.arange(np.ceil(360/step))
    x = np.cos(np.deg2rad(theta))
    y = np.sin(np.deg2rad(theta))
    z = np.zeros_like(theta)

    vec = eularv(np.array([x, y, z]), 90+norm[0], 90-norm[1], 0.)

    if len(norm) == 2:
        return xyz2sph(vec)[1:,:]
    else:
        return vec


def hgrc(los, pa, step=1., full=False):
    '''Calculate the half great circle in sky determined by
 line-of-sight and position angle

 Parameters
 ----------
 los : array-like, floating point
   The (RA, Dec) [deg], or (x, y, z) of line of sight
 pa  : array-like, floating point
   The position angles to be calculated [deg east of north]
 step : float, optional
   The step size of output x and y sequences, in degrees
 full : bool, optional
   If `True`, returns a full great circle rather than half great
   circle

 Returns
 -------
 2-D array of float of dimensions (2, 360/`step`)
   The (RA, Dec) coordinates of the half great circle determined by
   the `los` and `pa`.

 v1.0.0 : JYL @PSI, 6/13/2014
    '''

    n = paplane(los, pa)
    gc = grc(n, step)
    if full:
        return gc

    pagc = vecpa(los, gc)[0]

    return gc[:,abs(pagc-pa)<5]



##########################################


'''
Utility routines for general use.

Module dependency
-----------------
numpy, astropy.io.fits, scipy.interpolate.interp1d


History
-------
10/1/2013, started by JYL @PSI
'''


def nparr(*args):
    '''Converts input arguments to 1-D numpy arrays.

 nparr1, nparr2, ... = argu1d(arg1, arg2, ...)

 Parameters
 ----------
 Arbitrary number of array-like

 Returns
 -------
 A tuple containing 1-D numpy arrays for each corresponding input
 parameter of the same data type.  If any input argument is a tuple,
 then a tuple will be returned for it with each element in the tuple
 converted to a numpy array recursively.  Note that if any input
 parameter is a 2-D or higher dimention array, it will be flattened.
    '''

    argu = tuple()

    for parm in args:
        if type(parm) == tuple:
            parm1 = nparr(*parm)
        else:
            parm1 = np.asarray(parm).flatten()
        argu += parm1,

    if len(argu) == 1:
        return argu[0]
    else:
        return argu


def write_fitstable(filename, *args, **kwargs):
    '''Write the input array-like data into a FITS table file

 Parameters
 ----------
 filename : string, file object or file-like object
   File to write to.  If a file object, must be opened for append
   (ab+)
 *argu : arbitrary number of array-like
   Columns to be written to the FITS file.
 colnames : array-like, string, optional
   Name of columns to be saved in the output file.
 format : array-like, string, optional
   Format string of columns.  If omitted, all columns are assumed
   to be floating point
 header : iterable, optional
   The header keywords to be added to the primary extension of the
   output file.  See astropy.io.fits.header.
 append : boolean, default is True
    If True, then the new table is appended to the existing file.
    Otherwise the new table will replace the existing file.

 Returns
 -------
 Integer, a status flag.

 v1.0.0 : JYL @PSI, Oct 3, 2013
    '''

    ncol = len(args)
    colnames = kwargs.pop('colnames',['col{0}'.format(i) for i in range(ncol)])
    format = kwargs.pop('format',['E']*ncol)
    units = kwargs.pop('units',[None]*ncol)
    header = kwargs.pop('header',None)
    append = kwargs.pop('append',True)

    cols = [fits.Column(name=n, format=f, array=c, unit=u) for c, n, f, u in zip(args, colnames, format, units)]

    tbhdu = fits.new_table(fits.ColDefs(cols))
    if header != None:
        tbhdu.header.extend(header)

    import os.path
    if append and os.path.exists(filename):
        hdulist = fits.open(filename)
        hdulist.append(tbhdu)
        hdulist.writeto(filename, overwrite=True, **kwargs)
    else:
        pmhdu = fits.PrimaryHDU()
        hdulist = fits.HDUList([pmhdu,tbhdu])
        hdulist.writeto(filename, **kwargs)




def findfile(path, name=None, recursive=False, full=True, dir=False):
    '''Search file in directory.

 Parameters
 ----------
 path : string
   The path to be searched
 name : string or None, optional
   The strings contained in file names
 recursive : bool, optional
   Recursively search all subdirectories
 full : bool, optional
   If `True`, then the full path of files are returned in the list.
 dir : bool, optional
   If `True`, then search for directories rather than files

 Returns
 -------
 List of strings : the name of files (directories)

 v1.0.0 : JYL @PSI, Nov 17, 2013
 v1.0.1 : JYL @PSI, Mar 13, 2014
   Corrected for a bug resulting in double path name for iterative
   search.
 v1.0.2 : JYL @PSI, December 5, 2014
   * Corrected some bugs that cause problems when `full`=False and
   `recursive`=True
   * Added recursive search for the case of `dir`=True
   * Made sure the relative path w/r to `path` is included in the
   returned file names when `recursive`=True and `full`=False.
   * Added `name` matching for directory search
    '''

    import os
    allfiles = os.listdir(path)
    rdir = [f for f in allfiles if os.path.isdir(path+'/'+f)]

    if dir:
        files = rdir[:]
    else:
        files = [f for f in allfiles if f not in rdir]

    if name is not None:
        if dir:
            files = [d for d in rdir if d.find(name)>=0]
        else:
            files = [f for f in files if f.find(name)>0]

    # add full path
    if full:
        files = [path+'/'+f for f in files]

    # recursively search all subdir
    if recursive:
        for d in rdir:
            fs = findfile(path+'/'+d,name=name,recursive=True,full=False, dir=dir)
            fs = [d+'/'+f for f in fs]
            if full:
                fs = [path+'/'+f for f in fs]
            files.extend(fs)

    return files



def duplicate(value, num=2, unit=None):
    '''
    '''

    if num == 1:
        if np.size(value) == 1:
            v = value
        else:
            v = value[0]
    else:
        single = False
        if isinstance(value, units.Quantity):
            if np.size(value) == 1:
                single = True
        else:
            if not hasattr(value, '__iter__'):
                single = True
            elif len(value) == 1:
                single = True
                value = value[0]
        if single:
            v = [value]*num
        else:
            assert len(value) >= num, 'Length of `value` must be greater than or equal to `num`'
            v = value[:num]

    if unit is None:
        return v
    else:
        if (not isinstance(v, units.Quantity)) and hasattr(v, '__iter__'):
            for i in range(len(v)):
                if not isinstance(v[i], units.Quantity):
                    v[i] *= unit
#       return units(v, unit)
        return v


def islengthone(x):
    '''Check if input `x` is length 1 variable'''

    if isinstance(x, np.ndarray):
        if x.size == 1:
            return True
    elif np.isscalar(x):
        return True
    elif len(x) == 1:
        return True
    return False


def isinteger(x):
    '''Check if input scalor 'x' is an integer'''
    import numbers
    return isinstance(x, numbers.Integral)


def condition(cond, expr1, expr2):
    if cond:
        return expr1
    else:
        return expr2


def ulen(x):
    '''Universal replacement for `len`, can accept any type'''
    if islengthone(x):
        return 1
    try:
        out = len(x)
    except:
        try:
            out = np.array(x).shape[0]
        except:
            raise TypeError("I'm out of ideas for {0} of type {1}".format(x, type(x)))
    return out



def latlon2ccdxy_oblate(lats, lons, xc, yc, rpole, reqtr, subelat, subelon, npang, angoffset):

    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)
    subelat, subelon = np.deg2rad(subelat), np.deg2rad(subelon)

    tanlats = np.tan(lats)
    zp1 = tanlats * reqtr/np.sqrt(1.+(tanlats*reqtr/rpole)**2)
    dp1 = np.sqrt(1.-(zp1/rpole)**2)*reqtr

    xp1 = dp1 * np.cos(lons)
    yp1 = dp1 * np.sin(lons)

    mags = np.sqrt((xp1**2+yp1**2+zp1**2).sum())
    mags = 1.
    xp = xp1/mags
    yp = yp1/mags
    zp = zp1/mags

    sd = np.sin(subelat)
    cd = np.cos(subelat)
    sa = np.sin(subelon)
    ca = np.cos(subelon)
    ta = sa/ca

    R = np.array([[-sa, -sd*ca, cd*ca], [ca, -sd*sa, cd*sa], [0., cd, sd]])

    Rinv = R.T

    npts = len(lons)
    vp = np.empty((npts,3))
    for ip in range(npts):
        vp[ip,:] = Rinv.dot([xp[ip],yp[ip],zp[ip]])

    #angsepcos = np.sin(lats)*sd + np.cos(lats)*cd*np.cos(subelon-lons)
    #print angsepcos.shape
    angsep = vecsep(np.rad2deg([subelon,subelat]), np.rad2deg(np.array([lons,lats])))
    indisk = angsep < 90.

    beta = -np.deg2rad(npang)
    cb = np.cos(beta)
    sb = np.sin(beta)

    xpn = cb*vp[:,0]+sb*vp[:,1]
    ypn = -sb*vp[:,0]+cb*vp[:,1]

    xpts = xpn+xc
    ypts = ypn+yc

    return xpts, ypts, indisk


def ccdxy2latlon_oblate(xpts, ypts, xc, yc, rpole, reqtr, subelat, subelon, npang, angoffset):

    subelat, subelon = np.deg2rad([subelat, subelon])

    npts = len(xpts)
    beta = np.deg2rad(npang)
    cb, sb = np.cos(beta), np.sin(beta)

    xnorm, ynorm = xpts-xc, ypts-yc

    xp = cb*xnorm+sb*ynorm
    yp = -sb*xnorm+cb*ynorm

    ap = np.zeros_like(xp)
    cp = ap.copy()
    zp = ap.copy()

    AA = ap.copy()
    BB = ap.copy()
    CC = ap.copy()
    delta = ap.copy()

    co = ap.copy()
    so = ap.copy()
    indisk = (abs(xp) < reqtr).nonzero()[0]

    if len(indisk) == 0:
        return None, None, []

    ap[indisk] = np.sqrt(reqtr**2 - xp[indisk]**2)
    cp[indisk] = rpole*np.sqrt(1-xp[indisk]**2/reqtr**2)
    cd, sd = np.cos(subelat), np.sin(subelat)
    AA[indisk] = (ap[indisk]*sd)**2+(cp[indisk]*cd)**2
    BB[indisk] = 2*ap[indisk]*yp[indisk]*sd
    delta1 = ap*ap*sd*sd+cp*cp*cd*cd-yp*yp
    delta = delta1*cp*cp*cd*cd*4

    indisk = indisk[(delta1[indisk] > 0).nonzero()[0]]

#   return None, None, None

    if len(indisk) == 0:
        return None, None, []

    inside = indisk[(AA[indisk] != 0).nonzero()[0]]

    if len(inside) > 0:

        co[inside] = ((-BB[inside]+np.sqrt(delta[inside]))/(2*AA[inside])).clip(-1,1)

        so[inside] = np.sqrt(1-co[inside]**2).clip(0,1)


        w = (yp[inside] < -ap[inside]*sd).nonzero()[0]

        if len(w) > 0:
            so[inside[w]] = -so[inside[w]]
        zp[inside] = cp[inside]*sd*so[inside]+ap[inside]*sd*co[inside]

    zpts = zp.copy()

    sd, cd = np.sin(subelat), np.cos(subelat)
    sa, ca = np.sin(subelon), np.cos(subelon)
    ta = sa/ca

    R = np.array([[-sa, -sd*ca, cd*ca], [ca, -sd*sa, cd*sa], [0., cd, sd]])

    vp = np.zeros((npts, 3))
    for ip in range(npts):
        vp[ip, :] = R.dot([xp[ip],yp[ip],zp[ip]])

    lons = np.zeros_like(ap)
    lats = lons.copy()
    norm = np.sqrt(vp[:,0]**2+vp[:,1]**2+vp[:,2]**2)
    vp[indisk,:] = vp[indisk,:]/np.expand_dims(norm[indisk],1).dot(np.ones((1,3)))
    lons[indisk] = np.rad2deg(np.arctan(vp[indisk,1], vp[indisk,0]))
    lats[indisk] = np.rad2deg(np.arcsin(vp[indisk,2]))

    return lats, lons, indisk



#############################


def append_fields(base, data, **kwargs):
    '''Append array `data` to array `base`, return a structured array

    This is a wrapper of numpy.lib.recfunctions.append_fields().  The
    differences are:
    1. Simplified function signature.
    2. Changed the default usemask=False.  Use default names 'f0',
        'f1', ... if not specified.
    3. If `base` and `data` don't have the same shape, a ValueError is
        raised.
    4. Change the name of fields in data that have the same names as
        those in `base` by appending '_#', with # increasing from 1.

    v1.0.0 : JYL @PSI, 01/27/2015
    '''
    from numpy.lib.recfunctions import append_fields
    usemask = kwargs.pop('usemask', False)
    names = kwargs.pop('names', None)
    data = np.asarray(data)
    if base.shape != data.shape:
        raise ValueError('the shape of new data {0} is different from the shaep of base {1}'.format(base.shape, data.shape))
    if data.dtype.names is None:
        if names is None:
            names = condition(base.dtype.names is None, 'f1', 'f0')
            if base.dtype.names is not None:
                i = int(names[1])
                while names in base.dtype.names:
                    i += 1
                    names = 'f'+str(i).strip()
        return append_fields(base.flatten(), names, data.flatten(), usemask=usemask, **kwargs).reshape(base.shape)
    else:
        out = base.copy().flatten()
        if names is None:
            names = list(data.dtype.names)
        # processing duplicated names
        bn = condition(base.dtype.names is None, ['f0'], base.dtype.names)
        for i in range(len(names)):
            if names[i] in bn:
                j = 1
                n = names[i]+'_'+str(j).strip()
                while n in bn:
                    j += 1
                    n = names[i]+'_'+str(j).strip()
                names[i] = n
        # add fields
        for n,dn in zip(names, data.dtype.names):
            out = append_fields(out, n, data[dn].flatten(), usemask=usemask, **kwargs)
        out = out.reshape(base.shape)
        return out


def drop_fields(base, drop_names, usemask=False, **kwargs):
    '''Remove fields from a structured array

    This is a wrapper of numpy.lib.recfunctions.drop_fields(), with
    default usemask=False.

    v1.0.0 : JYL @PSI, 01/27/2015
    '''
    from numpy.lib.recfunctions import drop_fields
    return drop_fields(base, drop_names, usemask=usemask, **kwargs)


class Measurement(np.ndarray):
    '''Measurement can be initialized by another Measurement instance,
    an (array-like) `data`, or an (array-like) `data` and optional
    (array-like) `error` and other fields.

        # Returns a Measurement instance with the input array-like data
        # or a Measurement
        m = Measurement(data)

        # Returns a Measurement instance with data and other fields
        m = Measurement(data, error=array1, field1=array2, ...)

    Keyword `dtype` specifies the data types of data and all additional
    fields.  It follows the same rule as the `dtype` keyword of ndarray
    generator routines.

    v0.1.0 : JYL @PSI, 01/27/2015
    '''

    def __new__(cls, data, **kwargs):
        dtype = kwargs.pop('dtype', None)
        nbp = len(kwargs)
        err = kwargs.pop('error', None)
        if nbp == 0:
            return np.asanyarray(data, dtype=dtype).view(Measurement)
        else:
            if dtype is None:
                dtype = [('data', np.asarray(data).dtype)]
                if err is not None:
                    dtype.append(('error', np.asarray(err).dtype))
                for n in list(kwargs.keys()):
                    if kwargs[n] is not None:
                        if hasattr(kwargs[n], '__iter__'):
                            dt = np.asanyarray(kwargs[n]).dtype
                        else:
                            dt = type(kwargs[n])
                    else:
                        dt = 'float'
                    dtype.append((n, dt))
        obj = np.zeros_like(data, dtype=dtype).view(Measurement)
        obj.view(np.ndarray)['data'] = data
        if err is not None:
            obj.view(np.ndarray)['error'] = err
        for n in list(kwargs.keys()):
            obj.view(np.ndarray)[n] = condition(kwargs[n] is None, None, kwargs[n])

        return obj

    def __array_finalize__(self, obj): pass

    def _get_data(self,other=None):
        if other is None:
            other = self
        if np.isscalar(other):
            return other
        other = np.asanyarray(other)
        dtype = other.dtype
        if getattr(dtype, 'names', None) is None:
            out = other
        elif 'data' in dtype.names:
            out = other['data']
        else:
            out = other[dtype.names[0]]
        return out.view(np.ndarray)

    def _get_error(self,other=None):
        if other is None:
            other = self
        if np.isscalar(other):
            return None
        dtype = np.asarray(other).dtype
        if getattr(dtype, 'names', None) is None:
            return None
        if 'error' in dtype.names:
            return other['error'].view(np.ndarray)
        else:
            None

    def _carryover_fields(self, other, out):
        f1 = self.fields
        if f1 is not None:
            f1 = f1[1:]  # 'data' is always the first field
            if 'error' in f1: f1.pop(f1.index('error'))
            if len(f1) > 0:
                out = out.append_fields(self[f1])
        if isinstance(other, Measurement):
            f2 = self.fields
            if f2 is not None:
                f2 = f2[1:]
                if 'error' in f2: f2.pop(f2.index('error'))
                if len(f1) > 0:
                    out = out.append_fields(other[f2])
        return out

    def __add__(self, other):
        if self.fields is None:
            return super(Measurement, self).__add__(other)
        else:
            d1, e1 = self.data, self.error
            d2, e2 = self._get_data(other), self._get_error(other)
            s = add(d1, d2, e1, e2)
            if hasattr(s, '__iter__'):
                out = Measurement(s[0], error=s[1])
            else:
                out = Measurement(s)
            #print d1, e1, d2, e2
            return self._carryover_fields(other, out)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        temp = self.__add__(other)
        self['data'] = temp.data
        self['error'] = temp.error
        return self

    def __sub__(self, other):
        return self+(-Measurement(other))

    def __rsub__(self, other):
        return -self+other

    def __isub__(self, other):
        temp = self.__sub__(other)
        self['data'] = temp.data
        self['error'] = temp.error
        return self

    def __mul__(self, other):
        if self.fields is None:
            return super(Measurement, self).__mul__(other)
        else:
            d1, e1 = self.data, self.error
            d2, e2 = self._get_data(other), self._get_error(other)
            s = mul(d1, d2, e1, e2)
            if hasattr(s, '__iter__'):
                out = Measurement(s[0], error=s[1])
            else:
                out = Measurement(s)
            return self._carryover_fields(other, out)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        temp = self.__mul__(other)
        self['data'] = temp.data
        self['error'] = temp.error
        return self

    def __truediv__(self, other):
        if self.fields is None:
            return super(Measurement, self).__truediv__(other)
        else:
            d1, e1 = self.data, self.error
            d2, e2 = self._get_data(other), self._get_error(other)
            s = div(d1, d2, e1, e2)
            if hasattr(s, '__iter__'):
                out = Measurement(s[0], error=s[1])
            else:
                out = Measurement(s)
            return self._carryover_fields(other, out)

    def __rtruediv__(self, other):
        if self.fields is None:
            return super(Measurement, self).__rtruediv__(other)
        else:
            d1, e1 = self.data, self.error
            d2, e2 = self._get_data(other), self._get_error(other)
            s = div(d2, d1, e2, e1)
            if hasattr(s, '__iter__'):
                out = Measurement(s[0], error=s[1])
            else:
                out = Measurement(s)
            return self._carryover_fields(other, out)

    def __itruediv__(self, other):
        temp = self.__truediv__(other)
        self['data'] = temp.data
        self['error'] = temp.error
        return self

    def __pow__(self, other):
        return super(Measurement, self).__pow__(other)

    def __rpow__(self, other):
        return super(Measurement, self).__rpow__(other)

    def __neg__(self):
        out = self.copy()
        out.data = -out.data
        return out

    def __inv__(self):
        return self.__neg__(self)

    def __getitem__(self, *args):
        return super(Measurement, self).__getitem__(*args).view(Measurement)

    def __getslice__(self, *args):
        return super(Measurement, self).__getslice__(*args).view(Measurement)

    @property
    def data(self):
        '''Data field of object'''
        return self._get_data()
    @data.setter
    def data(self, value):
        if self.dtype.names is None:
            v = self.view(np.ndarray)
            v[:] = value
            return
        if 'data' in self.dtype.names:
            self.view(np.ndarray)['data'] = value
        else:
            self.view(np.ndarray)[self.dtype.names[0]] = value

    @property
    def error(self):
        '''The 'error' field of object.  `None` if no error field.'''
        return self._get_error()
    @error.setter
    def error(self, value):
        if self.dtype.names is None:
            raise AttributeError('no error attribute')
        if 'error' not in self.dtype.names:
            raise AttributeError('no error attribute')
        self.view(np.ndarray)['error'] = value

    @property
    def fields(self):
        '''The name of fields'''
        if self.dtype.names is None:
            return None
        return list(self.dtype.names)

    def mean(self, axis=None):
        if axis is None:
            l = self.size
        else:
            l = self.shape[axis]
        return self.sum(axis=axis)/l

    def sum(self, axis=None):
        if self.fields is None:
            return super(Measurement, self).sum(axis=axis)
        else:
            s = np.sum(self.data, axis=axis)
            err = self.error
            e = np.sqrt(np.sum(err*err, axis=axis))
            return Measurement(s, error=e)

    def median(self, axis=None):
        if self.fields is None:
            return np.median(self, axis=axis)
        else:
            m = np.median(self.data, axis=axis)
            err = self.error
            e = np.sqrt(np.sum(err*err, axis=axis))
            if axis is None:
                l = self.size
            else:
                l = self.shape[axis]
            e /= l
            return Measurement(m, error=e)

    def append_fields(self, data, names=None, ignore_unit=True):
        '''Return a new Measurement object with the new array appended'''

        out = append_fields(self.view(np.ndarray), data, names=names).view(Measurement)
        return out

    def astable(self):
        ''' Return a table listing the measurements'''
        if self.fields is None:
            if self.shape is ():
                return Table([[self.dtype.type(self)]], names='data')
            else:
                return Table(self, names='data')
        f = self.flatten()
        tbl = Table()
        for k in self.fields:
            tbl.add_column(Column(f[k], name=k))
        return tbl

    def arrayview(self, copy=False):
        '''Return a view of the object in a simple array.  The simple
        array has a dtype of the first field in Measurement.

        Note: If the dtype of a field in Measurement is different from
        the first (data) field, then the values of that field in the
        simple array could be wrong.'''

        if self.fields is None:
            if copy:
                return self.copy().view(np.ndarray)
            else:
                return self.view(np.ndarray)
        if copy:
            out = np.empty(self.shape+(len(self.fields),),dtype=self.dtype[0])
            for i in range(len(self.fields)):
                out[...,i] = self[self.fields[i]]
            return out
        else:
            return self.view(type=np.ndarray, dtype=self.dtype[0]).reshape(self.shape+(-1,))


class QuantityMeasurement(Measurement):
    '''Measurement class with units'''

    def __new__(cls, *args, **kwargs):
        obj = Measurement(*args, **kwargs).view(QuantityMeasurement)
        if obj.fields is None:
            obj.unit = condition(isinstance(args[0], units.Quantity), args[0].unit, None)
        else:
            obj.unit = []
            for f in obj.fields:
                obj.unit.append(condition(isinstance(kwargs[f], units.Quantity), units.Quantity, None))

    def __array_finalize__(self, obj):
        if obj is None: return
        super(QuantityMeasurement, self).__array_finalize__(obj)
        self.unit = getattr(obj, 'unit', None)


class ImageMeasurement(Measurement):
    '''Image class wrapper

    v0.0.1 : JYL @PSI, 01/26/2015
    '''
    def __new__(cls, data, **kwargs):
        obj = Measurement(data, **kwargs).view(ImageMeasurement)
        return obj

def time_stamp(format=0):
    now = Time.now()
    if format in [0, 'date']:
        out = ''.join(now.isot.split('T')[0].split('-'))
    elif format in [1, 'isot']:
        out = now.isot
    return out


#class QuantityImage(QuantityMeasurement, Image): pass


def imageclean(im, threshold=3., pos=None, box=20, step=None, untouch=None, mask=None):
    '''
 Clean an image by filling the pixels outside of threshold with the
 average value inside a box.

 im : input image
 threshold : float, optional
   Clean threshold in unit of sigma.  Default is 3.
 pos : [x, y] or list of [x, y], optional
   The positions to be cleaned.  Default is to clean the whole image
 box : int, optional
   Size of box, default 20x20
 step : int, optional
   Step size from one box to the next.  Default is half box size
 untouch : slice or iterable of slices, optional
   The region in the image that should not be touched
 mask : array variable of the same shape as `im`, optional
   Returns the mask of pixels being cleaned

 v1.0.0 : 4/29/2015, JYL @PSI
    '''

    box2 = box//2
    if step is None:
        step = box2
    if pos is None:
        ys, xs = im.shape
        xv = np.arange(box2,xs,step)
        yv = np.arange(box2,ys,step)
        pos = np.array(np.meshgrid(xv,yv)).reshape(2,len(xv)*len(yv))
    else:
        pos = np.asarray(pos).T

    if mask is None:
        mask = np.zeros_like(im,dtype=int)

    im1 = im.copy()
    for x,y in pos.T:
        subim = im[y-box2:y+box2,x-box2:x+box2]#*(1-mask[y-box2:y+box2,x-box2:x+box2])
        submsk = np.zeros_like(subim,dtype=int)
        m,std = resmean(subim,threshold,std=True)
        submsk[abs(subim-m) > std*threshold] = 1
        mask[y-box2:y+box2,x-box2:x+box2] = submsk
        im1[y-box2:y+box2,x-box2:x+box2][submsk==1] = m

    if untouch is not None:
        im1[untouch] = im[untouch]
        mask[untouch] = 0

    return im1


def iter_exec(func, indata, outdata=None, verbose=True, ext=None, **kwargs):
    '''Iterate a command in subdirectories

    Parameters
    ----------
    func : Function to execute
    indata : str
      A single file, or a directory.  If directory, then all files in
      it, or all subdirectories, will be processed iteratively.
    outdata : str, optinonal
      Output file or directory corresponding to `indata`.  I.e., if
      `indata` is a file/dir, then `outdata` is considered a file/dir.
    ext : str, optional
      Select particular extension in the input directories
    verbose : bool, optional
      Verbose mode.  Default is True
    **kwargs : other keywords accepted by `func`

    v1.0.0, 05/26/2015, JYL @PSI
    v1.0.1, 10/26/2015, JYL @PSI
      Add keyword `ext`
      Change argument `outdata` to an (optional) keyword argument to
        accomodate for the case where output file is not needed.
      Add the capability to accept lists in `indata` and `outdata`
      Improve the procedure
    '''

    from os.path import isdir, isfile, basename, dirname, join
    from os import makedirs

    if hasattr(indata, '__iter__'):
    # If `indata` is a list of files or directories
        if outdata is not None:
            if hasattr(outdata, '__iter__'):
                if len(indata) != len(outdata):
                    raise ValueError('`indata` and `outdata` do not have the same length')
            else:
                outdata = [join(outdata, basename(x)) for x in indata]
            for fi, fo in zip(indata, outdata):
                iter_exec(func, fi, fo, verbose, ext, **kwargs)
            print()
        else:
            for fi in indata:
                iter_exec(func, fi, None, verbose, ext, **kwargs)
            print()
    else:
    # If `indata` is a single string
        if isfile(indata):
            if verbose:
                print(('Processing file: ', basename(indata)))
            if outdata is not None:
                outpath = dirname(outdata)
                outfile = basename(outdata)
                if not isdir(outpath):
                    makedirs(outpath)
                func(indata, outdata, **kwargs)
            else:
                func(indata, **kwargs)
        elif isdir(indata):
            insidedir = findfile(indata, dir=True)
            insidefile = findfile(indata, name=ext)
            if len(insidefile) > 0:
                if verbose:
                    print(('Directory {0}: {1} files found'.format(basename(indata), len(insidefile))))
                if outdata is not None:
                    od = [join(outdata, basename(x)) for x in insidefile]
                else:
                    od = None
                iter_exec(func, insidefile, od, verbose, ext, **kwargs)
            if len(insidedir) > 0:
                if verbose:
                    print(('Directory {0}: {1} subdirectories found'.format(basename(indata), len(insidedir))))
                if outdata is not None:
                    od = [join(outdata, basename(x)) for x in insidedir]
                else:
                    od = None
                iter_exec(func, insidedir, od, verbose, ext, **kwargs)
        else:
            raise ValueError('input not found')


def scale(arr, range=[0,255], a_min=None, a_max=None, clip=False):
    '''Scale the input array to a range

    arr : array-like
      Array to be scaled
    range : two-element array-like, optional
      The range to be scaled.  Default is byte
    a_min, a_max : scales, optional
      The minimum and maximum of `arr` in the scaling.
    clip : bool, optional
      If `True`, then `arr` will be clipped with `a_min` and `a_max`
      beofre scaling.  Otherwise just scaling.

    v1.0.0 : 10/27/2015, JYL @PSI
    v1.0.1 : 3/5/2016, JYL @PSI
      Change `range` to an optional argument with a default [0,255]
    '''
    arr = np.asarray(arr)
    if a_min is None:
        a_min = arr.min()
    if a_max is None:
        a_max = arr.max()
    if clip:
        arr = np.clip(arr, a_min, a_max)
    return (arr-arr.min())/(arr.max()-arr.min())*(range[1]-range[0])+range[0]
