# Package to process Hubble Space Telescope data
#
#

from warnings import warn
import numpy as np
import spiceypy as spice
from collections import OrderedDict
from ..core import Image, ImageMeasurement, readfits, ascii_read, sflux, rebin
from ..apext import Table
from ..image import ImageSet
import astropy.units as u
from astropy.table import QTable, Column, Table
from astropy.io import fits, ascii
from astropy.nddata import StdDevUncertainty
import ccdproc


filter_table = '/Users/jyli/work/references/HST/WFC3/WFC3_Filter_List.csv'
wfc3dir = '/Users/jyli/work/references/HST/WFC3/'

def load_filter():
    flist = QTable(ascii.read(filter_table))
    flist['PHOTPLAM'].unit = u.nm
    flist['PHOTFLAM'].unit = u.Unit('W m-2 um-1')
    flist['PHOTBW'].unit = u.nm
    flist['SolarFlux'].unit = u.Unit('W m-2 um-1')
    return flist

def filter_bandpass(flt):
    '''Return filter bandpass in a Table'''
    from os.path import isfile
    flist = ascii_read(filter_table)
    thfile = flist.query('Filter',flt,'ThroughputFile')
    if len(thfile) == 0:
        raise ValueError('{0} not found'.format(flt))
    thfile = wfc3dir+'Filter_Throughput/'+thfile[0]
    if not isfile(thfile):
        raise IOError('{0}: Filter throughput file not found: {1}'.format(flt,thfile))
    return Table(fits.open(thfile)[1].data)


def aspect(files, out=None, target=None, kernel=None, keys=None, verbose=False):
    '''Extract aspect data for WFC3 images

 Parameters
 ----------
 files : array-like, str
   Data files to be processed
 target : str, optional
   Name of target.  If present, it has to be in SPICE name space.
 out : str, optional
   Output file name.  If `None`, then the aspect will be printed out
   on screen.
 kernel : str, optional
   SPICE kernel file name.
 keys : str or array-like str, optional
   Additional keys to be included in the output.  If a key is not
   found in FITS header, then the corresponding output table element
   will be masked out.
 verbose : bool, optional
   Verbose mode

 Returns
 -------
 astropy table
   Contains the extracted aspect table

 History
 -------
 v1.0.0 : JYL @PSI, Feb 12, 2014
 v1.1.0 : JYL @PSI, Oct 21, 2014
   * Major reorganization of the program.
   * Add keyword `keys` to retrieve additional FITS keys.
    '''

    from jylipy.vector import vecpa, vecsep
    from numpy.linalg.linalg import norm

    # Load SPICE kernel if needed
    geo = False
    if kernel is not None and target is not None:
        geo = True
        try:
            import spice
        except ImportError:
            print('No SPICE module found.')
            geo = False
        if kernel is not None:
            spice.furnsh(kernel)
            if spice.bodn2c(target) is None:
                print('Target name not in SPICE name space.  Geometry keys ignored')
                geo = False

    # Set up table columns
    fitskeys = 'RootName Date-Obs Time-Obs ExpStart ExpEnd Filter ExpTime Orientat'.split()
    fitsfmt = '{:s} {:s} {:s} {:.7f} {:.7f} {:s} {:.2f}'.split()
    if keys is not None:
        if not isinstance(keys, (str,bytes)):
            keys = list(keys)
        else:
            keys = [keys]
    else:
        keys = []
    #fitskeys += keys
    nfk = len(fitskeys + keys)

    if geo:
        spicekeys = 'Rh Range Phase PxlScl NorPA SunPA SunAlt VelPA VelAlt'.split()
        spicefmt = '{:.5f} {:.5f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.split()
        unt = 'AU AU deg km deg deg deg deg deg'.split()
    else:
        spicekeys = []
    row, mask = [None]*(nfk+len(spicekeys)), [False]*(nfk+len(spicekeys))
    rows, masks = [], []

    # Loop through images
    for f, j in zip(files,list(range(len(files)))):
        if verbose:
            print('Processing image '+f+'...')
        img = fits.open(f)

        # Collect FITS keywords
        for k, i in zip(fitskeys+keys, list(range(nfk))):
            try:
                row[i] = img[0].header[k]
            except KeyError:
                mask[i] = True
        row[3] = np.float64(row[3])+np.float64(2400000.5)
        row[4] = np.float64(row[4])+np.float64(2400000.5)

        # Collect geometry information
        if geo:
            cjd = (row[3]+row[4])/2
            et = spice.str2et('JD'+repr(cjd))
            az = (360-img[1].header['orientat']) % 360
            pos1, ltime1 = spice.spkezr(target, et, 'j2000', 'lt+s', 'earth')
            losxyz = pos1[0:3]
            pos2, ltime2 = spice.spkezr('sun', et-ltime1, 'j2000', 'lt+s', target)
            velxyz = pos2[3:6]
            sunxyz = pos2[0:3]
            sun = vecpa(losxyz, sunxyz)
            vel = vecpa(losxyz, -np.array(velxyz))
            row[nfk] = norm(pos2)/1.496e8  # rh
            row[nfk+1] = norm(pos1)/1.496e8  # delta
            row[nfk+2] = vecsep(-np.array(pos1), pos2, False)  # phase
            row[nfk+3] = norm(pos1)*0.04/206265  # pixel scale
            row[nfk+4] = (az)  # noraz
            row[nfk+5] = (sun[0]+az+360) % 360  # sunaz
            row[nfk+6] = sun[1]  # sunalt
            row[nfk+7] = (vel[0]+az+360) % 360  # velpa
            row[nfk+8] = vel[1]   # velalt

        rows.append(row[:])
        masks.append(mask[:])

    # Create table
    asp = Table(names=fitskeys+keys+spicekeys, rows=rows, masked=True)
    for k, m in zip(list(asp.keys()), np.array(masks).T):
        asp[k].mask = m

    # Post-processing for the table
    for k, f in zip(fitskeys,fitsfmt):
        asp[k].format = f
    if geo:
        for k,f,u in zip(spicekeys,spicefmt,unt):
            asp[k].format = f
            asp[k].unit = u
    asp['ExpTime'].unit='sec'
    asp.sort('ExpStart')

    # Save or print table
    if out is not None:
        ascii.write(asp, out, delimiter='\t')
    else:
        asp.pprint(show_unit=True)

    # Unload SPICE
    if kernel is not None:
        spice.unload(kernel)

    return asp


def solarflux(filters, spec=None):
    '''
 Calculate equivalent solar flux through the given filter(s)

 Parameters
 ----------
 filters : str, or list of str
   Name of WFC3 filters
 spec : (2, n) array
   Solar spectrum [[Wavelength], [Spectrum]].  Wavelength and spectrum
   can be either numbers or astropy quantities.  If only one is
   astropy quantities, then its unit will be ignored.  If numbers,
   then the unit of wavelength is 'um', and the unit of spectrum is
   'W m-2 um-1'.

 Returns
 -------
 astropy quantity or list of it :
   Equivalent solar flux(es)
    '''

    flist = ascii.read(filter_table)

    if isinstance(filters, (str,bytes)):
        filters = [filters]

    sf = []
    for flt in filters:
        z = flist['Filter']==flt
        thfile = wfc3dir+'Filter_Throughput/'+flist['ThroughputFile'][z].data[0]
        if thfile != np.ma.core.MaskedConstant:
            try:
                th = fits.open(thfile)
                sflx = sflux(th[1].data['Wavelength']*u.angstrom, th[1].data['THROUGHPUT'], spec=spec)
                mag = -2.5*np.log10((sflx/(flist['PHOTFLAM'][z].data[0]*u.Unit('W m-2 um-1'))).decompose().value)+flist['VEGAmag'][z].data[0]
                sf.append([sflx,mag])

            except IOError:
                print(flt+': Filter throughput file not found: '+ thfile)
        else:
            print(flt+': Filter throughput file not defined.')

    return sf


def listfilters(jsviewer=True):
    '''
 List the WFC3 filters in a browser window

 Note that the list is NOT complete.  Only the filters currently
 included in the local filter table are listed.

 v1.0.0 : JYL @PSI, November 5, 2014
    '''

    return load_filter().show_in_browser(jsviewer=jsviewer)


#class UVISImageMeasurement(ImageMeasurement):
#
#   def __new__(cls, inputfile, dtype=None):
#       from astropy.io import fits
#       if not (inputfile.endswith('_flt.fits') or inputfile.endswith('_drz.#fits')):
#           raise ValueError('input file extension not recognized')
#       fitsfile = fits.open(inputfile)
#       header = OrderedDict()
#       header['primary'] = fitsfile.pop(0).header
#       data = []
#       if inputfile.endswith('_flt.fits'):  # _FLT file
#           if header['primary']['aperture'].find('SUB') == -1:
#               sci = np.concatenate((fitsfile[0].data,fitsfile[3].data))
#               err = np.concatenate((fitsfile[1].data,fitsfile[4].data))
#               dq = np.concatenate((fitsfile[2].data,fitsfile[5].data))
#               header['sci'] = [fitsfile[0].header,fitsfile[3].header]
#               header['err'] = [fitsfile[1].header,fitsfile[4].header]
#               header['dq'] = [fitsfile[2].header,fitsfile[5].header]
#               obj = Image(sci, error=StdDevUncertainty(err), dq=dq, dtype=#dtype).view(UVISImage)
#           else:
#               for k in 'sci err dq'.split():
#                   header[k] = fitsfile[k].header
#               obj = Image(fitsfile[0].data, error=StdDevUncertainty(fitsfile[#1].data), dq=fitsfile[2].data, dtype=dtype).view(UVISImage)
#       else:
#           obj = Image(fitsfile[0].data, wht=fitsfile[1].data, ctx=fitsfile[2]#.data)
#           header['sci'] = fitsfile[0].header
#           header['wht'] = fitsfile[1].header
#           header['ctx'] = fitsfile[2].header
#       obj.header = header
#       obj.source = inputfile
#       obj.geometry = {}
#       obj.calibration = {}
#       return obj
#
#   def __finalize_array__(self, obj):
#       super(UVISImage, self).__array_finalize__(obj)
#       if obj is None: return
#       from copy import copy
#       self.source = getattr(obj, 'source', None)
#       self.header = copy(getattr(obj, 'header', None))
#       self.geometry = copy(getattr(obj, 'geometry', {}))
#       self.calibration = copy(getattr(obj, 'calibration', {}))


class UVISImage(Image):
    @classmethod
    def read(cls, inputfile):
        from astropy.nddata import StdDevUncertainty, FlagCollection
        if not (inputfile.endswith('_flt.fits') \
                or inputfile.endswith('_drz.fits')):
            raise ValueError('input file extension not recognized')
        fitsfile = fits.open(inputfile)
        header = OrderedDict()
        header['primary'] = fitsfile.pop(0).header
        data = []
        if inputfile.endswith('_flt.fits'):  # _flt file
            if header['primary']['aperture'].find('SUB') == -1:
                sci = np.concatenate((fitsfile[0].data,fitsfile[3].data))
                err = np.concatenate((fitsfile[1].data,fitsfile[4].data))
                dq = np.concatenate((fitsfile[2].data,fitsfile[5].data))
                header['sci'] = fitsfile[0].header,fitsfile[3].header
                header['err'] = fitsfile[1].header,fitsfile[4].header
                header['dq'] = fitsfile[2].header,fitsfile[5].header
            else:
                for k in 'sci err dq'.split():
                    header[k] = fitsfile[k].header
                sci = fitsfile[0].data
                err = fitsfile[1].data
                dq = fitsfile[2].data
            obj = cls(sci, unit='electron', uncertainty=StdDevUncertainty(err),
                        mask=(dq != 0), meta=header, flags=dq)
        else:  # _drz file
            obj = cls(fitsfile[0].data, unit='electron/s', meta=header)
            header['sci'] = fitsfile[0].header
            header['wht'] = fitsfile[1].header
            header['ctx'] = fitsfile[2].header
            flags = FlagCollection(shape=fitsfile[0].data.shape)
            flags['wht'] = fitsfile[1].data
            flags['ctx'] = fitsfile[2].data
            obj = ccdproc.create_deviation(obj,
                    gain=header['primary']['exptime']*u.s,
                    readnoise=3*u.electron)
        return obj


def read_uvis(inputfile):
    return UVISImage.read(inputfile)


class UVISCalibration(object):

    pamfits = [wfc3dir+'Pixel_Area_Map/UVIS'+str(x)+'wfc3_map.fits' for x in [1,2]]
    pxlscl = 0.04 * u.arcsec

    def __init__(self):
        self.pam = [readfits(f,ext=1) for f in self.pamfits]
        self.photcal = load_filter()


class PAM():
    """Pixel area map class"""

    pam1file = wfc3dir + 'Pixel_Area_Map/UVIS1wfc3_map.fits'
    pam2file = wfc3dir + 'Pixel_Area_Map/UVIS2wfc3_map.fits'
    amps = [['FQ387N','FQ437N','FQ508N','FQ619N','FQ889N'],
            ['FQ378N','FQ492N','FQ674N','FQ750N','FQ937N'],
            ['FQ232N','FQ422M','FQ575N','FQ634N','FQ906N'],
            ['FQ243N','FQ436N','FQ672N','FQ727N','FQ924N']]

    def __init__(self, datafiles=[pam1file, pam2file]):
        with fits.open(datafiles[0]) as f_:
            self._pam1 = f_[1].data.copy()
        with fits.open(datafiles[1]) as f_:
            self._pam2 = f_[1].data.copy()
        self._A = self._pam1[-512:, :513]
        self._B = self._pam1[-512:, -513:]
        self._C = self._pam2[:512, :513]
        self._D = self._pam2[:512, -513:]

    def __call__(self, aperture='UVIS', filter=None, binning=1):
        """Return the PAM corresponding to specified aperture and/or filter"""
        if binning not in [1, 2, 3]:
            raise ValueError('binning must be in [1, 2, 3].')
        if aperture not in Aperture()['Aperture']:
            raise ValueError('invalid aperture {}'.format(aperture))
        if aperture == 'UVIS':
            pam = np.r_[self._pam2, self._pam1]
            if binning == 3:
                return rebin(pam, (3,3), mean=True)[:, 1:-1]
            elif binning == 2:
                raise ValueError("NO IDEA ABOUT BINNING == 2")
            else:
                return pam
        elif aperture.find('UVIS1') != -1:
            pam = self._pam1
        elif aperture.find('UVIS2') != -1:
            pam = self._pam2
        elif aperture.find('QUAD') != -1:
            if filter == None:
                raise ValueError("filter must be specified for 'QARD' "
                                 "apertures")
            if filter in self.amps[0]+self.amps[1]:
                pam = self._pam1
            elif filter in amps[2] + self.amps[3]:
                pam = self._pam2
        # 2k subarrays
        if (aperture.find('2K2A') != -1) or (aperture.find('2K2C') != -1):
            return pam[1:,:2047]
        if (aperture.find('2K2B') != -1) or (aperture.find('2K2D') != -1):
            return pam[1:,2049:]
        # quad
        if aperture.find('quad') != -1:
            if filter in amp[0] + amps[2]:
                return pam[1:,:2047]
            else:
                return pam[1:,2049:]
        # 1k subarrays
        if aperture.find('C1K1C') != -1:
            return pam[:1024, :1025]
        if aperture.find('M1K1C') != -1:
            return pam[-1024:,-1025:]
        # 512 subarrays: M512C, C512C
        if aperture.find('C512C') != -1:
            return pam[:512,:512]
        if aperture.find('M512C') != -1:
            return pam[-512:,-512:]
        # quad filters
        if aperture.find('QUAD') != -1:
            if filter in self.amps[0]+self.amps[2]:
                return pam[1:, :2047]
            else:
                return pam[1:, 2049:]
        return pam


def read_jit(fn):
    '''Read jitter data and return in a Table

    v1.0.0 : 5/1/2016, JYL @PSI
    '''

    jit = Table().read(fn)
    for k in list(jit.keys()):
        if jit[k].unit == 'seconds':
            jit[k].unit = u.s
        elif jit[k].unit == 'arcsec':
            jit[k].unit = u.arcsec
        elif jit[k].unit == 'degrees':
            jit[k].unit = u.deg
        elif jit[k].unit == 'Gauss':
            jit[k].unit = u.G

    return jit


class Aperture(QTable):
    """WFC3 aperture list"""
    def __init__(self, *args, aperture_list=wfc3dir+'Aperture_File.csv',
                 **kwargs):
        if len(args) == 0:
            super().__init__(ascii.read(aperture_list))
            self['v2pos'].unit = u.arcsec
            self['v3pos'].unit = u.arcsec
            self['xscl'].unit = u.arcsec
            self['yscl'].unit = u.arcsec
            self['v3x'].unit = u.deg
            self['v3y'].unit = u.deg
        else:
            super().__init__(*args, **kwargs)


def obslog(files):
    '''Extract observation log from jif and jit files

    Input files are either the rootname, or full image names, in both
    cases with the full directory path

    '''
    from collections import OrderedDict

    if isinstance(files, (str,bytes)):
        files = [files]

    keys = [[0, 'ROOTNAME', ''],
            [0, 'TARGNAME', ''],
            [0, 'GUIDECMD', ''],
            [1, 'GUIDEACT', ''],
            [1, 'APERTURE', ''],
            [1, 'APER_V2', 'arcsec'],
            [1, 'APER_V3', 'arcsec'],
            [1, 'ALTITUDE', 'km'],
            [1, 'LOS_SUN', 'deg'],
            [1, 'LOS_MOON', 'deg'],
            [1, 'LOS_LIMB', 'deg'],
            [1, 'RA_AVG', 'deg'],
            [1, 'DEC_AVG', 'deg'],
            [1, 'ROLL_AVG', 'deg']]

    values = OrderedDict()
    for e,k,u in keys:
        values[k] = []
    for f in files:
        if f[-9] != '_':   # rootname
            jf = f[:-1]+'j_jif.fits'
        else:  # full names
            jf = f[:-10]+'j_jif.fits'
        fjf = fits.open(jf)
        for e, k, u in keys:
            values[k].append(fjf[e].header[k])

    log = Table(values)
    for e,k,u in keys:
        if u != '':
            log[k].unit = u.Unit(u)

    return log


class AperturePhotometry(ImageSet):
    """Aperture photometry of UVIS images

    Parameters
    ----------
    yc, xc : the coordinates of centroids, required
    uvis_aper : UVIS aperture name from FITS header keyword 'APERTURE', required
    filter : Filter name from FITS header keyword 'FILTER', required
    exptime : Exposure time in seconds, from FITS header key 'EXPTIME'
    bin : Image binning, from FITS header key 'BINAXIS1' or 'BINAXIS2', optional
    background : Background measurement, optional.  If provided, then background
        will be subtracted from photometric measurements.
    background_error : Background measurement error, optional.  If provided,
        then it will be included in the photometric measurement errors.

    Image can be passed or loaded by the specified loader as numpy arrays, or
    `astropy.nddata.NDData`.  If `astropy.nddata.NDData`, then the error can
    be provided in attribute `.uncertainty`.  If provided, then it will be used
    to estimate the error in photometry.  A mask can also be provided in
    attribute `.mask`, and will be used in photometry if available.

    Photometry is saved in attribute `.phot` as an `astropy.unit.Quantity`, and
    the uncertainty is saved in `.photerr` if available.
    """

    def __init__(self, *args, **kwargs):
        keys = kwargs.keys()
        if ('xc' not in keys) or ('yc' not in keys):
            raise ValueError('`yc` and `xc` are required keyword arguments.')
        required_keys = ['uvis_aper', 'filter', 'exptime']
        for k in required_keys:
            if k not in keys:
                raise ValueError('`{}` is a required argument.'.format(k))
        optional_keys = ['bin', 'background', 'background_error']
        for k in optional_keys:
            if k not in keys:
                warn('`{}` is not provided.')
        super().__init__(*args, **kwargs)

    def apphot(self, aperture, photcal=False, **photcal_kwargs):
        """Measure aperture photometry

        aperture : number or array
            Aperture radii
        """
        from photutils import aperture_photometry, CircularAperture
        self.aperture = aperture
        if not hasattr(aperture, '__iter__'):
            aperture = [aperture]
        n_aper = len(aperture)
        pam = PAM()
        phot = np.zeros((self._size, n_aper))
        photerr = np.zeros((self._size, n_aper))
        for i in range(self._size):
            if self.image is None or self._1d['image'][i] is None:
                self._load_image(i)
            uvis_aper = self._1d['image'][i]
            binning = self._1d['_bin'][i] if '_bin' in self.attr else 1
            im_pam = pam(aperture=self._1d['_uvis_aper'][i],
                         filter=self._1d['_filter'][i],
                         binning=binning)
            im = getattr(self._1d['image'][i], 'data', self._1d['image']) \
                    * im_pam
            err = getattr(self._1d['image'][i], 'uncertainty', None)
            if err is not None:
                if isinstance(err, StdDevUncertainty):
                    err = err.array
                err = err * im_pam
            mask = getattr(self._1d['image'][i], 'mask', None)
            apers = [CircularAperture((self._1d['_xc'][i], self._1d['_yc'][i]),
                        r) for r in aperture]
            phot_table = aperture_photometry(im, apers, error=err, mask=mask)
            phot_cols = ['aperture_sum_{}'.format(i) for i in range(n_aper)]
            p = phot_table[phot_cols].as_array().view((float, n_aper))
            phot[i] = p.flatten()
            if '_background' in self.attr:
                ap_area = np.array([x.area for x in apers])
                bg = self._1d['_background'][i] * ap_area
                phot[i] -= bg
            phot[i] /= self._1d['_exptime'][i]
            if err is not None:
                err_cols = ['aperture_sum_err_{}'.format(i) for i in \
                        range(n_aper)]
                e = phot_table[err_cols].as_array().view((float, n_aper))
                photerr[i] = e.flatten()
                if '_background_error' in self.attr:
                    bgerr = self._1d['_background_error'][i] * ap_area
                    photerr[i] = np.sqrt(photerr[i]**2 + bgerr**2)
                photerr[i] /= self._1d['_exptime'][i]
        countrate = (phot * u.electron / u.s)
        self.countrate = countrate.reshape(self._shape + (n_aper,))
        if not (photerr == 0).all():
            countrate_error = photerr * u.electron / u.s
            self.countrate_error = countrate_error.reshape(self._shape \
                                                            + (n_aper,))
        if photcal:
            self.photcal(**photcal_kwargs)

    def photcal(self, STmag=False, ABmag=False):
        """Convert count rate to flux and Vega magnitude.

        """
        if not hasattr(self, 'countrate'):
            raise ValueError('count rate not available.')
        filter_table = load_filter()
        photflam = []
        vegamag = []
        stmag = []
        abmag = []
        for i, x in enumerate(self._1d['_filter']):
            w = x == filter_table['Filter']
            row = filter_table[w]
            photflam.append(row['PHOTFLAM'])
            vegamag.append(row['VEGAmag'])
            stmag.append(row['STmag'])
            abmag.append(row['ABmag'])
        photflam = np.reshape(np.squeeze(photflam), self._shape)
        vegamag = np.reshape(np.squeeze(vegamag), self._shape)
        stmag = np.reshape(np.squeeze(stmag), self._shape)
        abmag = np.reshape(np.squeeze(abmag), self._shape)
        self.flux = np.moveaxis(np.moveaxis(self.countrate, -1, 0) * photflam,
                                0, -1)
        self.mag = -2.5 * np.log10(self.countrate.value) * u.mag
        mag = np.moveaxis(self.mag.value, -1, 0)
        self.VEGAmag = np.moveaxis(mag + vegamag, 0, -1) * u.mag
        if STmag:
            self.STmag = np.moveaxis(mag + stmag, 0, -1) * u.mag
        if ABmag:
            self.ABmag = np.moveaxis(mag + abmag, 0, -1) * u.mag
        if hasattr(self, 'countrate_error'):
            self.flux_error = (self.countrate_error.T * photflam).T
            self.mag_error = 2.5 * np.log10((self.countrate_error \
                                                / self.countrate) + 1) * u.mag

    def write(self, filename, **kwargs):
        """Write photometry to output file

        The results will be saved to a multi-extension FITS file with the
        following structure:
            Primary extension only saves some information in the header
            1st extension ['params'] : Binary table listing the measuring
                parameters
            2nd extension ['aperture'] : .aperture
            3rd extension ['countrat'] : .countrate
            4th extension ['flux'] : .flux
            5th extension ['mag'] : .mag, instrument magnitude
            6th extension ['vegamag'] : .VEGAmag
            7th extension ['stmag'] : .STmag
            8th extension ['abmag'] : .ABmag
            9th extension ['crerror'] : .countrate_error if available
            10th extension ['flxerror'] : .flux_error if available
            11th extension ['magerror'] : .mag_error if available
        """
        hdu = fits.PrimaryHDU()
        for i in range(len(self._shape)):
            hdu.header['axis{}'.format(i)] = self._shape[i]
            hdu.header['ndim'] = len(self._shape)
        outfits = fits.HDUList([hdu])
        # 1st extension
        cols = []
        for k in self.attr:
            cols.append(Column(self._1d[k], name=k.strip('_')))
        if self.file is not None:
            cols.insert(0, Column(self._1d['file'], name='file'))
        out = Table(cols)
        tblhdu = fits.BinTableHDU(out, name='params')
        outfits.append(tblhdu)
        # 2nd extension
        outfits.append(fits.ImageHDU(self.aperture, name='aperture'))
        # 3rd
        hdu = fits.ImageHDU(self.countrate.value, name='countrat')
        hdu.header['bunit'] = str(self.countrate.unit)
        outfits.append(hdu)
        # 4th - 11th
        if hasattr(self, 'flux'):
            hdu = fits.ImageHDU(self.flux.value, name='flux')
            hdu.header['bunit'] = str(self.flux.unit)
            outfits.append(hdu)
        if hasattr(self, 'mag'):
            hdu = fits.ImageHDU(self.mag.value, name='mag')
            hdu.header['bunit'] = str(self.mag.unit)
            outfits.append(hdu)
        if hasattr(self, 'VEGAmag'):
            hdu = fits.ImageHDU(self.VEGAmag.value, name='vegamag')
            hdu.header['bunit'] = 'VEGAmag'
            outfits.append(hdu)
        if hasattr(self, 'STmag'):
            hdu = fits.ImageHDU(self.STmag.value, name='stmag')
            hdu.header['bunit'] = 'STmag'
            outfits.append(hdu)
        if hasattr(self, 'ABmag'):
            hdu = fits.ImageHDU(self.ABmag.value, name='abmag')
            hdu.header['bunit'] = 'ABmag'
            outfits.append(hdu)
        if hasattr(self, 'countrate_error'):
            hdu = fits.ImageHDU(self.countrate_error.value, name='cterror')
            hdu.header['bunit'] = str(self.countrate_error.unit)
            outfits.append(hdu)
        if hasattr(self, 'flux_error'):
            hdu = fits.ImageHDU(self.flux_error.value, name='fluxerror')
            hdu.header['bunit'] = str(self.flux_error.unit)
            outfits.append(hdu)
        if hasattr(self, 'mag_error'):
            hdu = fits.ImageHDU(self.mag_error.value, name='magerror')
            hdu.header['bunit'] = 'mag'
            outfits.append(hdu)
        outfits.writeto(filename, **kwargs)

    def read(self, infile):
        """Read photometry from input file

        Parameters
        ----------
        infile : str
            Input file name
        """
        with fits.open(infile) as f_:
            extnames = [x.header['extname'] for x in f_[1:]]
            ndim = f_[0].header['ndim']
            shape = ()
            for i in range(ndim):
                shape = shape + (f_[0].header['axis{}'.format(i)],)
            intable = Table(f_['params'].data)
            self.aperture = f_['aperture'].data
            self.countrate = f_['countrat'].data \
                                 * u.Unit(f_['countrat'].header['bunit'])
            if 'FLUX' in extnames:
                self.flux = f_['flux'].data * u.Unit(f_['flux'].header['bunit'])
            if 'MAG' in extnames:
                self.mag = f_['mag'].data * u.Unit(f_['mag'].header['bunit'])
            if 'VEGAMAG' in extnames:
                self.VEGAmag = f_['vegamag'].data * u.mag
            if 'STMAG' in extnames:
                self.STmag = f_['stmag'].data * u.mag
            if 'ABMAG' in extnames:
                self.ABmag = f_['abmag'].data * u.mag
            if 'CTERROR' in extnames:
                self.countrate_error = f_['cterror'].data \
                                * u.Unit(f_['cterror'].header['bunit'])
            if 'FLUXERROR' in extnames:
                self.flux_error = f_['fluxerror'].data \
                                * u.Unit(f_['fluxerror'].header['bunit'])
            if 'MAGERROR' in extnames:
                self.mag_error = f_['magerror'].data \
                                * u.Unit(f_['magerror'].header['bunit'])
        keys = intable.keys()
        if 'file' in keys:
            self.file = np.array(intable['file'])
            keys.remove('file')
            self.image = None
        else:
            self.file = None
            self._ext = None
        self.attr = []
        for k in keys:
            n = '_' + k
            self.attr.append(n)
            setattr(self, n, np.array(intable[k]))
        # adjust shape
        if ndim == 0:
            self._shape = len(intable),
            self._size = self._shape[0]
        else:
            self._shape = shape
            self._size = int(np.array(shape).prod())
        if self.file is not None:
            self.file = self.file.reshape(self._shape)
        # process attributes
        for k in self.attr:
            v = getattr(self, k)
            if np.all(v == v[0]):
                setattr(self, k, v[0])
        # generate flat view
        self._generate_flat_views()

    @classmethod
    def from_fits(cls, infile, loader=None):
        obj = cls('', xc=0, yc=0, uvis_aper=0, filter=0, exptime=0)
        obj.read(infile)
        obj.loader = loader
        return obj

    def explode(self, outfile, overwrite=True):
        """Separate photometry by filters
        """
        from os.path import splitext
        flts = np.unique(self._filter)
        flds = ['countrate', 'flux', 'mag', 'VEGAmag', 'STmag', 'ABmag',
                'countrate_error', 'flux_error', 'mag_error']
        rootname, ext = splitext(outfile)
        for f in flts:
            ww = self._1d['_filter'] == f
            kwargs = {}
            for k in self.attr:
                kwargs[k.strip('_')] = self._1d[k][ww]
            ap = AperturePhotometry(self._1d['file'][ww], **kwargs)
            if hasattr(self, 'aperture'):
                ap.aperture = self.aperture
            for x in flds:
                if hasattr(self, x):
                    setattr(ap, x, getattr(self, x)[ww])
            ap.write(rootname+'_'+f+ext, overwrite=overwrite)
