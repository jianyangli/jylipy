# Package to process Hubble Space Telescope data
#
#

import numpy as np
import spiceypy as spice
from collections import OrderedDict
from ..core import Image, ImageMeasurement, readfits, ascii_read, sflux
from ..apext import Table, nddata, units
import ccdproc
from astropy.nddata import StdDevUncertainty


filter_table = '/Users/jyli/work/references/HST/WFC3/WFC3_Filter_List.csv'
wfc3dir = '/Users/jyli/work/references/HST/WFC3/'

def load_filter():
    flist = ascii_read(filter_table)
    flist['PHOTPLAM'].unit = units.nm
    flist['PHOTFLAM'].unit = units.Unit('W m-2 um-1')
    flist['PHOTBW'].unit = units.nm
    flist['SolarFlux'].unit = units.Unit('W m-2 um-1')
    return flist

def filter_bandpass(flt):
    '''Return filter bandpass in a Table'''
    from os.path import isfile
    from astropy.io import fits
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
    from astropy.io import ascii, fits
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

    from astropy.io import ascii, fits
    import astropy.units as u

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


class UVISImage(Image): pass


def read_uvis(inputfile):
    from astropy.io import fits
    if not (inputfile.endswith('_flt.fits') or inputfile.endswith('_drz.fits')):
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
        obj = UVISImage(sci, unit='electron', uncertainty=StdDevUncertainty(err), mask=(dq != 0), meta=header, flags=dq)
    else:  # _drz file
        obj = UVISImage(fitsfile[0].data, unit='electron/s', meta=header)
        header['sci'] = fitsfile[0].header
        header['wht'] = fitsfile[1].header
        header['ctx'] = fitsfile[2].header
        flags = nddata.FlagCollection(shape=fitsfile[0].data.shape)
        flags['wht'] = fitsfile[1].data
        flags['ctx'] = fitsfile[2].data
        obj = ccdproc.create_deviation(obj, gain=header['primary']['exptime']*units.s, readnoise=3*units.electron)

    return obj


class UVISCalibration(object):

    pamfits = [wfc3dir+'Pixel_Area_Map/UVIS'+str(x)+'wfc3_map.fits' for x in [1,2]]
    pxlscl = 0.04 * units.arcsec

    def __init__(self):
        self.pam = [readfits(f,ext=1) for f in self.pamfits]
        self.photcal = load_filter()


def load_pam(aperture=None, filter=None):
    '''Load pixel area map

    v1.0.0: JYL @PSI, 5/4/2016'''

    if aperture == None:
        aperture = 'UVIS'
    aperture = aperture.lower()
    if aperture.upper() not in load_aperture()['Aperture']:
        raise ValueError('invalid aperture {0}'.format(aperture))

    amps = [['FQ387N','FQ437N','FQ508N','FQ619N','FQ889N'],
            ['FQ378N','FQ492N','FQ674N','FQ750N','FQ937N'],
            ['FQ232N','FQ422M','FQ575N','FQ634N','FQ906N'],
            ['FQ243N','FQ436N','FQ672N','FQ727N','FQ924N']]

    pamfits = [wfc3dir+'Pixel_Area_Map/UVIS'+str(x)+'wfc3_map.fits' for x in [1,2]]
    if aperture.find('uvis1') != -1:
        pams = readfits(pamfits[0],ext=1,verbose=False)
    elif aperture.find('uvis2') != -1:
        pams = readfits(pamfits[1],ext=1,verbose=False)
    elif aperture.find('quad') != -1:
        if filter == None:
            raise ValueError('`filter'' has to be specified for ''QUAR'' apertures')
        if (filter in amps[0]) or (filter in amps[1]):
            pams = readfits(pamfits[0],ext=1,verbose=False)
        elif (filter in amps[2]) or (filter in amps[3]):
            pams = readfits(pamfits[1],ext=1,verbose=False)
    else:
        pams = np.concatenate([readfits(x,ext=1,verbose=False) for x in pamfits][::-1])
    if aperture.find('c1k1c') != -1:
        return pams[:1024,:1025]
    if aperture.find('c512c') != -1:
        return pams[:512,:512]
    if aperture.find('m1k1c') != -1:
        return pams[-1024:,-1025:]
    if aperture.find('m512c') != -1:
        return pams[-512:,-512:]
    if (aperture.find('2k2a') != -1) or (aperture.find('2k2c') != -1):
        return pams[1:,:2047]
    if (aperture.find('2k2b') != -1) or (aperture.find('2k2d') != -1):
        return pams[1:,2049:]
    if aperture.find('quad') != -1:
        if (filter in amps[0]) or (filter in amps[2]):
            return pams[1:,:2047]
        else:
            return pams[1:,2049:]
    return pams


def read_jit(fn):
    '''Read jitter data and return in a Table

    v1.0.0 : 5/1/2016, JYL @PSI
    '''

    jit = Table().read(fn)
    for k in list(jit.keys()):
        if jit[k].unit == 'seconds':
            jit[k].unit = units.s
        elif jit[k].unit == 'arcsec':
            jit[k].unit = units.arcsec
        elif jit[k].unit == 'degrees':
            jit[k].unit = units.deg
        elif jit[k].unit == 'Gauss':
            jit[k].unit = units.G

    return jit


def load_aperture(show_in_browser=False):
    '''Load WFC3 aperture file
    Source of aperture file: http://www.stsci.edu/hst/observatory/apertures/wfc3.html
    Updated June 21, 2012

    v1.0.0 : 5/1/2016, JYL @PSI
    '''

    aper = ascii_read(wfc3dir+'Aperture_File.csv')
    aper['v2pos'].unit = units.arcsec
    aper['v3pos'].unit = units.arcsec
    aper['xscl'].unit = units.arcsec
    aper['yscl'].unit = units.arcsec
    aper['v3x'].unit = units.deg
    aper['v3y'].unit = units.deg
    if show_in_browser:
        aper.show_in_browser()
    return aper


def obslog(files):
    '''Extract observation log from jif and jit files

    Input files are either the rootname, or full image names, in both
    cases with the full directory path

    '''
    from astropy.io import fits
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
            log[k].unit = units.Unit(u)

    return log
