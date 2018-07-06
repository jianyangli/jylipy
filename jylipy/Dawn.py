# Dawn facilities
#
#


import numpy as np, string, spiceypy as spice
from copy import copy
import ccdproc
from .core import *
from .vector import xyz2sph #Image, readfits, condition, xyz2sph, Time, num, findfile, writefits, CCDData, ImageMeasurement, CaseInsensitiveOrderedDict, findfile, ascii_read
from .geometry import subcoord
from .apext import Table, Column, units, fits
from . import convenience as conv, PDS
from os import remove
from os.path import join
from jylipy import pysis_ext, Photometry


# Dawn related variables
DAWN_NAIF_CODE = -203
DAWN_DIR = conv.dir.work+'Dawn/'
DAWN_CERES_KERNEL = conv.dir.work+'Dawn/spice/meta/dawn_ceres.kernel'
FC_ASPECT = DAWN_DIR+'data/fc2/fc_aspect.csv'

#datadir_config = DAWN_DIR+'data/Ceres_datasets.csv'
# temperary file directory
tempdir = '/Users/jyli/temp/'

# Ceres constants
class Ceres(object):
    # body shape and pole based on DLR RC3 shape model
    ra = 482.64 * units.km
    rb = 480.60 * units.km
    rc = 445.57 * units.km
    r = np.sqrt((ra+rb)*rc/2)
    pole = (291.41, 66.79)
    GM = 62.68 * units.Unit('km3/s2')
    M = (GM/constants.G).decompose()

# Convenience functions
def load_dawn_kernels(kernel=DAWN_CERES_KERNEL):
    spice.furnsh(kernel)

def unload_dawn_kernels(kernel=DAWN_CERES_KERNEL):
    spice.unload(kernel)


# Classes

u_iof = units.def_unit('iof')


class Data(object):
    '''Dawn data fetching class'''

    indexfile = DAWN_DIR+'data/dawn_data_catalog.csv'

    allinstruments = ['fc', 'vir']
    alllevels = ['1a', '1b', '1c']
    allphases = ['csa', 'csr', 'cts', 'css', 'cth', 'csh', 'ctl', 'csl', 'cxl', 'cxj', 'cxg', 'cto', 'cx2', 'c2i', 'c2e', 'vsa', 'vss', 'vth', 'vsh', 'vtl', 'vsl', 'vt2', 'vh2', 'vtc']

    rmtroot = 'dscws.igpp.ucla.edu:/data/DSDb/data'
    localroot = '/Volumes/Dawn_Data'

    def __init__(self, instrument=allinstruments, level=alllevels, phase=allphases, quicklook=False):
        self.index = ascii_read(self.indexfile)
        if (not isinstance(instrument, (str,bytes))) and hasattr(instrument, '__iter__'):
            self.instrument = list(instrument)
        else:
            self.instrument = [instrument]
        if not set(self.instrument).issubset(set(self.allinstruments)):
            raise ValueError('{0} contains unrecognized instrument'.format(self.instrument))
        if (not isinstance(level, (str,bytes))) and hasattr(level, '__iter__'):
            self.level = list(level)
        else:
            self.level = [level]
        if not set(self.level).issubset(set(self.alllevels)):
            raise ValueError('{0} contains unrecognized level'.format(self.level))
        if (not isinstance(phase, (str,bytes))) and hasattr(phase, '__iter__'):
            self.phase = list(phase)
        else:
            self.phase = [phase]
        if not set(self.phase).issubset(set(self.allphases)):
            raise ValueError('{0} contains unrecognized phase'.format(self.phase))
        self.quicklook = quicklook

    def reload_index(self):
        self.index = ascii_read(self.indexfile)

    def tags(self):
        tg = []
        for i in self.instrument:
            for l in self.level:
                for p in self.phase:
                    if p[0] == 'c':
                        t = 'Ceres'
                    else:
                        t = 'Vesta'
                    tg.append([l, i, p, t])
        return tg

    def remote_path(self):
        rmtdir = []
        tgs = self.tags()
        for tg in self.tags():
            if tg[0] == '1a':
                ld = '1a_edr'
            elif tg[0] == '1b':
                ld = '1b_rdr'
            else:
                ld = '1c_cdr'
            if tg[2][0] == 'c':
                t = 'ceres'
            else:
                t = 'vesta'
            if self.quicklook:
                rmtdir.append('/'.join([self.rmtroot, '1a_quicklook', tg[1], t, tg[2]]))
            else:
                rmtdir.append('/'.join([self.rmtroot, ld, tg[1], t, tg[2]]))
        return rmtdir

    def local_path(self):
        lcldir = []
        for tg in self.tags():
                if tg[0] == '1a':
                    ld = 'level-1a'
                elif tg[0] == '1b':
                    ld = 'level-1b'
                else:
                    ld = 'level-1c'
                if tg[1] == 'fc':
                    tg[1] = 'FC2'
                if self.quicklook:
                    lcldir.append('/'.join([self.localroot, tg[3], tg[1].upper(), 'quicklook']))
                else:
                    lcldir.append('/'.join([self.localroot, tg[3], tg[1].upper(), ld]))
        return lcldir

    def fetch(self, fmt='IMG', delete=False):
        '''Fetch Dawn data'''
        from subprocess import call
        from os.path import isdir
        from os import mkdir

        rmtpath = self.remote_path()
        lclpath = self.local_path()
        tags = self.tags()

        print('Fetching data from DSDb')
        print('Level:', self.level)
        print('Instrument:', self.instrument)
        print('Phase:', self.phase)
        print('Total number of dataset:', len(tags))
        print()

        for tg, rp, lp in zip(tags, rmtpath, lclpath):
            datasets = self.index.query('Level',tg[0]).query('Instrument',tg[1]).query('Phase',tg[2])
            for d in datasets:
                cmd = ['rsync']
                cmd.append('-auvz')
                cmd.append('--progress')
                if delete:
                    cmd.append('--delete-after')
                # FC data
                if tg[1] == 'fc':
                    print()
                    print('Dataset:', d['Sequence'])
                    print()
                    # Quicklook
                    if self.quicklook:
                        if d['Quicklook_dir'] is np.ma.masked:
                            print('Data directory not specified on DSDb.  Skipped.')
                            continue
                        cmd.append('--exclude=FITS')
                        cmd.append('--exclude=*zip')
                        cmd.append('--exclude=JPEG')
                        cmd.append('--exclude=*xml')
                        remdir = '/'.join([rp, d['Quicklook_dir']])+'/'
                        locdir = '/'.join([lp, d['local_dir']])+'/'
                    # Normal
                    else:
                        if d['DSDb_dir'] is np.ma.masked:
                            print('Data directory not specified on DSDb.  Skipped.')
                            continue
                        remdir = '/'.join([rp, d['DSDb_dir'], fmt])+'/'
                        locdir = '/'.join([lp, d['local_dir']])+'/'
                # VIR data
                else:
                    print()
                    print('Dataset:', d['Sequence'], ', ', d['Band'])
                    print()
                    if d['DSDb_dir'] is np.ma.masked:
                        print('Data directory not specified on DSDb.  Skipped.')
                        continue
                    remdir = '/'.join([rp, d['DSDb_dir']])+'/'
                    locdir = '/'.join([lp, d['local_dir']+'_'+d['Band']])+'/'

                # Fetch data
                cmd.append(remdir)
                cmd.append(locdir)
                if not isdir(locdir):
                    mkdir(locdir)
                call(cmd)


class FCCalibration(object):
    '''FC image calibration

 FC calibration constants are mostly from Schroeder et al. (2013)
 Icar 226, 1304-1317, including `photcal`, `sunflux`, `wvctr`,
 `wvflux`.

 The constants `wviof` is calculated by JYL based on the filter
 transmission and solar spectrum.

 For F1, JYL calculated
    photcal(F1) = 3.479e7 adu/(J m-2 nm-1 sr-1),
    sunflux(F1) = 1.341 W m-2 nm-1
 using filter transmission and solar spectrum
 The `photcal` values used here is based on Schroeder et al. (2013)
 with a unit of adu/(J m-2 sr-1), and `sunflux` are scaled from
 JYL's calculation in W m-2 nm-1 to W m-2 to yield the same I/F.

 v1.0.0 : JYL @PSI, early 2015
 v1.0.1 : JYL @PSI, 10/26/2015
   Adopted calibration constant values from Schroeder et al. (2013)
     calibration paper.  The values previously used were based on
     JYL's calculation
   It is discovered that the level-1b data of F1 from MPS have values
     that are consistent with unit 'W m-2 nm-1 sr-1' rather than
     'W m-2 sr-1' as noted in headers.  The calibration constants for
     this particular filters still use the values by JYL because the
     MPS official value is unknown.
   *** Important: However, this means that the units for calibrated
     F1 flux images are wrong!  The values for MPS level 1b images
     appear to be consistent with JYL calibration.  Therefore the I/F
     calibration is not be affected.
    '''

    photcal = list([3.479e7, 1.88e6, 3.85e6, 1.82e6, 1.76e6, 2.36e6, 3.13e6, 2.18e5]*units.adu/'J m-2 nm-1 sr-1')
    photcal[0] = photcal[0].value*units.adu/'J m-2 sr-1'
    sunflux = list([1.341,  1.863,  1.274,  0.865,  0.785,  1.058,  1.572,  1.743]*units.Unit('W m-2 nm-1'))
    sunflux[0] = sunflux[0].value*units.Unit('W m-2')
    corr = [1.11, 1.13, 1.10, 1.10, 1.15, 1.13, 1.30, 1.45]
    wvctr = [735., 549, 749, 919, 978, 829, 650, 428]*units.nm
    wvflux = [732., 555, 749, 917, 965, 829, 653, 438]*units.nm
    wviof = [698.78, 553.53, 748.34, 915.57, 960.83, 828.19, 651.90, 437.58]*units.nm
    linetime = 1.25e-6  # Line time for readout smear correction
    darkfile = '/Users/jyli/work/Dawn/data/cal/DWNCALFC2/DATA/20121213_FC2_VCC_DC053_V01.IMG'
    flatfile = ['/Users/jyli/work/Dawn/data/cal/DWNCALFC2/DATA/FC2_F'+str(x)+'_FLAT_V02.IMG' for x in range(1,9)]

    def __init__(self):
        from . import PDS
        self.dark = PDS.readpds(self.darkfile).image
        self.flat = [PDS.readpds(x).image for x in self.flatfile]

    @staticmethod
    def _check_fcimage(image):
        if not isinstance(image, FCImage):
            raise TypeError('input image is not FCImage type')
        if issubclass(image.data.dtype.type, np.integer):
            image.data = image.data.astype('f4')

    @staticmethod
    def flag_saturation(image):
        FCCalibration._check_fcimage(image)
        if 'flag_saturation' in list(image.header.keys()):
            return image
        sat = image.data >= 16380
        if image.mask is None:
            image.mask = sat
        else:
            image.mask = image.mask | sat
        satflag = np.zeros(image.shape, dtype='byte')
        satflag[sat] = 1
        if image.flags is None:
            image.flags = satflag
        else:
            image.flags.append(satflag)
        image.header['flag_saturation'] = 'flag saturation by the first bit'
        return image

    @staticmethod
    def remove_bias(image):
        FCCalibration._check_fcimage(image)
        if 'subtract_bias' in list(image.header.keys()):
            return image
        bias = np.asarray(image.frame_2_image).astype(float).mean()
        image.data -= bias
        image.header['subtract_bias'] = 'bias = {0}'.format(bias)
        return image

    def remove_dark(self, image):
        self._check_fcimage(image)
        if 'subtract_dark' in list(image.header.keys()):
            return image
        temp = image.header['DETECTOR_TEMPERATURE'].value
        texp = image.header['EXPOSURE_DURATION'].to('s')
        dark = self.dark*np.exp(0.172*(temp-221.0))
        image.data -= dark
        image.header['subtract_dark'] = 'dark = {0}'.format(self.darkfile)
        return image

    def smear_correct(self, image):
        self._check_fcimage(image)
        if 'smear_correct' in list(image.header.keys()):
            return image
        texp = image.header['EXPOSURE_DURATION'].to('s').value
        image.data[1] -= image.data[0]*self.linetime/texp
        for i in range(2,len(image.data)):
            image.data[i] -= image.data[:i-1,:].sum(axis=0)*self.linetime/texp
        image.header['smear_correct'] = 'exposure={0}'.format(texp)
        return image

    def flatfield(self, image):
        self._check_fcimage(image)
        if 'flat_correct' in list(image.header.keys()):
            return image
        fno = num(image.header['FILTER_NUMBER'])-1
        image.data /= self.flat[fno]
        image.header['flat_correct'] = 'flat = {0}'.format(self.flatfile[fno])
        return image

    def fluxcal(self, image):
        self._check_fcimage(image)
        if 'flux_cal' in list(image.header.keys()):
            return image
        texp = image.header['EXPOSURE_DURATION']
        fno = num(image.header['FILTER_NUMBER'])
        if fno == 1:
            to_unit = 'W m-2 sr-1'
        else:
            to_unit = 'W m-2 nm-1 sr-1'
        if 'iof_cal' in list(image.header.keys()):
            if 'rh' not in image.geometry:
                image.calcgeom()
            rh = image.geometry['rh'].value
            sflx = self.sunflux[fno-1]/rh**2
            image.data *= (image.unit*sflx/(np.pi*units.Unit('sr'))).to(to_unit).value
        else:
            photcal = self.photcal[fno-1]
            image.data *= (image.unit/(texp*photcal)).to(to_unit).value
        image.unit = to_unit
        image.header['flux_cal'] = 'fluxcal, photcal={0}, solar flux={1}'.format(self.photcal[fno], self.sunflux[fno])
        image.header.pop('iof_cal',None)
        return image

    def iofcal(self, image):
        self._check_fcimage(image)
        if 'iof_cal' in list(image.header.keys()):
            return image
        if 'rh' not in image.geometry:
            image.calcgeom()
        rh = image.geometry['rh'].value
        texp = image.header['EXPOSURE_DURATION']
        fno = num(image.header['FILTER_NUMBER'])
        sflx = self.sunflux[fno-1]/rh**2
        photcal = self.photcal[fno-1]
        if 'flux_cal' in list(image.header.keys()):
            image.data *= (image.unit*np.pi*units.Unit('sr')/sflx).to('')
        else:
            image.data *= (image.unit*np.pi*units.Unit('sr')/(texp*photcal*sflx)).to('')
        image.unit = ''
        image.header['iof_cal'] = 'iofcal, photcal={0}, solar flux={1}, rh={2}'.format(photcal, sflx, rh)
        image.header.pop('flux_cal', None)
        return image

    @staticmethod
    def add_mask(image, mask):
        if image.mask is None:
            image.mask = mask
        else:
            image.mask = image.mask | mask

    @staticmethod
    def add_flag(image, flag):
        if image.flags is None:
            image.flags = flag
        else:
            image.flags.append(flag)

    def calibrate(self, image, flux=False, iof=False, mask=None, flag=None):
        if image.unit == 'adu':  # Start from level-1a
            self.flag_saturation(image)
            self.remove_bias(image)
            self.remove_dark(image)
            self.smear_correct(image)
            self.flatfield(image)
        else: # Start from level-1b or 1c
            # mask
            if mask is not None:
                FCCalibration.add_mask(image, mask)
            # flag
            if flag is not None:
                FCCalibration.add_flag(image, flag)
        load_dawn_kernels()
        if flux:
            self.fluxcal(image)
        elif iof:
            self.iofcal(image)
        unload_dawn_kernels()
        return image

    def __call__(self, *args, **kwargs):
        return self.calibrate(*args, **kwargs)


class FCImage_old(ImageMeasurement):

    def __new__(cls, inputfile, dtype=None):
        from astropy.io import fits
        from . import PDS
        if inputfile.endswith('.LBL'):
            inputfile = inputfile.replace('.LBL','.FIT')
        if inputfile.endswith('.FIT'):
            fitsfile = fits.open(inputfile)
            data = fitsfile[0].data
            obj = ImageMeasurement(data, dtype=dtype).view(FCImage)
            lblfile = inputfile.replace('.FIT','.LBL')
            from os.path import isfile
            if isfile(lblfile):
                obj.header = PDS.Header(lblfile)
            else:
                obj.header = None
        elif inputfile.endswith('.IMG'):
            data = PDS.PDSData(inputfile)
            obj = getattr(data, data.records[0]).copy().view(FCImage)
            obj.header = data.header
            if len(data.records) > 1:
                obj.records = []
                for k in data.records[1:]:
                    obj.__dict__[k] = getattr(data, k).copy()
                    obj.records.append(k)
        else:
            raise IOError('input file not recognized')
        obj.source = inputfile
        obj.geometry = {}
        obj.spice = DAWN_CERES_KERNEL
        obj.calibration = {}
        return obj

    def __array_finalize__(self, obj):
        super(FCImage, self).__array_finalize__(obj)
        if obj is None: return
        self.source = getattr(obj, 'source', None)
        self.header = getattr(obj, 'header', None)
        self.geometry = getattr(obj, 'geometry', {})
        self.spice = getattr(obj, 'spice', None)
        self.calibration = getattr(obj, 'calibration', {})
        if hasattr(obj, 'records'):
            self.records = copy(obj.records)
            for k in obj.records:
                self.__dict__[k] = copy(getattr(obj, k))

    def _load_spice_kernels(self, kernel=None):
        if kernel is not None:
            if (not isinstance(kernel, (str,bytes))) and hasattr(kernel, '__iter__'):
                for k in kernel:
                    spice.furnsh(k)
            else:
                spice.furnsh(kernel)
            self.spice = kernel
        else:
            raise spice.utils.support_types.SpiceyError('kernel(s) not specified')

    def _unload_spice_kernels(self, kernel=None):
        if kernel is not None:
            if (not isinstance(kernel, (str,bytes))) and hasattr(kernel, '__iter__'):
                for k in kernel:
                    spice.unload(k)
            else:
                spice.unload(kernel)
        else:
            raise spice.utils.support_types.SpiceyError('kernel(s) not specified')

    def calcgeom(self, kernel=None):
        if kernel is not None: self._load_spice_kernels(kernel)
        t1 = spice.scs2e(DAWN_NAIF_CODE, self.header['SPACECRAFT_CLOCK_START_COUNT'])
        t2 = spice.scs2e(DAWN_NAIF_CODE, self.header['SPACECRAFT_CLOCK_STOP_COUNT'])
        t = Time((t1+t2)/2, format='et')
        geom = subcoord(t.isot, self.header['TARGET_NAME'].split()[1], observer='Dawn')
        for k in list(geom.keys()):
            self.geometry[k] = geom[k][0]*condition(geom[k].unit is None, 1, geom[k].unit)
        self.geometry['Range'] = self.geometry['Range'].to('km')
        try:
            m = np.asarray(spice.sxform('j2000','dawn_fc2', t.et))
            self.geometry['CelN'] = (360+270-xyz2sph(m[:3,:3].dot([0,0,1]))[1]) % 360 *units.deg
        except spice.utils.support_types.SpiceyError:
            print('CK not available for '+t.isot)
        if kernel is not None: self._unload_spice_kernels(kernel)


class FC2(object):
    gain = 17.7 * units.electron/units.adu
    readnoise = 1.14*17.7 * units.electron
    ifov = 93.7 * units.urad
    dark = 0.05 * units.adu


class FCImage(Image):
    '''FC Image class'''

    def __init__(self, data):
        '''FCImage class can be initialized with the same signature as
        ccdproc.CCDData class, or a str containing the input file name.

        data : str, PDS.PDSData, or FCImage
        '''
        from os.path import isfile
        if isinstance(data, FCImage):
            super(FCImage, self).__init__(data)
            self._copy_properties(data)
        elif isinstance(data, PDS.PDSData):
            self._init_from_pdsdata(data)
        elif isinstance(data, str):
            self.read(data)
        else:
            raise TypeError('a filename or a 2-D ndarray is required to initialize FCImage, {0} received'.format(type(data)))
        if not hasattr(self, 'geometry'):
            self.geometry = {}

    def _copy_properties(self, data):
        self.records = getattr(data, 'records', None)
        if self.records is not None:
            for k in self.records:
                setattr(self, k, getattr(data, k))

    def _init_from_pdsdata(self, inputfile):
        pdsdata = PDS.PDSData(inputfile)
        data = pdsdata.image
        header = pdsdata.header
        header.update(data.header)
        super(FCImage, self).__init__(data, meta=header)
        if self.unit == 'adu':  # if Level 1a data, add uncertainty plane
            self.uncertainty = ccdproc.create_deviation(self, gain=FC2.gain, readnoise=FC2.readnoise).uncertainty
            self.meta['create_deviation'] = 'ccd_data=<FCImage>, readnoise={0}, gain={1}'.format(FC2.readnoise, FC2.gain)
        else:  # If level 1b or 1c, add 'flux_cal' key to the header
            self.header['flux_cal'] = 'MPS'
        if len(pdsdata.records) > 1:
            self.records = pdsdata.records[1:]
            for k in self.records:
                setattr(self, k, getattr(pdsdata, k))
        else:
            self.records = []

    def read(self, inputfile):
        '''Input file must be in PDS format with .IMG extension'''
        from os.path import isfile
        from . import PDS
        if not isfile(inputfile):
            raise ValueError('file {0} not found'.format(data))
        if inputfile.endswith('IMG') or inputfile.endswith('LBL'):
            self._init_from_pdsdata(inputfile)
        elif inputfile.endswith('FIT'):
            self._init_from_fits(inputfile)
        else:
            raise ValueError('file type not recognized')


    def calcgeom(self):
        if self.header['TARGET_NAME'] == '':
            self.header['TARGET_NAME'] = 'ceres'
        if self.header['TARGET_NAME'].lower().find('vesta') >= 0:
            target = 'vesta'
        elif self.header['TARGET_NAME'].lower().find('ceres') >= 0:
            target = 'ceres'
        else:
            target = self.header['TARGET_NAME']
        if target in ['vesta', 'ceres']:
            t1 = spice.scs2e(DAWN_NAIF_CODE, self.header['SPACECRAFT_CLOCK_START_COUNT'])
            t2 = spice.scs2e(DAWN_NAIF_CODE, self.header['SPACECRAFT_CLOCK_STOP_COUNT'])
            t = Time((t1+t2)/2, format='et')
            geom = subcoord(t.isot, target, observer='Dawn')
            for k in list(geom.keys()):
                self.geometry[k] = geom.getcolumn(k)[0]
            self.geometry['Range'] = self.geometry['Range'].to('km')
            try:
                m = np.asarray(spice.sxform('j2000','dawn_fc2', t.et))
                self.geometry['celn'] = (360+270-xyz2sph(m[:3,:3].dot([0,0,1]))[1]) % 360 *units.deg
            except spice.utils.support_types.SpiceyError:
                print('CK not available for '+t.isot)


    def write(self, filename, **kwargs):
        if filename.lower().endswith('fits') or filename.lower().endswith('fit'):
            self._writefits(filename, **kwargs)
        else:
            raise ValueError('file extension unrecognized')

    def _writefits(self, filename, **kwargs):
        from astropy.io import fits
        out = fits.HDUList()
        hdu = fits.PrimaryHDU(self.data)
        hdu.header['bunit'] = str(self.unit)
        out.append(hdu)
        if self.flags is not None:
            out.append(fits.ImageHDU(self.flags.astype('uint8'), name='Flag'))
        out.writeto(filename, **kwargs)

    def _init_from_fits(self, filename):
        from astropy.io import fits
        fitsdata = fits.open(filename)
        data = fitsdata[0].data
        flag = fitsdata[1].data
        if 'bunit' in list(fitsdata[0].header.keys()):
            unit = units.Unit(fitsdata[0].header['bunit'])
        else:
            unit = ''
        super(FCImage, self).__init__(data, unit=unit, flags=flag, mask=(flag==1))


def aspect(files, saveto=None):
    '''Return aspect data in a table'''
    from os.path import basename, isfile, isdir
    from astropy.io import ascii
    from astropy import table
    name, oid, fid, utc, flt, texp = [], [], [], [], [], []
    geom = {}
    if isinstance(files, (str,bytes)):
        if isdir(files):
            files = findfile(files,'IMG')+findfile(files,'LBL')
        else:
            files = [files]
    load_dawn_kernels()
    for f in files:
        im = FCImage(f)
        im.calcgeom()
        name.append(basename(f))
        fid.append(im.header['PRODUCT_ID'])
        oid.append(im.header['OBSERVATION_ID'])
        utc.append(im.geometry['Time'])
        flt.append(num(im.header['FILTER_NUMBER']))
        texp.append(im.header['EXPOSURE_DURATION'].to('s'))
        for k in im.geometry:
            if k not in geom:
                geom[k] = []
            geom[k].append(im.geometry[k])
    unload_dawn_kernels()
    for k in geom:
        if isinstance(geom[k][0],units.Quantity):
            geom[k] = units.Quantity(geom[k])
    geom['pxlscl'] = (geom['Range']*93.7e-6).to('m')
    texp = units.Quantity(texp)
    fid = np.array(fid,dtype='int')
    tbl = Table([name, fid, oid, utc, flt, texp], names='Name FileID ObsID UTC Filter Texp'.split())
    geomkeys = zip('pxlscl Range rh Phase PolePA PoleInc SunPA SunInc SOLat SOLon SSLat SSLon'.split(), 'PxlScl Range Rh Phase PolePA PoleInc SunPA SunInc SCLat SCLon SSLat SSLon'.split(), '%.2f %.1f %.4f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'.split())
    for k, n, f in geomkeys:
        tbl.add_column(Column(geom[k],name=n,format=f))
    if 'celn' in list(geom.keys()):
        tbl.add_column(Column(geom['celn'],name='CelN',format='%.2f'), index=10)
    if saveto is not None:
        if isfile(saveto):
            asp = ascii_read(saveto)
            asp.update(tbl,'FileID',sort='FileID')
            for k in 'PxlScl Phase CelN PolePA PoleInc SunPA SunInc SCLat SCLon SSLat SSLon'.split():
                asp[k].format='%.2f'
            asp['Rh'].format='%.4f'
            asp['Range'].format='%.1f'
            from os import rename
            from os.path import splitext
            f,e = splitext(saveto)
            rename(saveto, ''.join([f+'_bck',e]))
            asp.write(saveto)
        else:
            tbl.write(saveto)
    return tbl


def fc2cal(indata, outdata, level='dn', verbose=True, overwrite=False, mask=None, flag=None):
    '''Calibrate all data in `indata` and save calibrated data to `outdata`

    `indata` could be a single image file, or a directory.  If directory,
    then all files in it, or all subdirectories, will be processed
    iteratively.

    `outdata` should correspond to `indata`, i.e., if `indata` is a file/dir,
    then `outdata` should be a file/dir.
    '''

    if level not in ['dn', 'iof', 'flux']:
        raise ValueError('Calibration level {0} not recognized'.format(level))
    iof = condition(level == 'iof', True, False)
    flux = condition(level == 'flux', True, False)
    if verbose:
        print('Target calibration level: ', level)

    from os.path import isdir, isfile, basename, dirname, join
    from os import makedirs
    if isfile(indata):
        if verbose:
            print('calibrating image ', basename(indata))
        cal = FCCalibration()
        im = FCImage(indata)
        if mask is not None:
            mask = readfits(mask, verbose=False)
        if flag is not None:
            flag = readfits(flag, verbose=False)
        cal(im, flux=flux, iof=iof, mask=mask, flag=flag)
        outpath = dirname(outdata)
        outfile = basename(outdata)
        if not isdir(outpath):
            makedirs(outpath)
        im.write(join(outpath,outfile), clobber=overwrite)
    elif isdir(indata):
        insidedir = findfile(indata,dir=True)
        insidefile = findfile(indata)
        insidefile = [x for x in insidefile if x.endswith('IMG') or x.endswith('img')]
        if mask is not None:
            maskfile = findfile(mask, 'fits')
            maskfile_id = [x.split('/')[-1][5:12] for x in maskfile]
            maskdir = findfile(mask, dir=True)
        if flag is not None:
            flagfile = findfile(flag)
            flagfile_id = [x.split('/')[-1][5:12] for x in flagfile]
            flagdir = findfile(flag, dir=True)
        if len(insidefile) > 0:
            if verbose:
                print()
                print('Directory {0}: {1} files found'.format(basename(indata), len(insidefile)))
            for fi in insidefile:
                ext = fi.split('.')[-1]
                fo = join(outdata, basename(fi).replace('1A0','1B0').replace(ext, 'fits'))
                fi_id = fi.split('/')[-1][5:12]
                msf = None
                if mask is not None:
                    if fi_id in maskfile_id:
                        msf = maskfile[maskfile_id.index(fi_id)]
                flf = None
                if flag is not None:
                    if fi_id in flagfile_id:
                        flf = flagfile[flagfile_id.index(fi_id)]
                fc2cal(fi, fo, level=level, verbose=verbose, overwrite=overwrite, mask=msf, flag=flf)
        if len(insidedir) > 0:
            if verbose:
                print()
                print('Directory {0}: {1} subdirectories found'.format(basename(indata), len(insidedir)))
            maskdir_base = []
            flagdir_base = []
            if mask is not None:
                maskdir_base = [x.split('/')[-1] for x in maskdir]
            if flag is not None:
                flagdir_base = [x.split('/')[-1] for x in flagdir]
            for di in insidedir:
                print('Processing directory ', basename(di))
                do = join(outdata, basename(di).replace('L1A','L1B'))
                di_base = di.split('/')[-1]
                msd = None
                fld = None
                if di_base in maskdir_base:
                    msd = maskdir[maskdir_base.index(di_base)]
                if di_base in flagdir_base:
                    fld = flagdir[flagdir_base.index(di_base)]
                fc2cal(di, do, level=level, verbose=verbose, overwrite=overwrite, mask=msd, flag=fld)
    else:
        raise ValueError('input not found')


def diskint_phot(asp, indata, outfile=None, threshold=0.003, ext='.fits', colnames='Name FileID ObsID UTC Filter Rh Range Phase SCLat SCLon SSLat SSLon'.split(), dtypes='S40 int S25 S30 int float float float float float float float'.split()):
    '''Measure the disk-integrated photometry of Ceres

 asp : aspect data table as generated by aspect()
 datadir : directory of input data (calibrated to I/F)
 outfile : output data table

 4/29/2015, JYL @PSI
 5/31/2015, JYL @PSI
   Add column `Filter` to output table
   Changed the background meausrement from the resistant mean in
   [50:200, 50:200] to the avearge of resistant means in four corners
   corresponding to [20:150, 20:150] in the lower left.
    '''
    from os.path import isdir, isfile, basename, dirname, join
    from astropy import table

    if isdir(indata):
        print('processing directory: ', basename(indata))
        tbls = []
        files = findfile(indata, ext)
        for f in files:
            fid = int(basename(f)[5:12])
            if fid not in asp['FileID']:
                continue
            print('  image: ', basename(f))
            phot = diskint_phot(asp, f, outfile=outfile, threshold=threshold, colnames=colnames, dtypes=dtypes)
            if phot is not None:
                tbls.append(phot)
        tbl = table.vstack(tbls)
        if outfile is not None:
            tbl.write(outfile)
        return tbl
    elif isfile(indata):
        fid = int(basename(indata)[5:12])
        tbl = asp.query('FileID', fid, colnames).copy()
        im = FCImage(indata)
        cr = Ceres.r.value/(tbl['Range'][0]*93.7e-6)  # Ceres radius
        if len(np.where(im.mask.flatten())[0])/(np.pi*cr*cr) > 0.001:
            print('    saturated, skipped')
            return None
        yc, xc = geometric_center(im.data, threshold)
        rad = cr*3
        bg = (resmean(im.data[20:150,20:150])+resmean(im.data[20:150,-150:-20])+resmean(im.data[-150:-20,20:150])+resmean(im.data[-150:-20,-150:-20]))/4
        im.data -= bg
        iof = np.array(im.data[np.clip(yc-rad,0,1024):np.clip(yc+rad,0,1024), np.clip(xc-rad,0,1024):np.clip(xc+rad,0,1024)].sum())/(np.pi*cr**2)
        #iof = units.Quantity(iof).value
        tbl.add_column(Column([iof], name='IoF',format='%.6f'))

        # Calculate latitude correction numerically
        x = Ceres.ra.value*np.cos(np.linspace(0,2*np.pi,721))
        y = Ceres.rc.value*np.sin(np.linspace(0,2*np.pi,721))
        for lat in tbl['SCLat']:
            x1, y1 = x*np.cos(np.deg2rad(lat))+y*np.sin(np.deg2rad(lat)), -x*np.sin(np.deg2rad(lat))+y*np.cos(np.deg2rad(lat))
            corr = (2*Ceres.rc.value)/(y1.max()-y1.min())
        tbl.add_column(Column([corr], name='LatCorr', format='%.4f'))

        return tbl


def fc_fits2cube(infile, outfile, rawdata=DAWN_DIR+'data/fc2/level-1a/', tempdir=tempdir, logfile=None, spice=True, **kwargs):
    '''Convert FC images that have been processed (e.g., calibration)
 in FITS format to ISIS cube.  The labels are generated from the
 corresponding raw data and copied over.

 Process:
   1. Convert raw IMG data to ISIS CUB.
   2. Convert input FITS data to ISIS CUB, and flip it in vertical direction
   3. Copy the ISIS labels from CUB in step 1 to CUB in step 2
   4. spiceinit the output CUB

 Parameters
 ----------
 infile, outfile : str
   Input and output file names
 rawdata : str or list, optional
   If str, then the root directory to store all raw data
   If list, then the list of the full paths to all raw files
 tempdir : str, optional
   Directory name to save intermediate files.  By default, all
   intermediate files will be deleted.
 logfile : str, optional
   Log file.
 spice : bool, optional
   If `True`, run spiceinit to the final cubes
 **kwargs :
   Other keywords that are accepted by isis.spiceinit

 v1.0.0, 10/26/2015, JYL @PSI
 v1.0.1, 1/11/2016, JYL @PSI
   Added keyward `spice`
    '''

    from os.path import isfile, basename, join
    from os import makedirs, remove
    from .pysis_ext import dawnfc2isis, fits2isis, flip, copylabel, spiceinit

    if not isfile(infile):
        raise ValueError('Input file not found: {0}'.format(infile))

    if logfile is not None:
        log = {'-log': logfile}
    else:
        log = {}

    print('Processing image ', basename(infile))

    # Find raw data
    if type(rawdata) == str:
        rawdata = findfile(rawdata, 'IMG', recursive=True)
    raw_id = [basename(x)[5:12] for x in rawdata]
    in_id = basename(infile)[5:12]
    print(in_id, raw_id[0], len(raw_id))
    if in_id not in raw_id:
        raise ValueError('Corresponding raw data not found.')
    rawfile = rawdata[raw_id.index(in_id)]

    # Convert Dawn data to ISIS cube
    print('Generating cube file from raw data:')
    rawcub = join(tempdir, basename(rawfile).replace('.IMG','.cub'))
    args = {'from': rawfile, 'to': rawcub}
    args.update(log)
    dawnfc2isis(**args)

    # Convert input FITS file to ISIS cube and flip
    print('Processing input file:')
    fitscub = join(tempdir, 'in_fits.cub')
    args = {'from': infile, 'to': fitscub}
    args.update(log)
    fits2isis(**args)
    tmp = outfile.split('.')
    if len(tmp)>1:
        tmp[-1] = 'cub'
        outfile = '.'.join(tmp)
    args = {'from': fitscub, 'to': outfile}
    args.update(log)
    flip(**args)

    # Copy labels
    print('Copying labels:')
    args = {'from': outfile, 'source': rawcub}
    args.update(log)
    copylabel(**args)

    # spiceinit
    if spice:
        print('spiceinit:')
        args = {'from': outfile}
        args.update(kwargs)
        args.update(log)
        spiceinit(**args)

    # Clean up
    remove(rawcub)
    remove(fitscub)
    if isfile('print.prt'):
        remove('print.prt')


def extract_phodata(illfile, ioffile=None, backplanes=['Phase Angle', 'Local Emission Angle', 'Local Incidence Angle', 'Latitude', 'Longitude'], outfile=None, bin=1, overwrite=False):
    '''Extract I/F data from a single image/backplane

 illdata : str or list of str
   The ISIS cube file that contains the illumination backplanes
 ioffile : str or list of str, optional
   The I/F image file, could be ISIS cube file or FITS file.  If
   specified by `ioffile` but not found, an error will be generated.
   If not specified, then the I/F data will be searched in the
   `illfile` before all geometry backplanes.  In this case, if no I/F
   data found, no error will be generated and the I/F data will simply
   not collected.
 backplanes : list of str, optional
   The names of backplanes to be extracted.
 outfile : str, optional
   The name of output file.  If `None`, then the extracted will be
   returned as an array
 bin : int, optional
   The binning size for images before extraction
 overwrite : bool, optional
   Overwrite existing output file

 v1.0.0 : 10/26/2015, JYL @PSI
    '''

    from jylipy.pysis_ext import CubeFile
    from os.path import basename, isfile
    from numpy import zeros, squeeze, isnan, where, empty, repeat, newaxis

    # List of all possible geometric backplanes generated by isis.phocube
    geo_backplanes = ['Phase Angle', 'Local Emission Angle', 'Local Incidence Angle', 'Latitude', 'Longitude', 'Emission Angle', 'Pixel Resolution', 'Line Resolution', 'Sample Resolution', 'Detector Resolution', 'North Azimuth', 'Sun Azimuth', 'Spacecraft Azimuth', 'OffNadir Angle', 'Sub Spacecraft Ground Azimuth', 'Sub Solar Ground Azimuth', 'Morphology', 'Albedo']

    # Read in and select illumination cube
    illcub = CubeFile(illfile)
    ill0 = illcub.apply_numpy_specials()
    if bin>0:
        ill0 = rebin(ill0, [1,bin,bin], axis=[0,1,2], mean=True)
    illbackplanes = [x.strip('"') for x in illcub.label['IsisCube']['BandBin']['Name']]
    for b in backplanes:
        if b not in illbackplanes:
            print('Warning: backplane {0} not found in input cube, dropped'.format(b))
            backplanes.pop(backplanes.index(b))
    ill = empty((len(backplanes),)+ill0.shape[1:])
    for i in range(len(backplanes)):
        ill[i] = ill0[illbackplanes.index(backplanes[i])]

    # Mask out invalide pixels
    masked = zeros(ill.shape[1:],dtype=bool)
    for k in backplanes:
        indx = backplanes.index(k)
        masked |= isnan(ill[indx])

    # Read in image data
    if ioffile is None:
        p = -1
        for k in illbackplanes:
            if k not in geo_backplanes:
                p += 1
            else:
                break
        if p < 0:
            print('I/F data not found.')
        im = ill0[0:p+1]
        imnames = illbackplanes[0:p+1]
    else:
        if not isfile(ioffile):
            raise IOError('I/F data not found.')
        ext = ioffile.split('.')[-1]
        if ext.lower() == 'cub':
            im = CubeFile(ioffile).apply_numpy_specials()
            if bin>0:
                im = rebin(im, [1,bin,bin], axis=[0,1,2], mean=True)
            for i in im:
                masked |= isnan(i)
        else:
            im = FCImage(ioffile)
            flags = rebin(im.flags, [bin,bin], axis=[0,1])
            im = im.data[newaxis,...]
            if bin>0:
                im = rebin(im, [1,bin,bin], axis=[0,1,2], mean=True)
            masked |= (flags != 0)
        p = im.shape[0]-1
        imnames = ['Data'+repr(i) for i in range(im.shape[0])]

    ww = where(~masked)
    if len(ww[0])>0:
        np = p+1+len(backplanes)
        out = empty((np, len(ww[0])))
        names = []
        for i in range(p+1):
            out[i] = im[i][ww]
            names.append(imnames[i])
        for i in range(len(backplanes)):
            out[p+1+i] = ill[i][ww].astype('f4')
        names.extend(backplanes)
        if outfile is not None:
            writefits(outfile, clobber=overwrite)
            for n, d in zip(names, list(out)):
                writefits(outfile, d, name=n, append=True)
        else:
            return Table(list(out), names=names)
    else:
        print('No valid data extracted.')
        return None


def collect_fc_phodata(illfile, outfile, ioffile=None, backplanes=['Phase Angle', 'Local Emission Angle', 'Local Incidence Angle', 'Latitude', 'Longitude'], bin=1, overwrite=False):
    '''Collect photometric data from many FC images.  It calls extract_phodata

    illfile : list of str
      Names of illumination backplane files
    outfile : str
      Format string of output files.  It is expected to contain `{0}`
      for the filter number to be inserted.
    ioffile : list of str, optional
      Names of corresponding I/F data files
    Other keywords are the same as extract_phodata()

    v1.0.0 : 10/26/2015, JYL @PSI
    '''
    from os.path import basename
    from numpy import where, concatenate, array, asarray

    illfile = asarray(illfile)
    if ioffile is not None:
        ioffile = asarray(ioffile)

    flts = array([int(basename(x)[25]) for x in illfile])
    names0 = ['Data0', 'Phase Angle', 'Local Emission Angle', 'Local Incidence Angle', 'Latitude', 'Longitude']
    for fno in range(8,0,-1):
        print('Processing filter ', fno)
        ww = flts == fno
        if not ww.any():
            continue
        data = {}
        for k in names0:
            data[k] = []
        if ioffile is not None:
            for illf, ioff in zip(illfile[ww], ioffile[ww]):
                print('  Extracting from ', basename(illf))
                d = extract_phodata(illf, ioff, backplanes, bin=bin)
                for k in names0:
                    data[k].append(d[k])
        else:
            for illf in illfile[ww]:
                print('  Extracting from ', basename(illf))
                d = extract_phodata(illf, backplanes=backplanes, bin=bin)
                if d is not None:
                    for k in names0:
                        data[k].append(d[k])

        print('  Saving data')
        for k in names0:
            data[k] = concatenate(data[k])
        outfilename = outfile.format(fno)
        writefits(outfilename, clobber=overwrite)
        names1 = 'I/F Pha Emi Inc Lat Lon'.split()
        for k0, k1 in zip(names0, names1):
            writefits(outfilename, data[k0], name=k1, append=True)


def sat_mask(im, out=None, flag=False, saturation=16383, forfile=False, overwrite=False):
    '''Returns the saturation mask for input image.
 This program simply search all pixels with values of `saturation`,
 and returns a bool array with those pixels marked as `True`.

 im : FCImage, or str
   Input image or input image file name
 out : variable, or str
   Returned array, or output file name
 flag : bool, optional
   If `True`, then the flag array is returned.  By default the mask
   array is returned
 saturation : number, optional
   Saturation level.  Default is for 14-bit ADC
 forfile : bool, optional
   If `True`, then the input and output should be strings
 overwrite : bool, optional
   Overwrite output file if exist

 10/26/2015, JYL @PSI
    '''
    if forfile:
        im = FCImage(im)
    FCCalibration.flag_saturation(im)
    if flag:
        result = im.flags
    else:
        result = im.mask
    if forfile:
        outsegs = out.split('.')
        if (len(outsegs) > 1) and ((outsegs[-1].lower() != 'fits') or (outsegs[-1].lower() != 'fit')):
            outsegs[-1] = 'fits'
            out = '.'.join(outsegs)
        writefits(out, result.astype('ubyte'), clobber=overwrite)
    else:
        out = result
    return out


def fc_color_mos(cubefile, order='wavelength', cccafile='CCCA.cub', cccrfile='CCCR.cub', cecfile='CEC.cub', tempdir=tempdir, log=None, **kwargs):
    '''Generate color mosaics from an FC color cube file.

    CEC (Ceres enhanced color) layers:
        1: 440
        2: 750
        3: 960

    CCCA (Ceres color composition albedo) layers:
        1: 440/750
        2: 750
        3: 960/750

    CCCAI (Inversed Ceres color composition albedo) layers:
        1: 750/440
        2: 750
        3: 960/750

    CCCR (Ceres color composition ratio) layers:
        1: 440/750
        2: 550/750
        3: 960/750

    v1.0.0 : JYL @PSI
      Based on Mineralogy WG recommendation
    '''
    from os.path import splitext, basename, dirname, join
    path = dirname(cubefile)
    base = basename(cubefile)
    base, ext = splitext(base)

    if order == 'wavelength':
        cubefile_ordered = cubefile
    elif order == 'filter':
        order = np.array([8,2,7,3,6,4,5])-1
        pysis_ext.explode(cubefile, join(path, base))
        cubs = [join(path, base)+'.band000%1i.cub' % x for x in order]
        outcube = join(path, base)+'_ordered.cub'
        pysis_ext.cubeit(cubs, outcube, tempdir=tempdir)
        for c in cubs:
            remove(cubs)
        cubefile_ordered = outcube

    pysis_ext.ratio(denominator=cubefile_ordered+'+4', numerator= cubefile_ordered+'+1',to=join(path,base)+'_440_750.cub')
    pysis_ext.ratio(denominator=cubefile_ordered+'+1', numerator= cubefile_ordered+'+4',to=join(path,base)+'_750_440.cub')
    pysis_ext.ratio(denominator=cubefile_ordered+'+4', numerator= cubefile_ordered+'+6',to=join(path,base)+'_920_750.cub')
    pysis_ext.ratio(denominator=cubefile_ordered+'+4', numerator= cubefile_ordered+'+7',to=join(path,base)+'_960_750.cub')
    pysis_ext.ratio(denominator=cubefile_ordered+'+1', numerator= cubefile_ordered+'+7',to=join(path,base)+'_960_440.cub')
    pysis_ext.ratio(denominator=cubefile_ordered+'+4', numerator= cubefile_ordered+'+2',to=join(path,base)+'_550_750.cub')

    # CEC
    strs = [cubefile+'+{0}'.format(i) for i in [1,4,7]]
    cecfile = join(path,base+'_'+cecfile)
    pysis_ext.cubeit(strs, cecfile, log=log, tempdir=tempdir, **kwargs)

    # CCCA
    strs = [join(path, base)+'_'+x+'_750.cub' for x in ['440', '960']]
    strs.insert(1, cubefile+'+4')
    cccafile = join(path,base+'_'+cccafile)
    pysis_ext.cubeit(strs, cccafile, log=log, tempdir=tempdir, **kwargs)

    # CCCAI
    strs = [join(path, base)+'_'+x+'.cub' for x in ['750_440','960_750']]
    strs.insert(1, cubefile+'+4')
    cccaifile = cccafile.replace('.cub','I.cub')
    pysis_ext.cubeit(strs, cccaifile, log=log, tempdir=tempdir, **kwargs)

    # CCCR
    strs = [join(path,base)+'_'+x+'_750.cub' for x in ['440','550','960']]
    cccrfile = join(path,base+'_'+cccrfile)
    pysis_ext.cubeit(strs, cccrfile, log=log, tempdir=tempdir, **kwargs)


def mosrange(datadir, outfile, tempdir=tempdir, precision=-1, log=None, **kwargs):
    '''Generate ISIS map file'''

    datafiles = findfile(datadir, '.cub', recursive=True)
    pysis_ext.mosrange(datafiles, outfile, precision=precision, log=log, **kwargs)


def fc_cube(infiles, outfile, layer=1, tempdir=tempdir, log=None, **kwargs):
    '''Generate FC color cubes.'''

    if layer is not None:
        l = '+'+str(layer)
    else:
        l = ''
    inputs = [x+l for x in infiles]
    pysis_ext.cubeit(inputs, outfile, tempdir=tempdir, log=log, **kwargs)


def fit_grid(datafiles, model, parms=None, fixed=None, ilim=None, elim=None, alim=None, rlim=None, outdir=None):
    '''Fit photometric model to data grid

 datafiles : str or str list
   Data to be fitted
 model : class name
   Model to be used
 trim : bool, optional
   Trim data for (i, e, a, r) or not
 ilim, elim, alim, ioflim : see `Photometry.PhotometricData.trim`

 v1.0.0 : 1/20/2016, JYL @PSI
    '''
    from os.path import dirname, basename
    if isinstance(datafiles, (str,bytes)):
        datafiles = [datafiles]
    for ii in range(len(datafiles)):
        df = datafiles[ii]
        if outdir is None:
            outdir = dirname(df)
        print('Load photometric data {0}'.format(df))
        dg = Photometry.PhotometricDataGrid(datafile=df)
    #   if trim:
    #       print 'Trimming data to remove invalid data points'
    #       dg.trim(ilim=ilim,elim=elim,alim=alim,ioflim=ioflim)
    #       print 'Saving trimmed data'
    #       dg.write()
        print('Fitting model')
        if parms is None:
            m0 = model()
        else:
            m0 = model(*(parms[ii]))
        if fixed is not None:
            print('set fixed')
            for jj in range(len(fixed)):
                getattr(m0, m0.param_names[jj]).fixed=fixed[jj]
        f = dg.fit(m0, ilim=ilim, elim=elim, alim=alim,rlim=rlim)
        outfile = outdir+'/'+basename(df).replace('.fits','_'+str(f.model.model_class.name)+'.fits')
        f.model.write(outfile,overwrite=True)
        nlat, nlon = dg.shape
        info = {}
        info['rms'] = np.zeros((nlat, nlon))
        for k in m0.param_names:
            info[k+'_err'] = np.zeros((nlat,nlon))
        info['niter'] = np.zeros((nlat,nlon),dtype=int)
        info['ierr'] = np.zeros((nlat,nlon),dtype=int)
        info['dof'] = np.zeros((nlat,nlon),dtype=int)
        for i in range(nlat):
            for j in range(nlon):
                if ~f.model.mask[i,j]:
                    info['rms'][i,j] = f.RMS[i,j]
                    for k in 'niter ierr dof'.split():
                        info[k][i,j] = f.fit_info[i,j][k]
                    for k,n in zip(m0.param_names,list(range(len(m0.param_names)))):
                        info[k+'_err'][i,j] = f.fit_info[i,j]['serror'][n]
        hdus = fits.open(outfile)
        keys = 'rms niter ierr dof'.split()
        for k in m0.param_names:
            keys.append(k+'_err')
        for k in keys:
            hdu = fits.ImageHDU(info[k],name=k)
            hdus.append(hdu)
        hdus.writeto(outfile, clobber=True)


def par2cube(modelfiles, sort=False, logfile=None, overwrite=True):
    '''
 Assemble grid model photometric parameters of all FC bands to ISIS cubes.

 v1.0.0 : 1/20/2016, JYL @PSI
    '''
    import os
    from jylipy import pysis_ext

    if logfile is None:
        logfile = os.path.join(tempdir,'temp.log')

    outdir = os.path.dirname(modelfiles[0])
    nf = len(modelfiles)
    modelfiles = np.asarray(modelfiles)
    modelfiles.sort()
    if sort is False:
        sort = list(range(nf))
    else:
        sort = [6,0,5,1,4,2,3]
    m = fits.open(modelfiles[0])
    ny,nx = m[1].data.shape
    parnames = eval(m[0].header['parnames'])
    par = {}
    for k in parnames:
        par[k] = np.zeros((nf,ny,nx))
    m.close()
    for j,i in enumerate(sort):
        f = modelfiles[i]
        m = fits.open(f)
        for k in parnames:
            par[k][j] = m[k].data[::-1,:]
    for k in parnames:
        outfile = outdir+'/model_'+k+'.fits'
        cubfile = outdir+'/model_'+k+'.cub'
        writefits(outfile, par[k], clobber=overwrite)
        pysis_ext.fits2isis(outfile, cubfile)
