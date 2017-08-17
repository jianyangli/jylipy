# HST WFPC2 processing tools
#
# 2/10/2015, JYL @PSI


from ..core import ImageMeasurement, Time, rebin
from ..apext import Table
from .. import convenience
from collections import OrderedDict
import numpy as np


def _assemble_wfpc2(arr, rescale_pc=False):
    '''Assemble WFPC2 full array from a 4x800x800 array'''
    out = np.zeros((1600,1600))
    if rescale_pc:
        out[800:1200,800:1200] = rebin(arr[0],(2,2),mean=True)
    else:
        out[800:,800:] = arr[0]
    out[800:,:800] = np.rot90(arr[1],3)
    out[:800,:800] = np.rot90(arr[2],2)
    out[:800,800:] = np.rot90(arr[3],1)
    return out

def _disassemble_wfpc2(im):
    out = np.zeros((4,800,800))
    out[0] = np.array(im[800:,800:])
    out[1] = np.asarray(np.rot90(im[800:,:800],1))
    out[2] = np.asarray(np.rot90(im[:800,:800],2))
    out[3] = np.asarray(np.rot90(im[:800,800:],3))
    return out

pamfile = convenience.dir.work+'references/HST/WFPC2/Pixel_Area_Map/f1k1552bu_r9f.fits'
from astropy.io import fits
pam = _assemble_wfpc2(fits.open(pamfile)[0].data)


class WFPC2Image(ImageMeasurement):

    def __new__(cls, inputfile, dtype=None):
        from astropy.io import fits
        fitsfile = fits.open(inputfile)
        header = OrderedDict()
        if inputfile.endswith('_c0f.fits'):
            im = _assemble_wfpc2(fitsfile[0].data)
            obj = Image(im, dtype=dtype).view(WFPC2Image)
            obj.header = fitsfile[0].header
        elif inputfile.endswith('_drz.fits'):
            obj = Image(fitsfile[1].data, wht=fitsfile[2].data, ctx=fitsfile[3].data,dtype=dtype).view(WFPC2Image)
            obj.header = OrderedDict()
            obj.header['primary'] = fitsfile[0].header
            obj.header['sci'] = fitsfile[1].header
            obj.header['wht'] = fitsfile[2].header
            obj.header['ctx'] = fitsfile[3].header
        obj.source = inputfile
        obj.geometry = {}
        obj.calibration = {}
        return obj

    def __finalize_array__(self, obj):
        super(WFPC2Image, self).__array_finalize__(obj)
        if obj is None: return
        from copy import copy
        self.source = getattr(obj, 'source', None)
        self.header = copy(getattr(obj, 'header', None))
        self.geometry = copy(getattr(obj, 'geometry', {}))
        self.calibration = copy(getattr(obj, 'calibration', {}))

    @property
    def pc(self):
        return self[800:,800:]
    @pc.setter
    def pc(self, value):
        self[800:,800:] = value

    @property
    def wf2(self):
        return self[800:,:800]
    @wf2.setter
    def wf2(self, value):
        self[800:,:800] = value

    @property
    def wf3(self):
        return self[:800,:800]
    @wf3.setter
    def wf3(self, value):
        self[:800,:800] = value

    @property
    def wf4(self):
        return self[:800,800:]
    @wf4.setter
    def wf4(self, value):
        self[:800,800:] = value


def aspect(files, keys=None, verbose=False):
    '''Extract image aspect data'''

    ks = 'rootname expstart expend filtnam1 filtnam2 exptime orientat'.split()
    nk = len(ks)
    if keys is not None:
        if (not isinstance(keys, (str,bytes))) and hasattr(keys,'__iter__'):
            ks = ks+keys
        else:
            ks = ks+[keys]
    vs = [[] for x in ks]

    if isinstance(files, (str,bytes)):
        files = [files]

    if verbose:
        print('{0} files to be processed'.format(len(files)))
        print('Keys to be processed: ', ks)

    for f in files:
        if verbose:
            print(f)
        im = WFPC2Image(f)
        for k, v in zip(ks, vs):
            v.append(im.header[k])
    st = vs.pop(ks.index('expstart'))
    ks.pop(ks.index('expstart'))
    ed = vs.pop(ks.index('expend'))
    ks.pop(ks.index('expend'))
    utc = Time((np.array(st)+ed)/2, format='mjd').isot
    vs.insert(1,utc)
    ks.insert(1,'utc')

    if verbose:
        print('Done.')

    return Table(vs, names=ks)


class WFPC2Calibration(object):

    photflam = {}
    photflam['F606W'] = [1.900e-17, 1.842e-17, 1.888e-17, 1.914e-17]
    pxlscl = [0.0455, 0.0996, 0.0996, 0.0996]
    fsun = {}
    fsun['F606W'] = 1722.61

    def calibrate(self,im):
        flt = (im.header['filtnam1']+im.header['filtnam2']).strip()
        texp = im.header['exptime']
        im /= texp
        im.pc = im.pc*self.photflam[flt][0]
        im.wf2 = im.wf2*self.photflam[flt][1]
        im.wf3 = im.wf3*self.photflam[flt][2]
        im.wf4 = im.wf4*self.photflam[flt][3]

