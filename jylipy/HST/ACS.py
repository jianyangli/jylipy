# HST ACS processing tools
#
# 2/5/2015, JYL @PSI


from ..core import ImageMeasurement, Time
from ..apext import Table
from .. import convenience
from collections import OrderedDict
import numpy as np

pamfile = convenience.dir.work+'references/HST/ACS/HRC/hrc_pam.fits'

class HRCImage(ImageMeasurement):

    def __new__(cls, inputfile, dtype=None):
        from astropy.io import fits
        fitsfile = fits.open(inputfile)
        header = OrderedDict()
        if inputfile.endswith('_flt.fits'):
            obj = Image(fitsfile[1].data, error=fitsfile[2].data, dq=fitsfile[3].data, dtype=dtype).view(HRCImage)
            for k in 'primary sci err dq'.split():
                header[k] = fitsfile[k].header
        elif inputfile.endswith('_drz.fits'):
            obj = Image(fitsfile[1].data, wht=fitsfile[2].data, ctx=fitsfile[3].data, dtype=dtype).view(HRCImage)
            for k in 'primary sci wht ctx'.split():
                header[k] = fitsfile[k].header
        obj.header = header
        obj.source = inputfile
        obj.geometry = {}
        obj.calibration = {}
        return obj

    def __finalize_array__(self, obj):
        super(HRCImage, self).__array_finalize__(obj)
        if obj is None: return
        from copy import copy
        self.source = getattr(obj, 'source', None)
        self.header = copy(getattr(obj, 'header', None))
        self.geometry = copy(getattr(obj, 'geometry', {}))
        self.calibration = copy(getattr(obj, 'calibration', {}))


def aspect(files, keys=None, verbose=False):
    '''Extract image aspect data'''

    ks = 'rootname expstart expend filter1 filter2 exptime aperture orientat'.split()
    nk = len(ks)
    if keys is not None:
        if hasattr(keys,'__iter__'):
            ks = ks+keys
        else:
            ks = ks+[keys]
    vs = [[] for x in ks]

    if not hasattr(files,'__iter__'):
        files = [files]

    if verbose:
        print('{0} files to be processed'.format(len(files)))
        print('Keys to be processed: ', ks)

    for f in files:
        if verbose:
            print(f)
        im = HRCImage(f)
        for k, v in zip(ks, vs):
            v.append(im.header['primary'][k])
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
