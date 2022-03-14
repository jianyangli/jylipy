'''PDS image data classes
v0.0.1 : Jan 2015, JYL@PSI
'''

import numpy as np, numbers
from .apext import units
from .core import Image, condition, num, CaseInsensitiveOrderedDict
from io import IOBase
import pvl

__all__ = ['VersionError', 'Header', 'pds_types', 'PDSData', 'readpds']


class VersionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Header(pvl.collections.PVLModule):
    """PDS label class"""

    @classmethod
    def load(cls, file):
        obj = cls(pvl.load(file))
        pointers = [s for s in obj.keys() \
                if ('^' in s) and ('HISTORY' not in s)]
        pointers = [s.replace('^', '') for s in pointers]
        obj.pointers = {}
        for k in pointers:
            obj.pointers[k] = obj['^'+k]
        return obj


pds_types = {'LSB_UNSIGNED_INTEGER': 'uint',
             'MSB_UNSIGNED_INTEGER': 'uint',
             'LSB_INTEGER': 'int',
             'MSB_INTEGER': 'int',
             'PC_REAL': 'float'} #, \
             # 'IEEE_REAL': 'float'}


class PDSData():

    def __init__(self, datafile=None):
        if datafile is not None:
            self.header = self._parse_label(datafile)
            if 'PDS_VERSION_ID' not in self.header.keys() or \
                    self.header['PDS_VERSION_ID'] != 'PDS3':
                raise VersionError('Unrecognized PDS version.')
            recds = self._parse_data(datafile)

    def _parse_label(self, datafile):
        return Header.load(datafile)

    def _parse_data(self, datafile):
        self.records = []
        with open(datafile, 'rb') as f:
            s = f.read()
        ims = {}
        for k, v in self.header.pointers.items():
            if isinstance(v, int):
                pt = (v-1) * self.header['RECORD_BYTES']
            else:
                from os.path import dirname, join
                if isinstance(v, str):
                    imgfile = k
                    start = 1
                elif hasattr(v, '__iter__'):
                    imgfile, pt = v
                    pt = int(pt)
                with open(join(dirname(datafile), imgfile), 'rb') as f:
                    s = f.read()
            if 'IMAGE' in k:
                self.__dict__[k] = self._read_image_rec(k, s[pt:])
                self.records.append(k)

    def __array__(self):
        return np.asarray(getattr(self, self.records[0], None))

    def _read_image_rec(self, obj, st):
        '''Read a PDS image data record'''
        import warnings
        objhdr = self.header[obj]
        dtype = pds_types[objhdr['SAMPLE_TYPE']] + str(objhdr['SAMPLE_BITS'])
        if 'BANDS' in objhdr.keys() and objhdr['BANDS'] > 1:
            if objhdr['BAND_STORAGE_TYPE'] == 'BAND_SEQUENTIAL':
                shape = objhdr['BANDS'], objhdr['LINES'], objhdr['LINE_SAMPLES']
            else:
                shape = objhdr['LINES'], objhdr['LINE_SAMPLES'], objhdr['BANDS']
        else:
            shape = objhdr['LINES'], objhdr['LINE_SAMPLES']
        count = np.array(shape).prod()
        out = np.frombuffer(st, dtype=dtype, count=count)
        if 'SCALING_FACTOR' in objhdr.keys():
            out = out * objhdr['SCALING_FACTOR']
        if ('unit' in objhdr.keys()):
            unitstr = objhdr['unit']
        elif ('UNIT' in objhdr.keys()):
            unitstr = objhdr['UNIT']
        else:
            unitstr = ''
        if unitstr.lower() == 'du':
            unit = units.adu
        else:
            unit = units.Unit(unitstr, parse_strict='warn')
        if 'BANDS' in objhdr.keys() and objhdr['BANDS'] == 1:
            im = Image(np.squeeze(out.reshape(shape)), meta=objhdr, unit=unit)
        else:
            im = np.squeeze(out.reshape(shape))
        if 'LINE_DISPLAY_DIRECTION' not in objhdr:
            warnings.warn('LINE_DISPLAY_DIRECTION not specified.')
        else:
            if objhdr['LINE_DISPLAY_DIRECTION'] not in ['UP', 'DOWN']:
                warnings.warn('LINE_DISPLAY_DIRECTION invalid')
            elif objhdr['LINE_DISPLAY_DIRECTION'] == 'DOWN':
                im = im[::-1, :]
        if 'SAMPLE_DISPLAY_DIRECTION' not in objhdr:
            warnings.warn('SAMPLE_DISPLAY_DIRECTION not specified.')
        else:
            if objhdr['SAMPLE_DISPLAY_DIRECTION'] not in ['LEFT', 'RIGHT']:
                warnings.warn('SAMPLE_DISPLAY_DIRECTION invalid')
            elif objhdr['SAMPLE_DISPLAY_DIRECTION'] == 'LEFT':
                im = im[:, ::-1]
        return im


def readpds(datafile):
    return PDSData(datafile)
