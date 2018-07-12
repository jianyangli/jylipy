'''PDS image data classes
v0.0.1 : Jan 2015, JYL@PSI
'''

import numpy as np, numbers
from .apext import units
from .core import Image, condition, num, CaseInsensitiveOrderedDict
from io import IOBase

__all__ = ['Object' , 'Header', 'pds_types', 'PDSData', 'readpds']


class VersionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Object(CaseInsensitiveOrderedDict):
    '''PDS Object class'''

    def __init__(self, obj=None):
        if isinstance(obj, IOBase):
            super(Object, self).__init__()
            self.readfile(obj)
        else:
            super(Header, self).__init__(obj)

    def readfile(self, f):
        if not isinstance(f, IOBase):
            f = f.open(f, 'rb')
        inobj = False
        while True:
            s = f.readline().decode()
            if not inobj and not s.startswith('OBJECT'):
                continue
            if '=' in s:
                k, v = s.split('=')
                k, v = k.strip(), v.strip()
                if k == 'OBJECT':
                    self['NAME'] = v
                    inobj = True
                elif k == 'END_OBJECT':
                    break
                else:
                    self[k] = v
            else:
                self[k] = ' '.join([self[k], s.strip()])
        Header._check_values(self)


class Header(CaseInsensitiveOrderedDict):
    '''PDS label class'''

    def __init__(self, data=None):
        if isinstance(data, str):
            super(Header, self).__init__()
            self.readfile(data)
        else:
            super(Header, self).__init__(data)

    def readfile(self, hdrfile):
        '''Read PDS headers from a file'''
        if not (hdrfile.lower().endswith('img') or hdrfile.lower().endswith('lbl') or hdrfile.lower().endswith('pds')):
            raise ValueError('input file does not have a PDS extension (IMG or LBL)')
        for k in list(self.keys()):
            self.pop(k)
        if isinstance(hdrfile, IOBase):
            f = hdrfile
        else:
            f = open(hdrfile,'rb')
        # Check PDS signature and version
        s = f.readline().split(b'=')
        if s[0].strip() != b'PDS_VERSION_ID':
            raise VersionError('No valid PDS label found.')
        if s[1].strip() != b'PDS3':
            raise VersionError('No PDS3 label found.')
        self[s[0].strip().decode()] = s[1].strip().decode()
        # Processing keys
        while True:
            s = f.readline().decode()#.strip('\r\n')
            if s.startswith('/*') or s.strip() == '':
                continue
            if '=' in s:
                k, v = s.split('=')
                key, v = k.strip(), v.strip()
                if key == 'OBJECT':
                    f.seek(-len(s),1)
                    self[v] = Object(f)
                elif '^' in key:
                    if not hasattr(self, 'pointers'):
                        self.pointers = CaseInsensitiveOrderedDict()
                    self.pointers[key.strip('^')] = v
                else:
                    self[key] = v
            elif s.strip() == 'END':
                break
            else:
                self[key] = ' '.join([self[key],s.strip()])
        f.close()
        self._check_values(self)
        self._check_values(self.pointers)

    @staticmethod
    def _check_values(self):
        for k in list(self.keys()):
            if isinstance(self[k], Object):
                continue
            if self[k].startswith('('):
                s = self[k].strip('()').strip()
                if s == '':
                    self[k] = None
                else:
                    s = s.split(',')
                    v = []
                    for i in s:
                        if '"' in i:
                            i = i.strip('" ')
                            v.append(condition(i=='N/A',None,i))
                        elif '<' in i:
                            vv, u = i.strip().split('<')
                            u = u.strip('<>')
                            if 'per' in u:
                                u = u.replace('per', '/')
                            u = u.lower()
                            v.append(float(vv)*units.Unit(u))
                        else:
                            v.append(num(i))
                    if len(v) > 0:
                        if isinstance(v[0], units.Quantity):
                            vv, vu = [v[0].value], v[0].unit
                            for i in v[1:]:
                                if isinstance(i, units.Quantity):
                                    vv.append(i.to(vu).value)
                                else:
                                    vv.append(np.nan)
                            self[k] = np.array(vv)*vu
                        else:
                            self[k] = np.array(v)
                    else:
                        self[k] = v
            elif '"' in self[k]:
                self[k] = self[k].strip('" ')
            elif '<' in self[k]:
                v, u = self[k].strip().split('<')
                v = num(v)
                u = u.strip('<>')
                if u == 'kelvin':
                    u = 'Kelvin'
                else:
                    u = u.lower()
                if u == 'degrees':
                    u = 'degree'
                elif u in ['w','v']:
                    u = u.upper()
                if isinstance(v, numbers.Number):
                    self[k] = v*units.Unit(u)
            else:
                self[k] = num(self[k])
            if k == 'SPICE_FILE_NAME':
                self[k] = [x.replace('\\', '/') for x in self[k]]


pds_types = {'LSB_UNSIGNED_INTEGER': 'uint', \
             'MSB_UNSIGNED_INTEGER': 'uint', \
             'PC_REAL': 'float'}

class PDSData(object):

    def __init__(self, datafile=None):
        if datafile is not None:
            self.__dict__['header'] = Header(datafile)
            recds = self.readpds(datafile)
            self.records = []
            for k in recds:
                self.__dict__[k.lower()] = recds[k]
                self.records.append(k.lower())

    def __array__(self):
        return np.asarray(getattr(self, self.records[0], None))

    @staticmethod
    def read_image_rec(obj, st):
        '''Read a PDS image data record'''
        import warnings
        dtype = pds_types[obj['SAMPLE_TYPE']]+str(obj['SAMPLE_BITS'])
        if obj['BANDS'] > 1:
            if obj['BAND_STORAGE_TYPE'] == 'BAND_SEQUENTIAL':
                shape = obj['BANDS'],obj['LINES'],obj['LINE_SAMPLES']
            else:
                shape = obj['LINES'],obj['LINE_SAMPLES'],obj['BANDS']
            count = obj['LINES']*obj['LINE_SAMPLES']*obj['BANDS']
        else:
            shape = obj['LINES'],obj['LINE_SAMPLES']
            count = obj['LINES']*obj['LINE_SAMPLES']
        out = np.fromstring(st, dtype=dtype, count=count)
        #if ('unit' in list(obj.keys())) or ('UNIT' in list(obj.keys())):
        #    if obj['unit'].lower() == 'du':
        #        unit = units.adu
        #    else:
        #        unit = units.Unit(obj['unit'])
        #else:
        #    unit = ''
        unit = ''
        if obj['BANDS'] == 1:
            im = Image(np.squeeze(out.reshape(shape)),meta=obj,unit=unit)
        else:
            im = np.squeeze(out.reshape(shape))
        if ('LINE_DISPLAY_DIRECTION' not in obj) or ('SAMPLE_DISPLAY_DIRECTION' not in obj):
            warnings.warn('DISPLAY_DIRECTION not specified')
        else:
            if obj['LINE_DISPLAY_DIRECTION'] not in ['UP', 'DOWN']:
                warnings.warn('LINE_DISPLAY_DIRECTION invalid')
            if obj['SAMPLE_DISPLAY_DIRECTION'] not in ['LEFT', 'RIGHT']:
                warnings.warn('SAMPLE_DISPLAY_DIRECTION invalid')
            if obj['LINE_DISPLAY_DIRECTION'] == 'DOWN':
                im = im[::-1,:]
            if obj['SAMPLE_DISPLAY_DIRECTION'] == 'LEFT':
                im = im[:,::-1]
        return im

    def readpds(self, infile):
        hdr = Header(infile)
        f = open(infile, 'rb')
        s = f.read()
        f.close()
        from collections import OrderedDict
        ims = OrderedDict()
        for objname in hdr.pointers:
            if objname in hdr:
                if isinstance(hdr.pointers[objname], int):
                    pt = (hdr.pointers[objname]-1)*hdr['RECORD_BYTES']
                    ims[objname] = self.read_image_rec(hdr[objname], s[pt:])
                elif isinstance(hdr.pointers[objname], str):
                    from os.path import dirname
                    f = open('/'.join([dirname(infile),hdr.pointers[objname]]), 'rb')
                    s = f.read()
                    f.close()
                    ims[objname] = self.read_image_rec(hdr[objname], s)
        return ims

def readpds(datafile):
    return PDSData(datafile)
