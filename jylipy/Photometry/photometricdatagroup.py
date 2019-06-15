# Photometric data group classes

import os
import numpy as np
from astropy.io import fits
from .core import PhotometricData

__all__ = ['phoarr_info', 'PhotometricDataArray', 'PhotometricModelArray']


_memory_size = lambda x: x*4.470348358154297e-08


def phoarr_info(phoarr, shape=False, all=False):
    """Return the information of photometric data group class instances


    phoarr : str or `PhotometricDataArray`
        The input `PhotometricDataArray` or the name of the file that contains
        the `PhotometricDataArray`.
    shape : bool
        If `True` then return the shape of `PhotometricDataArray`.
        If `False` then return the information table
    all : bool
        If `True`, then return everyting in a tuple

    Return
    info : `numpy.ndarray`
        A record array containing the information table of
        `PhotometricDataArray`
    shape : tuple
        The shape of `PhotometricDataArray`
    """
    if isinstance(phoarr, str):
        infile, indir = PhotometricDataArray._path_name(phoarr)
        inf = fits.open(infile)
        if inf[0].header['version'] != '1.0.0':
            raise IOError('Unrecognized version number.')
        ndim = inf[0].header['ndim']
        sp = tuple([inf[0].header['dim{}'.format(i+1)] \
            for i in range(ndim)])
        if shape:
            return sp
        elif all:
            return inf['info'].data, sp
        else:
            return inf['info'].data
    elif isinstance(phoarr, PhotometricDataArray):
        if shape:
            return phoarr.shape
        elif all:
            return phoarr.reshape(-1)[list(phoarr.dtype.names[1:-2])], \
                phoarr.shape
        else:
            return phoarr.reshape(-1)[list(phoarr.dtype.names[1:-2])]


class PhotometricDataArray(np.ndarray):
    """Photometric data array object
    """
    def __new__(cls, *args, **kwargs):
        maxmem = kwargs.pop('maxmem', 10)
        datafile = kwargs.pop('datafile', None)
        n = int((np.floor(np.log10(np.array(args[0]).prod()))+1))
        fmt = '%0' + '%i' % n + 'i'
        dtype = kwargs.pop('dtype', None)
        names = tuple('pho file count incmin incmax emimin emimax'
            ' phamin phamax lonmin lonmax latmin latmax masked loaded'
            ' flushed'.split())
        types = [object, 'U{}'.format(n+13), int, float, float, float, float,
            float, float, float, float, float, float, bool, bool, bool]
        dtype = [(n, d) for n, d in zip(names, types)]
        obj = super().__new__(cls, *args, dtype=dtype, **kwargs)
        obj.maxmem = maxmem
        obj.datafile = datafile
        i = 0
        for x in np.nditer(obj, flags=['refs_ok'], op_flags=['readwrite']):
            x['file'] = 'phoarr_' + fmt % i + '.fits'
            x['masked'] = True
            x['flushed'] = True
            i += 1
        obj.__version__ = '1.0.0'
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.maxmem = getattr(obj, 'maxmem', None)
        self.datafile = getattr(obj, 'datafile', None)
        self.__version__ = getattr(obj, '__version__', '1.0.0')

    def __repr__(self):
        """Return repr(self)."""
        return f'<{self.__class__.__name__} {self.shape}>'

    def __str__(self):
        """Return str(self)"""
        return self.__repr__()

    def __getitem__(self, k):
        out = super().__getitem__(k)
        if isinstance(out, PhotometricDataArray):
            if (out.dtype.names is None) or (len(out.dtype.names) < 16):
                return np.asarray(out)
            return out
        else:
            if (not out['masked']) and (not out['loaded']):
                if self.datafile is None:
                    raise ValueError('Data file not specified.')
                datafile, datadir = self._path_name(self.datafile)
                f = os.path.join(datadir, out['file'])
                if os.path.isfile(f):
                    self._memory_dump()
                    out['pho'] = PhotometricData(f)
                    out['loaded'] = True
                    out['flushed'] = True
                else:
                    raise IOError('Data record not found from file {}'.
                            format(f))
            return out['pho']

    def __setitem__(self, k, v):
        if isinstance(k, str) or ((hasattr(k, '__iter__')) and \
                np.any([isinstance(x, str) for x in k])):
            raise ValueError('Setting property fields not allowed')
        if (self[k] is None) or (isinstance(self[k], PhotometricData)):
            if not isinstance(v, PhotometricData):
                raise ValueError('`PhotometricData` instance required.')
            self._memory_dump()
            self['pho'][k] = v
            self['count'][k] = len(v)
            self['incmin'][k] = v.inclim[0].to('deg').value
            self['incmax'][k] = v.inclim[1].to('deg').value
            self['emimin'][k] = v.emilim[0].to('deg').value
            self['emimax'][k] = v.emilim[1].to('deg').value
            self['phamin'][k] = v.phalim[0].to('deg').value
            self['phamax'][k] = v.phalim[1].to('deg').value
            self['latmin'][k] = v.latlim[0].to('deg').value
            self['latmax'][k] = v.latlim[1].to('deg').value
            self['lonmin'][k] = v.lonlim[0].to('deg').value
            self['lonmax'][k] = v.lonlim[1].to('deg').value
            self['masked'][k] = False
            self['flushed'][k] = False
            self['loaded'][k] = True
        elif isinstance(self[k], PhotometricDataArray):
            if isinstance(v, PhotometricData):
                self._memory_dump()
                for x in np.nditer(self[k], flags=['refs_ok'],
                        op_flags=['readwrite']):
                    recname = str(x['file'].copy())
                    x[...] = (v.copy(),
                              recname,
                              len(v),
                              v.inclim[0].to('deg').value,
                              v.inclim[1].to('deg').value,
                              v.emilim[0].to('deg').value,
                              v.emilim[1].to('deg').value,
                              v.phalim[0].to('deg').value,
                              v.phalim[1].to('deg').value,
                              v.latlim[0].to('deg').value,
                              v.latlim[1].to('deg').value,
                              v.lonlim[0].to('deg').value,
                              v.lonlim[1].to('deg').value,
                              False,
                              True,
                              False)
            elif isinstance(v, PhotometricDataArray):
                fields = list(self.dtype.names)
                fields.remove('file')
                self._memory_dump()
                for f in fields:
                    from copy import deepcopy
                    self[f][k] = deepcopy(v[f])
                for x in np.nditer(self[k], flags=['refs_ok'],
                        op_flags=['readwrite']):
                    if not x['masked']:
                        x['flushed'] = False
            else:
                raise ValueError('Only `PhotometricData` or'
                    ' `PhotometricDataArray` instance allowed.')
        else:
            raise ValueError('Only `PhotometricData` or `PhotometricDataArray`'
                ' instance allowed.')
        self._memory_dump()

    def _set_property(self, k, v):
        """Set the property fields of the record array
        """
        super().__setitem__(k, v)

    def _memory_dump(self, forced=False, verbose=False):
        """Check the size of object, free memory by deleting
        """
        if not forced:
            total_counts = (self['count']*self['loaded'].astype('i')).sum()
            sz = _memory_size(total_counts*1.2)
            if sz<self.maxmem:
                return False
        # free memory by deleting all PhotometricData instances
        if verbose:
            print('Cleaning memory...')
        for x in np.nditer(self, flags=['refs_ok'], op_flags=['readwrite']):
            self._save_data(x)
            x['pho'] = None
            x['loaded'] = False
        return True

    @staticmethod
    def _path_name(filename):
        """Return the information file name and data file directory name
        """
        filename = os.path.splitext(filename)[0]
        datadir = filename+'_dir'
        infofile = filename+'.fits'
        return infofile, datadir

    #def _load_data(self, x):
    #    """Load photometric data record

    #     x : one element in `PhotometricDataArray`
    #    """
    #    if (not x['masked']) and (not x['loaded']):
    #        if self.datafile is None:
    #            raise ValueError('Data file not specified.')
    #        filename, datadir = self._path_name(self.datafile)
    #        f = os.path.join(datadir, x['file'].item())
    #        if os.path.isfile(f):
    #            x['pho'][()] = PhotometricData(f)
    #            x['loaded'] = True
    #            x['flushed'] = True
    #        else:
    #            raise IOError('Data record not found.')

    def _save_data(self, x, outdir=None, flush=True):
        """Save photometric data record

        x : one element in `PhotometricDataArray`
        outdir : str
            The directory name of data record files
        flush : bool
            The value that the 'flushed' flag will be set
        """
        if self.datafile is None:
            raise ValueError('Data file not specified.')
        if outdir is None:
            outfile, outdir = self._path_name(self.datafile)
        f = os.path.join(outdir, x['file'].item())
        if (x['masked']) and os.path.isfile(f):
            os.remove(f)
        elif x['loaded'] and ((not x['flushed']) or not flush):
            x['pho'].item().write(f, overwrite=True)
            x['flushed'] = flush

    def write(self, outfile=None, overwrite=False):
        """Write data to disk.

        outfile : str, optional
            Output file name.  If `None`, then the program does a memory
            flush.  If `self.datafile` is also `None`, then an exception will
            be thrown.
            If a file name is provided via `outfile`, then this method saves
            all data to the specified output file.  In this case, if
            `self.datafile` is `None`, then the provided file name will be
            assigned to `self.datafile` and becomes the default data file of
            the class instance.  Otherwise `self.datafile` will remain
            unchanged, and the location specified by `outfile` is just an
            extra copy of the data.
        overwrite : bool, optional
            Overwrite the existing data.  If `outfile` is `None` (meaning a
            memory flush), then `overwrite` will be ignored.
        """
        if outfile is None:
            if self.datafile is None:
                raise ValueError('output file not specified')
            outfile = self.datafile
            flush = True
        else:
            if self.datafile is None:
                self.datafile = outfile
                flush = True
            else:
                flush = False

        outfile, outdir = self._path_name(outfile)
        ### This segment is to be checked
        if (not flush) and (not overwrite) and os.path.isfile(outfile):
            raise IOError('output file {0} already exists'.format(outfile))
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # save envolope information
        hdu_list = fits.HDUList()
        hdr = fits.Header()
        hdr['version'] = self.__version__
        shape = self.shape
        hdr['ndim'] = len(shape), 'Number of dimensions'
        for i in range(hdr['ndim']):
            hdr['dim{}'.format(i+1)] = shape[i], \
                'Size in dimension {}'.format(i+1)
        hdu = fits.PrimaryHDU(header=hdr)
        hdu_list.append(hdu)
        hdr = fits.Header()
        hdr['extname'] = 'INFO'
        info_table = self.info()
        hdu = fits.BinTableHDU(info_table, header=hdr)
        hdu_list.append(hdu)
        hdu_list.writeto(outfile, overwrite=True)

        # save data records
        for x in np.nditer(self, flags=['refs_ok'], op_flags=['readwrite']):
            self._save_data(x, outdir=outdir, flush=flush)

    def info(self):
        """Print out the data information
        """
        return phoarr_info(self)

    def read(self, infile=None, verbose=False, load=False):
        '''Read data from a directory or a list of files

        infile : str
          The summary file of data storage

        v1.0.0 : 1/12/2016, JYL @PSI
        '''
        if infile is None:
            if self.datafile is None:
                raise ValueError('Input file not specified.')
            else:
                infile = self.datafile
        if self.datafile is None:
            self.datafile = infile
        info = phoarr_info(infile)
        infile, indir = self._path_name(infile)

        if not os.path.isfile(infile):
            raise ValueError('Input file {0} not found.'.format(infile))
        if not os.path.isdir(indir):
            raise ValueError('Input directory {0} not found.'.format(indir))

        # set up data structure
        if phoarr_info(infile, shape=True) != self.shape:
            raise IOError("Shape of PhotometricDataArray doesn't match input"
                " data.")
        for k in info.dtype.names:
            self._set_property(k, info[k].reshape(self.shape))

    @classmethod
    def from_file(cls, infile):
        shape = phoarr_info(infile, shape=True)
        obj = cls(shape)
        obj.read(infile)
        return obj


class PhotometricModelArray(np.ndarray):
    """Photometric model array

    This class is to support the modeling of `PhotometricDataArray` class.  It
    contains an array of the same photometric model.
    """

    def __new__(cls, shape, model):
        """
        shape : array-like int
            Shape of array
        model : class name
            The model class
        """
        names = ('masked',)
        names = names + model.param_names
        types = [bool] + [object] * len(model.param_names)
        dtype = [(n, d) for n, d in zip(names, types)]
        obj = super().__new__(cls, shape, dtype=dtype)
        for x in np.nditer(obj, flags=['refs_ok'], op_flags=['readwrite']):
            x['masked'] = True
        obj.__version__ = '1.0.0'
        obj._model_class = model
        obj._param_names = model.param_names
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.__version__ = getattr(obj, '__version__', '1.0.0')
        self._model_class = getattr(obj, 'model_class', None)
        self._param_names = getattr(obj, 'param_names', None)

    def __getattr__(self, name):
        if name not in self.param_names:
            try:
                prop = super().__getattr__(name)
            except AttributeError:
                raise AttributeError("'{}' object has no attribute '{}'".
                    format(self.__class__.__name__, name))
        else:
            return self.get_param(name)

    @property
    def param_names(self):
        """Parameter names"""
        return self._param_names

    @property
    def model_class(self):
        """Model class"""
        return self._model_class

    @property
    def mask(self):
        """Model grid mask, where invalide models are masked (True)"""
        return self['masked']

    def __repr__(self):
        """Return repr(self)."""
        return f'<{self.__class__.__name__} {self.shape}>'

    def __str__(self):
        """Return str(self)"""
        return self.__repr__()

    def __getitem__(self, k):
        out = super().__getitem__(k)
        if isinstance(out, PhotometricModelArray):
            if (out.dtype.names is None) or (len(out.dtype.names) < len(self.param_names)+1):
                return np.asarray(out)
            else:
                return out
        if not out['masked']:
            parms = [out[k] for k in self.param_names]
            if hasattr(parms[0], '__iter__'):
                n_models = len(parms[0])
            else:
                n_models = 1
            return self.model_class(*parms, n_models=n_models)
        else:
            return None

    def __setitem__(self, k, v):
        if isinstance(k, str) or ((hasattr(k, '__iter__')) and \
                np.any([isinstance(x, str) for x in k])):
            raise ValueError('Setting property fields not allowed')
        if (self[k] is None) or (isinstance(self[k], self.model_class)):
            if not isinstance(v, self.model_class):
                raise ValueError('`{}` instance required.'.
                    format(self.model_class.__class__.__name__))
            self['masked'][k] = False
            for p in self.param_names:
                self[p][k] = getattr(v, p).value
        elif isinstance(self[k], PhotometricModelArray):
            if isinstance(v, self.model_class):
                for x in np.nditer(self[k], flags=['refs_ok'],
                        op_flags=['readwrite']):
                    pars = tuple([getattr(v, p).value for p in
                        self.param_names])
                    x[...] = (False,) + pars
            elif isinstance(v, PhotometricModelArray):
                for f in list(self.dtype.names):
                    from copy import deepcopy
                    self[f][k] = deepcopy(v[f])
            else:
                raise ValueError('Only `{}` or `PhotometricModelArray`'
                    ' instance allowed.'.format(self.model_class.__class__.
                    __name__))
        else:
            raise ValueError('Only `{}` or `PhotometricModelArray`'
                    ' instance allowed.'.format(self.model_class.__class__.
                    __name__))

    def get_param(self, p):
        """Return parameter 'p'
        """
        if p not in self.param_names:
            raise ValueError("'{}' is not a parameter of model '{}'".
                format(p, self.model_class.__name__))
        len_arr = np.ma.zeros(self.shape, dtype=int)
        len_arr.mask = self.mask
        it = np.nditer(self, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            if not len_arr.mask[it.multi_index]:
                par = self[self.param_names[0]][it.multi_index]
                if hasattr(par, '__iter__'):
                    len_arr[it.multi_index] = np.size(par)
                else:
                    len_arr[it.multi_index] = 1
            it.iternext()
        len_max = len_arr.max()
        if len_max == 1:
            # single model
            return np.ma.array(self[p], mask=self.mask, \
                dtype=float)
        elif (len_arr == len_max).all():
            # homogeneous number of models
            par = self[p]
            non_masked = np.where(~self.mask.flatten())[0].min()
            shape = np.shape(par.flatten()[non_masked])
            it = np.nditer(par, flags=['multi_index', 'refs_ok'])
            while not it.finished:
                if self.mask[it.multi_index]:
                    par[it.multi_index] = np.zeros(shape)
                it.iternext()
            par = np.vstack(par.reshape(-1))
            par = par.reshape(self.shape+shape).astype(float)
            mask = np.broadcast_to(self.mask, shape+self.shape)
            ndim = len(mask.shape)
            for i in range(ndim - len(self.shape)):
                mask = np.rollaxis(mask, 0, ndim)
            return np.ma.array(par, mask=mask)
        else:
            # variable number of models
            return np.ma.array(self[p], mask=self.mask)
