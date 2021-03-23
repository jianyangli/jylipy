# Photometric data group classes

import warnings
import os
import numpy as np
from astropy.io import fits
from .core import PhotometricData, PhotometricMPFitter

__all__ = ['phoarr_info', 'PhotometricDataArray', 'ModelArray',
           'PhotometricDataArrayFitter', 'PhotometricDataArrayMPFitter']


_memory_size = lambda x: x*4.470348358154297e-08


class MemoryWarning(Warning):
    "Memory dump failure warning."


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
    def __new__(cls, shape, maxmem=10, datafile=None, order=None):
        """Initialize object"""
        n = int((np.floor(np.log10(np.array(shape).prod()))+1))
        fmt = '%0' + '%i' % n + 'i'
        names = tuple('pho file count incmin incmax emimin emimax'
            ' phamin phamax lonmin lonmax latmin latmax masked loaded'
            ' flushed'.split())
        types = [object, 'U{}'.format(n+13), int, float, float, float, float,
            float, float, float, float, float, float, bool, bool, bool]
        dtype = [(n, d) for n, d in zip(names, types)]
        obj = super().__new__(cls, shape, dtype=dtype, order=order)
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
            # the type of `out` is casted to type(self) if k is a slice or
            # a field name
            if (out.dtype.names is None) or (len(out.dtype.names) < 16):
                # if getting a field not the whole record array
                return np.asarray(out)
            else:
                # if getting a slice of object
                return out
        else:
            # if getting a single element
            if (not out['masked']) and (not out['loaded']):
                self._load_data(out)
            return out['pho']

    def __setitem__(self, k, v):
        """Assign values to specified PhotometricDataArray elements

        Cases allowed for assignment:
            1. assign a single element by a PhotometricData object
            2. assign a slice by a PhotometricDataArray object of the
               same shape
            3. assign a slice by repeating a PhotometricData object
        """
        if isinstance(k, str) or ((hasattr(k, '__iter__')) and \
                np.any([isinstance(x, str) for x in k])):
            raise ValueError('Setting property fields not allowed')
        self._memory_dump()
        if (self[k] is None) or (isinstance(self[k], PhotometricData)):
            # assign a single element by a PhotometricData object
            if not isinstance(v, PhotometricData):
                raise ValueError('`PhotometricData` instance required.')
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
            # assigne a slice
            if isinstance(v, PhotometricData):
                # by repeating a single PhotometricData object
                prop = (len(v),
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
                for x in np.nditer(self[k], flags=['refs_ok'],
                        op_flags=['readwrite']):
                    recname = str(x['file'].copy())
                    x[...] = (v.copy(), recname) + prop
            elif isinstance(v, PhotometricDataArray):
                # by a PhotometricDataArray object
                # force load all elements if not already loaded
                for x in v.reshape(-1):
                    pass
                fields = list(self.dtype.names)
                fields.remove('file')
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

    def _memory_dump(self, forced=False, verbose=False):
        """Check the size of object, free memory by deleting
        """
        if not forced:
            total_counts = (self['count']*self['loaded'].astype('i')).sum()
            sz = _memory_size(total_counts*1.2)
            if sz<self.maxmem:
                return False
        # free memory by deleting all PhotometricData instances
        if self.datafile is None:
            warnings.warn("Data file not specified.  Memory dump failed.",
                    MemoryWarning)
            return False
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

    def _load_data(self, x):
        """Load photometric data record

         x : reference to an element in `PhotometricDataArray`
        """
        if self.datafile is None:
            raise ValueError('Data file not specified.')
        datafile, datadir = self._path_name(self.datafile)
        f = os.path.join(datadir, x['file'])
        if os.path.isfile(f):
            self._memory_dump()
            x['pho'] = PhotometricData(f)
            x['loaded'] = True
            x['flushed'] = True
        else:
            raise IOError('Data record not found from file {}'.
                    format(f))

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
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
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
            super().__setitem__(k, info[k].reshape(self.shape))

    @classmethod
    def from_file(cls, infile):
        shape = phoarr_info(infile, shape=True)
        obj = cls(shape)
        obj.read(infile)
        return obj

    def trim(self,  verbose=True, **kwargs):
        """Trim photometric data in the array

        See `PhotometricData.trim()` for details
        """
        tag = -0.1
        flattened = self.reshape(-1)
        masked = flattened['masked']
        # prepare property array
        fields = list(flattened.dtype.names)
        fields.remove('pho')
        fields.remove('file')
        prop = np.asarray(flattened[fields]).copy()
        # loop through all elements
        for i in range(self.size):
            prog = float(i)/self.size*100
            if verbose:
                if prog > tag:
                    import sys
                    sys.stdout.write('{:.1f} completed\r'.format(prog))
                    sys.stdout.flush()
                    tag = np.ceil(prog + 0.1)
            if masked[i]:
                continue
            x = flattened[i]
            orig_len = len(x)
            x.trim(**kwargs)
            final_len = len(x)
            if final_len != orig_len:
                prop['count'][i] = final_len
                flattened['flushed'][i] = False
                if len(x) == 0:
                    prop['incmin'][i] = 0.
                    prop['incmax'][i] = 0.
                    prop['emimin'][i] = 0.
                    prop['emimax'][i] = 0.
                    prop['phamin'][i] = 0.
                    prop['phamax'][i] = 0.
                    prop['latmin'][i] = 0.
                    prop['latmax'][i] = 0.
                    prop['latmin'][i] = 0.
                    prop['latmax'][i] = 0.
                    flattened['masked'][i] = True
                else:
                    prop['incmin'][i] = x.inclim[0].to('deg').value
                    prop['incmax'][i] = x.inclim[1].to('deg').value
                    prop['emimin'][i] = x.emilim[0].to('deg').value
                    prop['emimax'][i] = x.emilim[1].to('deg').value
                    prop['phamin'][i] = x.phalim[0].to('deg').value
                    prop['phamax'][i] = x.phalim[1].to('deg').value
                    prop['latmin'][i] = x.latlim[0].to('deg').value
                    prop['latmax'][i] = x.latlim[1].to('deg').value
                    prop['latmin'][i] = x.lonlim[0].to('deg').value
                    prop['latmax'][i] = x.lonlim[1].to('deg').value
        # update properties
        prop['masked'] = flattened['masked']
        prop['loaded'] = flattened['loaded']
        prop['flushed'] = flattened['flushed']
        for f in fields:
            super().__setitem__(f, prop[f].reshape(self.shape))


class ModelArray(np.ndarray):
    """Model array class

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
        obj.extra = {}
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.__version__ = getattr(obj, '__version__', '1.0.0')
        self._model_class = getattr(obj, 'model_class', None)
        self._param_names = getattr(obj, 'param_names', None)
        self.extra = getattr(obj, 'extra', {})

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
        if isinstance(out, ModelArray):
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
        elif isinstance(self[k], ModelArray):
            if isinstance(v, self.model_class):
                for x in np.nditer(self[k], flags=['refs_ok'],
                        op_flags=['readwrite']):
                    pars = tuple([getattr(v, p).value for p in
                        self.param_names])
                    x[...] = (False,) + pars
            elif isinstance(v, ModelArray):
                for f in list(self.dtype.names):
                    from copy import deepcopy
                    self[f][k] = deepcopy(v[f])
            else:
                raise ValueError('Only `{}` or `ModelArray`'
                    ' instance allowed.'.format(self.model_class.__class__.
                    __name__))
        else:
            raise ValueError('Only `{}` or `ModelArray`'
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


def fit(args):
    i, data, mask, model, ilim, elim, alim, rlim, verbose, fitter, kwargs \
        = args
    if (not mask) and isinstance(data, PhotometricData):
        d = data.copy()
        d.validate()
        d.trim(ilim=ilim, elim=elim, alim=alim, rlim=rlim)
        if len(d) > 10:
            fitter = d.fit(model, fitter=fitter(), verbose=False,
                    **kwargs)
        else:
            fitter = None

    if verbose:
        print('Data cell {0}'.format(i), end=': ')
        if (fitter is not None) and hasattr(fitter, 'model'):
            if not hasattr(fitter.model, '__iter__'):
                print(fitter.model.__repr__())
            else:
                params = np.array([m.parameters for m in fitter.model])
                print(type(fitter.model[0])(*params.T,
                      n_models=params.shape[0]))
        else:
            print('not fitted.')

    return fitter


class PhotometricDataArrayFitter():
    """Fitter to fit `PhotometricDataArray` class object

    One can directly use the fitter by providing an astropy fitter
    class as a parameter in the call:

    >>> fitter = PhotoemtricDataArrayFitter()
    >>> models = fitter(m0, data, fitter=PhotometricMPFitter)

    Or can subclass this class with the default fitter specified.  See
    `PhotometricDataArrayMPFitter` as an example.
    """
    def __init__(self):
        self.fitted = False

    def __call__(self, model, data, fitter=None, ilim=None, elim=None,
        alim=None, rlim=None, multi=4, **kwargs):
        """Fit PhotometricDataArray to model

        model : `~astropy.modeling.Model` instance
            Model to be fitted
        data : `PhotometricDataArray`
            Data to be fitted
        fitter : astropy fitter class
            Fitter used to fit the data
        ilim : 2-element array like number or `astropy.units.Quantity`
            Limit of incidence angle
        elim : 2-element array like number or `astropy.units.Quantity`
            Limit of emission angle
        alim : 2-element array like number or `astropy.units.Quantity`
            Limit of phase angle
        rlim : 2-element array like number or `astropy.units.Quantity`
            Limit of bidirectional reflectance
        **kwargs : dict
            Keyword arguments accepted by `PhotometricData.fit()`

        Return : `ModelArray` instance
        """
        verbose = kwargs.pop('verbose', True)
        if fitter is not None:
            self.fitter = fitter
        if not hasattr(self, 'fitter'):
            raise ValueError('Fitter not defined.')
        self.model = ModelArray(data.shape, type(model))
        model1d = self.model.reshape(-1)
        self.fit_info = np.zeros(data.shape, dtype=object)
        fit_info1d = self.fit_info.reshape(-1)
        self.fit = np.zeros(data.shape, dtype=np.ndarray)
        fit1d = self.fit.reshape(-1)
        self.RMS = np.zeros(data.shape, dtype=object)
        rms1d = self.RMS.reshape(-1)
        self.RRMS = np.zeros(data.shape, dtype=object)
        rrms1d = self.RRMS.reshape(-1)
        self.mask = np.ones(data.shape, dtype=bool)
        fitmask1d = self.mask.reshape(-1)

        data1d = data.reshape(-1)
        mask = data['masked'].reshape(-1)

        from multiprocessing import Pool, Process, Queue
        from time import sleep
        pool = Pool(processes=multi)
        pool_params = [(i, data1d[i], mask[i], model, ilim, elim, alim, rlim,
                        verbose, self.fitter, kwargs)
                        for i in range(data1d.size)]
        results = pool.map(fit, pool_params)

        for i, fitter in enumerate(results):
            if (fitter is not None) and hasattr(fitter, 'model'):
                if hasattr(fitter.model, '__iter__'):
                    # assemble to a model set if spectral data
                    params = np.array([m.parameters for m in fitter.model])
                    model_set = type(fitter.model[0])(*params.T,
                        n_models=params.shape[0])
                else:
                    # single band data
                    model_set = fitter.model
                model1d[i] = model_set
                fit_info1d[i] = fitter.fit_info
                fit1d[i] = fitter.fit
                rms1d[i] = fitter.RMS
                rrms1d[i] = fitter.RRMS
                fitmask1d[i] = False
            elif fitter is not None:
                fitmask1d[i] = True
                rms1d[i] = -1e10
                rrms1d[i] = -1e10
            else:
                fitmask1d[i] = True
                rms1d[i] = 0.
                rrms1d[i] = 0.

        self.fitted = True
        self.model.extra['RMS'] = self.RMS
        self.model.extra['RRMS'] = self.RRMS
        pool.close()
        return self.model


class PhotometricDataArrayMPFitter(PhotometricDataArrayFitter):
    fitter = PhotometricMPFitter

