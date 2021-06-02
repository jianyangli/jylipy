"""
Image processing and analysis related classes and functions
"""

__all__ = ['Centroid', 'ImageSet', 'Background']

import numpy as np
from astropy.io import fits, ascii
from astropy import nddata, table
from photutils.centroids import centroid_2dg, centroid_com
from .saoimage import getds9, CircularRegion, CrossPointRegion, TextRegion, RegionList
from .core import resmean, gaussfit


class ImageSet():
    """Convenience class to organize a set of images.

    Attributes
    ----------
    .image : 3D or higher dimension array of shape (..., N, M), images
        to be centroided.  The shape of each image is (N, M).
    .file : string array, FITS file names of images to be centroided
    ._ext : string, number, or array of them
        The FITS extensions of images in the source files.  If array,
        it has the same shape as `.file`.
    .loader : function
        Image loader to provide flexibility in accessing various image
        formats.
    .attr : list
        A list of other attributes set by `**kwargs` to `__init__()`.
    ._1d : dict
        The 1D views of `.image`, `.file`, `._ext`, and all listed in
        `.attr`.
    """
    def __init__(self, im, ext=0, loader=None, **kwargs):
        """
        Parameters
        ----------
        im : string, string sequence, array-like numbers of shape[..., N, M]
            File name(s) if string or string sequence.  Images must be
            stored in fits files with the extension specified by `ext`.
            Image(s) if array-like numbers.  Has to be 2D or high
            dimension.  The last two dimension are the shape of images
            (NxM).
        ext : int, str, or sequence of them, optional
            The extension of FITS files that stores the images to be
            loaded.  If sequence, then it must have the same shape as
            input file names.
        loader : function, optional
            A function to load an image from input file.  It provides
            flexibility in accessing various image formats.  The call
            signature of the loader is
                im = loader(input, **kwargs)
            where input is a string of file name, and it returns an image
            in a 2D array.
            If `loader` == `None`, then a FITS file will be assumed.
        **kwargs : int or sequence of int, optional
            Any other arguments needed.  If a scaler, it is assumed to
            be the same for all images.  If sequences, it must have the
            same shape as input file names or images.
        """
        # process input images/files
        im = np.array(im)
        if im.dtype.kind in ['S', 'U']:
            # image file name
            if im.ndim == 0:
                self.file = np.array([im])
            else:
                self.file = im
            self.image = None
            self._shape = self.file.shape
            self._size = self.file.size
        elif im.dtype.kind in ['i', 'u', 'f']:
            # image array
            if im.ndim < 2:
                raise ValueError('for images, `.ndim` must be >=2')
            if im.ndim == 2:
                self.image = np.zeros(1, dtype='O')
                self.image[0] = im
            else:
                self.image = np.zeros(im.shape[:-2], dtype='O')
                _from = im.reshape(-1, im.shape[-2], im.shape[-1])
                _to = self.image.reshape(-1)
                for i in range(len(_from)):
                    _to[i] = _from[i]
            self.file = None
            self._shape = self.image.shape
            self._size = self.image.size
        elif im.dtype.kind == 'O':
            types = [x.dtype.kind in ['i', 'u', 'f'] for x in im.reshape(-1)]
            if not np.all(types):
                raise ValueError('unrecognized image type')
            self.image = np.zeros(im.shape, dtype='O')
            _from = im.reshape(-1)
            _to = self.image.reshape(-1)
            for i in range(len(_from)):
                _to[i] = _from[i]
            self.file = None
            self._shape = self.image.shape
            self._size = self.image.size
        else:
            raise ValueError('unrecognized input type')
        # FITS extension
        if isinstance(ext, str) or (not hasattr(ext, '__iter__')):
            self._ext = ext
        else:
            if np.asarray(ext).shape != self._shape:
                raise ValueError('`ext` must have the same shape as `file` '
                        'or `image`')
            self._ext = np.asarray(ext)
        # other keywrods
        self.attr = []
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                n = '_'+k
                self.attr.append(n)
                if isinstance(v, str) or (not hasattr(v, '__iter__')):
                    setattr(self, n, v)
                else:
                    if np.asarray(v).shape != self._shape:
                        raise ValueError('invalide shape for kwargs {}: '
                                'expect '. format(k) + str(self._shape) + \
                                ', got ' + str(np.asarray(v).shape))
                    setattr(self, n, np.asarray(v))
        self.loader = loader
        # generate flat views
        self._generate_flat_views()

    def _generate_flat_views(self):
        self._1d = {}
        self._1d['image'] = None if (self.image is None) else \
                    self.image.reshape(-1)
        self._1d['file'] = None if (self.file is None) else \
                    self.file.reshape(-1)
        attr = ['_ext'] + self.attr
        for k in attr:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                self._1d[k] = v.reshape(-1)
            else:
                self._1d[k] = np.array([v] * self._size)

    def _load_image(self, i):
        """Load the ith image in flattened file name list
        """
        if self.file is None:
            raise ValueError('no input file specified')
        if self.image is None:
            self.image = np.zeros_like(self.file, dtype='O')
            self.image[:] = None
            self._1d['image'] = self.image.reshape(-1)
        if self.loader is None:
            self._1d['image'][i] = \
                    fits.open(self._1d['file'][i])[self._1d['_ext'][i]].data
        else:
            self._1d['image'][i] = self.loader(self._1d['file'][i])

    def _ravel_indices(self, index):
        """Convert index to 1d index

        Parameters
        ----------
        index : None, or int, slice, list of int, or tuple of them
            If `None`, return 1d indices of all items
            If int, slice, or list of int, then specify the 1d index
                (index of flattened `._1d['file']` or `._1d['image']`)
            If tuple, then specify the multi-dimentional index.

        Return
        ------
        index1d : 1d array or tuple of 1d arrays
            The index of flattened `._1d['file']` or `._1d['image']`
        """
        if index is None:
            _index = np.r_[range(self._size)]
        elif isinstance(index, tuple):
            # multi-dimensional indices
            _index = ()
            for i, ind in enumerate(index):
                if isinstance(ind, slice) and (ind.stop is None):
                    ind = slice(ind.start, self._shape[i], ind.step)
                _index = _index + (np.r_[ind], )
            len_index = len(_index)
            if len_index < len(self._shape):
                for i in range(len(self._shape) - len_index):
                    _index = _index + (np.r_[range(self._shape[len_index+i])],)
            _index = np.ravel_multi_index(_index, self._shape)
            if not hasattr(_index, '__iter__'):
                _index = [_index]
        else:
            # 1d indices
            _index = np.r_[index]
        return _index

    def write(self, outfile, format=None, save_images=False, **kwargs):
        """Write centers to output file

        Parameters
        ----------
        outfile : str
            Output file name
        format : [None, 'ascii', 'fits'], optional
            Format of output file.
            If `None`, then the format is implied by the extension of
            `outfile`.  If no extension or extension not recognized,
            then an error will be issued.
            If 'ascii', then the centers are flattened if
            `self.file.ndim` > 1, and saved to an ASCII table.  The
            first column is either image file names if provided, or
            image indices otherwise.  The second and third columns are
            centers (x, y), with headings 'xc', 'yc'.  The last column
            is 'status', with value 1 means center is measured successfully
            and 0 means center is not measured.
            If 'fits', then the centers are saved in a FITS array in the
            primary extension, status in the first extension, and the
            image file names in a binary table in the second extension
            if provided.
        save_images : bool, optional
            If `True`, then if `.file` is None and images are provided,
            then save images to a FITS file named
            `'{}_images.fits'.format(outfile)`.
        **kwargs : dict, optional
            Other keyword arguments accepted by the `astropy.io.ascii.write`
            or `astropy.io.fits.HDUList.writeto`.

        The data will be organized in a table and saved in either an ASCII
        file or a FITS file.  The table will include file names (if
        available), extension number (if applicable), and all information
        supplied by **kwargs.  If the image set is a multi-dimensional
        array, then the dimension and shape of the set is also saved in
        the FITS binary table headers if the output is a FITS file.  If
        the output is an ASCII file, then the shape information of image
        set will be discarded and all data saved in flat arrays.
        """
        cols = []
        for k in self.attr:
            cols.append(table.Column(self._1d[k], name=k.strip('_')))
        if self.file is not None:
            cols.insert(0, table.Column(self._1d['file'], name='file'))
            cols.insert(1, table.Column(self._1d['_ext'], name='ext'))
        out = table.Table(cols)
        if format is None:
            format = ''
            from os.path import splitext
            ext = splitext(outfile)[1].lower()
            if ext in ['.fits', '.fit']:
                format = 'fits'
            elif ext in ['.csv', '.tab']:
                format = 'ascii'
        if format == 'ascii':
            out.write(outfile, **kwargs)
        elif format == 'fits':
            outfits = fits.HDUList(fits.PrimaryHDU())
            tblhdu = fits.BinTableHDU(out, name='info')
            tblhdu.header['ndim'] = len(self._shape)
            for i in range(len(self._shape)):
                tblhdu.header['axis{}'.format(i)] = self._shape[i]
            outfits.append(tblhdu)
            outfits.writeto(outfile, **kwargs)
        else:
            raise ValueError('unrecognized output format')
        if save_images:
            if self.file is None:
                from os import path
                hdu0 = fits.PrimaryHDU()
                hdu0.header['ndim'] = len(self._shape)
                for i in range(len(self._shape)):
                    hdu0.header['axis{}'.format(i)] = self._shape[i]
                hdulist = fits.HDUList([hdu0])
                for im in self._1d['image']:
                    hdulist.append(fits.ImageHDU(im))
                outname = '{}_images.fits'.format(path.splitext(outfile)[0])
                overwrite = kwargs.pop('overwrite', False)
                hdulist.writeto(outname, overwrite=overwrite)

    def read(self, infile, format=None, **kwargs):
        """Read centers from input file

        Parameters
        ----------
        infile : str
            Input file name
        format : [None, 'ascii', 'fits'], optional
            Format of input file.
            If `None`, then the format is implied by the extension of
            `outfile`.  If no extension or extension not recognized,
            then an error will be issued.
            If 'ascii':  If file names are available from the input file,
            then they will replace whatever in `.file` attribute, and
            any loaded images will be cleared.  The centers and status
            will be reshaped to the same shape as `.file` or `.image`.
            If 'fits', then the centers and status will be loaded, and
            if file names are avaialble from input file, then they will
            replace whatever in `.file` attribute, and reshaped to the
            same shape as centers and status.  Any loaded images will
            be cleared in this case.
        **kwargs : dict, optional
            Other keyword arguments accepted by the `astropy.io.ascii.read`.
        """
        if format is None:
            format = ''
            from os.path import splitext
            ext = splitext(infile)[1].lower()
            if ext in ['.fits', '.fit']:
                format = 'fits'
            elif ext in ['.csv', '.tab']:
                format = 'ascii'
        if format == 'ascii':
            intable = ascii.read(infile, **kwargs)
            ndim = 0
        elif format == 'fits':
            with fits.open(infile) as _f:
                intable = table.Table(_f['info'].data)
                ndim = _f['info'].header['ndim']
                shape = ()
                for i in range(ndim):
                    shape = shape + (_f['info'].header['axis{}'.format(i)],)
        else:
            raise ValueError('unrecognized input format')
        keys = intable.keys()
        if 'file' in keys:
            self.file = np.array(intable['file'])
            self._ext = np.array(intable['ext'])
            keys.remove('file')
            keys.remove('ext')
            self.image = None
        else:
            from os import path
            imgfile = '{}_images.fits'.format(path.splitext(infile)[0])
            if not path.isfile(imgfile):
                raise IOError('input image file not found')
            self.file = None
            self._ext = None
            # load image
            with fits.open(imgfile) as _f:
                ndim = _f[0].header['ndim']
                shape = ()
                for i in range(ndim):
                    shape = shape + (_f[0].header['axis{}'.format(i)],)
                self.image = np.zeros(shape, dtype='O')
                image1d = self.image.reshape(-1)
                for i in range(len(_f)-1):
                    image1d[i] = _f[i+1].data
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
        keys = self.attr if self._ext is None else ['_ext'] + self.attr
        for k in keys:
            v = getattr(self, k)
            if np.all(v == v[0]):
                setattr(self, k, v[0])
        # generate flat view
        self._generate_flat_views()

    @classmethod
    def from_fits(cls, infile, loader=None):
        obj = cls('')
        obj.read(infile, format='fits')
        obj.loader = loader
        return obj


class Centroid(ImageSet):
    """Image centroiding

    Extra attributes from `ImageSet`
    --------------------------------
    .center : float array of shape (..., 2), centers, the shape of
        (...) is the same as `.file` or `.image`.
    ._yc, ._xc : float array of the same shape as `.file` or `.image`
        The (y, x) coordinates of centers
    ._status : bool array
        Centroiding status.  Same shape as `.file` or `.image`.
    ._box : int array of the same shape as `.file` or `.image`
        Box size for centroiding
    """
    def __init__(self, *args, box=5, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs : see `ImageSet`
        box : int or sequence of int, optional
            Box size for centroid refining.  See `mskpy.gcentroid`.  If
            sequence, it must have the same shape as input file names
            or images.
        """
        kwargs['box'] = box
        super().__init__(*args, **kwargs)
        # preset other attributes
        self.center = np.zeros(self._shape + (2,))
        self._status = np.zeros(self._shape, dtype=bool)
        self._1d['_yc'] = self.center[..., 0].reshape(-1)
        self._1d['_xc'] = self.center[..., 1].reshape(-1)
        self._1d['_status'] = self._status.reshape(-1)
        self.attr.extend(['_xc', '_yc', '_status'])

    def centroiding(self, ds9=None, newframe=False, refine=True, box=5,
                    resume=True, verbose=True):
        """Interactive centroiding

        Parameters
        ----------
        ds9 : ds9 instance, or None, optional
            The target DS9 window.  If `None`, then the first opened
            DS9 window will be used, or a new window will be opened
            if none exists.
        newframe : bool, optional
            If set `False`, then the image will be displayed in the
            currently active frame in DS9, and the previous image
            will be overwritten.  By default, a new frame will be
            created to display the image.
        refine : bool, optional
            If `True`, then the centroid will be refined by
            `mskpy.gcentroid`.
        box : number, optional
            Box size for centroid refining.  See `mskpy.gcentroid`
        resume : bool, optional
            If `True`, then resume centroiding for images with
            `_status == False`, otherwise do centroiding for all images.
        verbose : bool, optional
            If `False`, then all screen output is suppressed.  Note that
            this only suppresses information output.  All error or
            warning messages will still be output to screen.
        """
        # Loop through all images
        if verbose:
            print('Centroiding {} images from input\n'.format(self._size))
        i, j = 0, 0
        retry = False
        if ds9 is None:
            ds9 = getds9('Centroid')
        while i < self._size:
            if resume and self._1d['_status'][i]:
                i += 1
                continue
            if (self.image is None) or (self._1d['image'][i] is None):
                if verbose:
                    print('Image {} in the list: {}.'.format(i,
                            self._1d['file'][i]))
                self._load_image(i)
            else:
                if verbose:
                    print('Image {} in the list.'.format(i))
            if not retry:
                ds9.imdisp(self._1d['image'][i], newframe=newframe,
                        verbose=verbose)
            else:
                if verbose:
                    print('Retry clicking near the center.')
            retry = False
            ct = ds9.get('imexam coordinate image').split()
            if len(ct) != 0:
                ct = [float(ct[i]) for i in [1,0]]
                if refine:
                    self._1d['_yc'][i], self._1d['_xc'][i] = \
                            centroid(self._1d['image'][i], ct,
                            box=self._1d['_box'][i], verbose=verbose)
                else:
                    self._1d['_yc'][i], self._1d['_xc'][i] = ct
                if verbose:
                    print('Centroid at (x, y) = ({:.4f}, {:.4f})'.format(
                            self._1d['_yc'][i], self._1d['_xc'][i]))
                self._1d['_status'][i] = True
                j += 1
            else:
                # enter interactive session
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

        if verbose:
            print('{} images measured out of a total of {}'.format(j,
                    self._size))

    def refine(self, index=None, verbose=False):
        """Refine centroid

        Parameters
        ----------
        index : int, slice, list of int, or tuple of them, optional
            If int, slice, or list of int, then it's the index (indices)
            of flattened `.file` or `.image` whose centroid will be
            refined.
            If tuple, then it's the multi-dimentional indices of `.file`
            or `.image` whose centroid will be refined.
            If None, all centroids will be refined.
            However, refinement skips those with `.status` == False.
        verbose : bool, optional
            Print out information
        """
        # process index
        _index = self._ravel_indices(index)
        # _center, _status, _image1d, _box
        n = 1
        nn = len(_index)
        for i in _index:
            if verbose:
                if self.file is None:
                    print('Image {} of {}:'.format(n, nn))
                else:
                    print('Image {} of {}: {}'.format(n, nn,
                            self._1d['file'][i]))
            if not self._1d['_status'][i]:
                if verbose:
                    print('...centroid not measured, skipped\n')
                continue  # skip if hasn't measured
            if (self.image is None) or (self._1d['image'][i] is None):
                self._load_image(i)
            self._1d['_yc'][i], self._1d['_xc'][i] = \
                    centroid(self._1d['image'][i],\
                            (self._1d['_yc'][i], self._1d['_xc'][i]),
                            box=self._1d['_box'][i], verbose=verbose)
            n += 1
            if verbose:
                print()

    def read(self, infile, **kwargs):
        super().read(infile, **kwargs)
        self.center = np.stack([self._yc, self._xc], \
                axis=-1).reshape(self._shape + (2,))
        if not hasattr(self._status, '__iter__'):
            status = self._status
            self._status = np.zeros(self._shape, dtype=bool)
            self._status[:] = status

    def show(self, index=None, ds9=None):
        """Show centroid in DS9

        Parameters
        ----------
        ds9 : ds9 instance, or None, optional
            The target DS9 window.  If `None`, then the first opened
            DS9 window will be used, or a new window will be opened
            if none exists.
        newframe : bool, optional
            If set `False`, then the image will be displayed in the
            currently active frame in DS9, and the previous image
            will be overwritten.  By default, a new frame will be
            created to display the image.
        """
        if ds9 is None:
            ds9 = getds9('Centroid')
        _index = self._ravel_indices(index)
        for i in _index:
            if (self.image is None) or (self._1d['image'][i] is None):
                self._load_image(i)
            ds9.imdisp(self._1d['image'][i])
            xc = self._1d['_xc'][i]
            yc = self._1d['_yc'][i]
            ds9.sets(['pan to {} {}'.format(xc, yc),
                      'zoom to 2'])
            r = CircularRegion(xc, yc, 3)
            c = CrossPointRegion(xc, yc)
            t = TextRegion(xc, yc+ds9.height/2/2-10,
                    text='{}: {}'.format(i, self._1d['file'][i]),
                    color='white', font='helvetica 15 bold roman')
            RegionList([r, c, t]).show(ds9=ds9)


class Background(ImageSet):
    """Image background

    Extra attributes from `ImageSet`
    --------------------------------
    .background : float array of the same shape as `.images` or `.file`
        Measured background
    ._background_region#, ._background_error_region# : float array of
        the same shape as `.image` or `.file`
        The measured background and error of region#, where # is int
        starting from 0.
     : float array of the same shape as
    ._region#_x1, _region#_y1, _region#_x2, _region#_y2 : int arrays
        The (x, y) coordinates of the lower-left and upper right corners
        for region#, where # is int starting from 0.
    ._n_regions : int or int array
        The number of regions for each image
    ._gain : float or float array
        If exist, then it is the gain of images DN = e- * gain, used to
        calculate photon noise.
    """
    def __init__(self, *args, region=None, **kwargs):
        """
        Parameters
        ----------
        region : None or array-like int
            The region(s) from which the background is measured.  When
            it's array-like, the length of the last dimension must be >=4
            to specify the [y1, x1, y2, x2] pixel coordinates of the
            lower-left and upper-right corners of region(s).
            If None: Background is measured from the whole image for all
                images.
            If 1d array: Background is measured from one single region
                for all images.
            If 2d array of shape (M, N): Background is measured from all
                M regions specified by `region` for all images.
            If 3d or higher dimension array of shape (..., M, N):
                Background is measured from all M regions individually
                specified for each image.  where ... must have the
                same shape as `.image` or `.file`.
        """
        super().__init__(*args, **kwargs)
        if region is None:
            self._region0_x1 = 0
            self._region0_x2 = 0
            self._region0_y1 = 0
            self._region0_y2 = 0
            self._n_regions = 1
            self.attr.extend(['_n_regions', '_region0_x1', '_region0_x2',
                              '_region0_y1', '_region0_y2'])
        else:
            region = np.array(region, dtype='O')
            same_number_of_region = True
            try:
                region = region.astype('i')
            except ValueError:
                pass
            if region.dtype.kind == 'i':
                # when numbers of regions for all images are the same
                if region.shape[-1] < 4:
                    raise ValueError("the last dimension of 'region' must "
                        "be >= 4")
                if region.ndim == 1:
                    self._region0_y1, self._region0_x1, self._region0_y2, \
                            self._region0_x2 = region[:4]
                    self._n_regions = 1
                    self.attr.extend(['_n_regions',
                                      '_region0_x1', '_region0_x2',
                                      '_region0_y1', '_region0_y2'])
                elif region.ndim == 2:
                    shape = region.shape
                    self._n_regions = shape[0]
                    self.attr.append('_n_regions')
                    for i in range(shape[0]):
                        reg = ['_region{}_y1'.format(i),
                               '_region{}_x1'.format(i),
                               '_region{}_y2'.format(i),
                               '_region{}_x2'.format(i)]
                        for n in range(4):
                            setattr(self, reg[n], region[i, n])
                        self.attr.extend(reg)
                else:
                    shape = region.shape
                    if shape[:-2] != self._shape:
                        raise ValueError("The shape of 'region' is not "
                            "compatible with the shape of `.file` or `.image`")
                    _region = region.reshape(-1, shape[-2], shape[-1])
                    self._n_regions = shape[-2]
                    self.attr.append('_n_regions')
                    for i in range(self._n_regions):
                        reg = ['_region{}_y1'.format(i),
                               '_region{}_x1'.format(i),
                               '_region{}_y2'.format(i),
                               '_region{}_x2'.format(i)]
                        for n in range(4):
                            setattr(self, reg[n], [])
                        for j in range(self._size):
                            for n in range(4):
                                getattr(self, reg[n]).append(_region[j, i, n])
                        for n in range(4):
                            rr = np.array(getattr(self, reg[n]))
                            rr = rr.reshape(self._shape)
                            setattr(self, reg[n], rr)
                        self.attr.extend(reg)
            else:
                # when the number of regions for each image is different
                shape = region.shape
                if region.shape != self._shape:
                    raise ValueError("The shape of 'region' is not compatible "
                        "with the shape of `.file` or `.image`")
                _region = region.reshape(-1)
                self._n_regions = np.array([len(r) for r in _region]).\
                        reshape(self._shape)
                self.attr.append('_n_regions')
                n_reg_1d = self._n_regions.reshape(-1)
                n_reg_max = self._n_regions.max()
                for i in range(n_reg_max):
                    reg = ['_region{}_y1'.format(i),
                           '_region{}_x1'.format(i),
                           '_region{}_y2'.format(i),
                           '_region{}_x2'.format(i)]
                    for n in range(4):
                        setattr(self, reg[n], [])
                    for j in range(self._size):
                        if i < n_reg_1d[j]:
                            for n in range(4):
                                getattr(self, reg[n]).append(_region[j][i][n])
                        else:
                            for n in range(4):
                                getattr(self, reg[n]).append(0)
                    for n in range(4):
                        rr = np.array(getattr(self, reg[n]))
                        rr = rr.reshape(self._shape)
                        setattr(self, reg[n], rr)
                    self.attr.extend(reg)
        self._generate_flat_views()

    def measure(self, index=None, method='mean'):
        """Measure background

        Parameters
        ----------
        index : None, or int, slice, list of int, or tuple of them
            Index/indices of images to be measured.
            If `None`, measure all images
            If int, slice, or list of int, then they are the index of
                images in the flattened arrays to be measured
                (`._1d['file']` or `._1d['image']`)
            If tuple, then specify the multi-dimentional index of images
                to be measured.
        method : str in ['mean', 'gauss','median'], optional
            The method to measure background.
            'mean': Uses resistant mean, rather than simple mean.
            'gauss': Uses a Gaussian fit to the histogram of image/regions
                to estimate the background and standard deviation.
            'median': Uses median of image/regions
        """
        from astropy.stats import sigma_clipped_stats
        # gain settings
        gain = self._1d['_gain'] if '_gain' in self.attr \
                                                    else np.ones(self._size)
        # prepare storage
        max_n_regions = np.array(self._1d['_n_regions']).max()
        for i in range(max_n_regions):
            regstr = '_region{}'.format(i)
            setattr(self, '_background' + regstr, np.full(self._shape, np.nan))
            setattr(self, '_background_error' + regstr,
                                        np.full(self._shape, np.nan))
            self.attr.append('_background' + regstr)
            self.attr.append('_background_error' + regstr)
        self.background = np.full(self._shape, np.nan)
        self.background_error = np.full(self._shape, np.nan)
        self.attr.extend(['background', 'background_error'])
        self._generate_flat_views()
        # loop through all images to measure background
        for i in range(self._size):
            if self.image is None or self._1d['image'][i] is None:
                self._load_image(i)
            # loop through all regions
            for j in range(self._1d['_n_regions'][i]):
                regstr = '_region{}'.format(j)
                y1 = round(self._1d[regstr+'_y1'][i])
                x1 = round(self._1d[regstr+'_x1'][i])
                y2 = round(self._1d[regstr+'_y2'][i])
                x2 = round(self._1d[regstr+'_x2'][i])
                imsz = self._1d['image'][i].shape
                # all zero means using whole image
                if (np.array([y1, x1, y2, x2]) == 0).all():
                    y2 = imsz[0] - 1
                    x2 = imsz[1] - 1
                if (y1 >= 0) and (y1 < imsz[0]) \
                        and (x1 >= 0) and (x1 < imsz[1]) \
                        and (y2 >= 0) and (y2 < imsz[0]) \
                        and (x2 >= 0) and (x2 < imsz[1]) \
                        and (y1 < y2) and (x1 < x2):
                    # measure background from valid region
                    subim = self._1d['image'][i][y1:y2, x1:x2]
                    mean, median, stddev = sigma_clipped_stats(subim.data,
                                                           subim.mask)
                    if method == 'median':
                        self._1d['_background'+regstr][i] = median
                        self._1d['_background_error'+regstr][i] = \
                                    np.sqrt(stddev**2 + bg*gain[i])
                    elif method == 'mean':
                        self._1d['_background'+regstr][i] = mean
                        self._1d['_background_error'+regstr][i] = \
                                    np.sqrt(stddev**2 + mean*gain[i])
                    elif method == 'gaussian':
                        subim = subim.data[~subim.mask]
                        hist, bin = np.histogram(subim, bins=100,
                                range=[mean-10*stddev, mean+10*stddev])
                        par0 = [max(hist), mean, stddev]
                        x = (bin[0:-1] + bin[1:]) / 2
                        par = gaussfit(x, hist, par0=par0)[0]
                        self._1d['_background'+regstr][i] = par[1]
                        self._1d['_background_error'+regstr][i] = \
                                    np.sqrt(par[2]**2 + par[1]*gain[i])
                else:
                    # skip invalid region
                    import warning
                    warning.warn("invalide region skipped:\n  Image {}, "
                        "region ({} {} {} {})".format(self._1d['file'][i],
                            y1, x1, y2, x2))

            # calculate average background for image
            bgs = np.array([self._1d['_background_region{}'.format(j)][i] \
                    for j in range(self._1d['_n_regions'][i])])
            bgs_err = np.array([self._1d['_background_error_region{}'. \
                    format(j)][i] for j in range(self._1d['_n_regions'][i])])
            bgs_err2 = bgs_err * bgs_err
            dom = np.nansum(1 / bgs_err2)
            self._1d['background'][i] = np.nansum(bgs / bgs_err2) / dom
            self._1d['background_error'][i] = np.sqrt(1 / dom)


def centroid(im, center=None, error=None, mask=None, method=0, box=6,
        tol=0.01, maxiter=50, threshold=None, verbose=False):
    """Wrapper for photutils.centroiding functions

    Parameters
    ----------
    im : array-like, astropy.nddata.NDData or subclass
        Input image
    center : (y, x), optional
        Preliminary center to start the search
    error : array-like, optional
        Error of the input image.  If `im` is NDData type, then `error`
        will be extracted from NDData.uncertainty.  This keyword
        overrides the uncertainty in NDData.
    mask : array-like bool, optional
        Mask of input image.  If `im` is NDData type, then `mask` will
        be extracted from NDData.mask.  This keyword overrides the mask
        in NDData.
    method : int or str, optional
        Method of centroiding:
        [0, '2dg', 'gaussian'] - 2-D Gaussian
        [1, 'com'] - Center of mass
        [2, 'geom', 'geometric'] - Geometric center
    box : int, optional
        Box size for the search
    tol : float, optional
        The tolerance in pixels of the center.  Program exits iteration
        when new center differs from the previous iteration less than
        `tol` or number of iteration reaches `maxiter`.
    maxiter : int, optional
        The maximum number of iterations in the search
    threshold : number, optional
        Threshold, only used for method=2
    verbose : bool, optional
        Print out information

    Returns
    -------
    (y, x) as a numpy array

    This program uses `photutils.centroids.centroid_2dg` or
    `photutils.centroids.centroid_com`.
    """
    # extract error array if provided
    if isinstance(im, nddata.NDData):
        if error is None:
            if im.uncertainty is not None:
                error = im.uncertainty.array
        if mask is None:
            if im.mask is not None:
                mask = im.mask
        im = im.data
    # preliminary center at image center if not provided
    if center is None:
        center = np.asarray(im.shape)/2.
    else:
        center = np.asarray(center)
    # methods
    if method not in [0, 1, 2, '2dg', 'gaussian', 'com', 'geom', 'geometric']:
        raise ValueError("unrecognized `method` {} received.  Expected to "
            "be [0, '2dg', 'gaussian'] or [1, 'com'] or [2, 'geom', "
            "'geometric']".format(method))
    if (method in [2, 'geom', 'geometric']) and (threshold is None):
        raise ValueError('threshold is not specified')
    # print out information
    if verbose:
        print('Image provided as a '+str(type(im))+', shape = ', im.shape)
        print(('Centroiding image in {0}x{0} box around ({1:.4f}, {2:.4f})'. \
                format(box, center[0], center[1])))
        print('Error array ' + ('not ' if error is None else ' ') + 'provided')
        print('Mask array ' + ('not ' if mask is None else ' ') + 'provided')
    i = 0
    delta_center = np.array([1e5,1e5])
    b2 = box/2
    while (i < maxiter) and (delta_center.max() > tol):
        if verbose:
            print(('  iteration {}, center = ({:.4f}, {:.4f})'.format(i,
                    center[0], center[1])))
        p1 = np.floor(center - b2).astype('int')
        p2 = np.ceil(center + b2).astype('int')
        subim = np.asarray(im[p1[0]:p2[0], p1[1]:p2[1]])
        suberr = None if error is None else \
                 np.asarray(error[p1[0]:p2[0], p1[1]:p2[1]])
        submask = None if mask is None else \
                 np.asarray(mask[p1[0]:p2[0], p1[1]:p2[1]])
        if method in [0, '2dg', 'gaussian']:
            xc, yc = centroid_2dg(subim, error=suberr, mask=submask)
        elif method in [1, 'com']:
            xc, yc = centroid_com(subim, mask=submask)
        elif method in [2, 'geom', 'geometric']:
            xc, yc = geometric_center(subim, threshold, mask=submask)
        center1 = np.asarray([yc + p1[0], xc + p1[1]])
        delta_center = abs(center1 - center)
        center = center1
        i += 1

    if verbose:
        print('centroid = ({:.4f}, {:.4f})'.format(center[0], center[1]))
    return center
