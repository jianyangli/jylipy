"""
Image processing and analysis related classes and functions
"""

__all__ = ['Centroid']

import numpy as np
from astropy.io import fits, ascii
from astropy import nddata, table
from photutils.centroids import centroid_2dg, centroid_com
from .saoimage import getds9

class Centroid():
    """Image centroiding

    Attributes
    ----------
    .image : 3D or higher dimension array of shape (..., N, M), images
        to be centroided.  The shape of each image is (N, M).
    .file : string array, FITS file names of images to be centroided
    .ext : string, number, or array of them
        The FITS extensions of images in the source files.  If array,
        it has the same shape as `.file`.
    .center : float array of shape (..., 2), centers, the shape of
        (...) is the same as `.file` or `.image[...]`.
    .status : bool array
        Centroiding status.  Same shape as `.file` or `.image[...]`.
    """

    def __init__(self, im, ext=0, box=5):
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
        box : int or sequence of int, optional
            Box size for centroid refining.  See `mskpy.gcentroid`.  If
            sequence, it must have the same shape as input file names
            or images.
        """
        # process input images/files
        im = np.array(im)
        if im.dtype.kind in ['S', 'U']:
            # image file name
            if im.ndim == 0:
                self.file = np.array([im])
            else:
                self.file = im
            self._file1d = self.file.reshape(-1)
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
            self._image1d = self.image.reshape(-1)
            self.file = None
            self._shape = self.image.shape
            self._size = self.image.size
        else:
            raise ValueError('unrecognized input type')
        # FITS extension
        self.ext = ext
        if (not isinstance(self.ext, str)) and \
                hasattr(self.ext, '__iter__') and \
                (np.array(self.ext).shape != self._shape):
            raise ValueError('invalide extensions from `ext`')
        if isinstance(self.ext, str) or \
                (not hasattr(self.ext, '__iter__')):
            self._ext = np.array([self.ext] * self._size)
        else:
            self._ext = self.ext.reshape(-1)
        # box size
        self.box = box
        if hasattr(self.box, '__iter__') and \
                (np.array(self.box).shape != self._shape):
            raise ValueError('invalide shape for `box`')
        if hasattr(self.box, '__iter__'):
            self._box = self.box.reshape(-1)
        else:
            self._box = np.array([self.box] * self._size)
        # preset other attributes
        self.center = np.zeros(self._shape + (2,))
        self.status = np.zeros(self._shape, dtype=bool)

    def _load_image(self, i):
        """Load the ith image in flattened file name list
        """
        if self.file is None:
            raise ValueError('no input file specified')
        if self.image is None:
            self.image = np.zeros_like(self.file, dtype='O')
            self.image[:] = None
            self._image1d = self.image.reshape(-1)
        self._image1d[i] = fits.open(self._file1d[i])[self._ext[i]].data

    def centroiding(self, ds9=None, newframe=False, refine=True, box=5,
                    verbose=True):
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
        _center = self.center.reshape(-1, 2)
        _status = self.status.reshape(-1)
        while i < self._size:
            if (self.image is None) or (self._image1d[i] is None):
                if verbose:
                    print('Image {} in the list: {}.'.format(i,
                            self._file1d[i]))
                self._load_image(i)
            else:
                if verbose:
                    print('Image {} in the list.'.format(i))
            if not retry:
                ds9.imdisp(self._image1d[i], newframe=newframe,
                        verbose=verbose)
            else:
                if verbose:
                    print('Retry clicking near the center.')
            retry = False
            ct = ds9.get('imexam coordinate image').split()
            if len(ct) != 0:
                ct = [float(ct[i]) for i in [1,0]]
                if refine:
                    _center[i] = centroid(self._image1d[i], ct,
                            box=self._box[i], verbose=verbose)
                else:
                    _center[i] = ct
                if verbose:
                    print('Centroid at (x, y) = ({:.4f}, {:.4f})'.format(
                            _center[i][0], _center[i][1]))
                _status[i] = True
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
            # 1D indices
            _index = np.r_[index]
        # _center, _status, _image1d, _box
        _status = self.status.reshape(-1)
        _center = self.center.reshape(-1, 2)
        n = 1
        nn = len(_index)
        for i in _index:
            if verbose:
                if self.file is None:
                    print('Image {} of {}:'.format(n, nn))
                else:
                    print('Image {} of {}: {}'.format(n, nn,
                            self._file1d[i]))
            if not _status[i]:
                if verbose:
                    print('...centroid not measured, skipped\n')
                continue  # skip if hasn't measured
            if (self.image is None) or (self._image1d[i] is None):
                self._load_image(i)
            _center[i] = centroid(self._image1d[i], _center[i],
                    box=self._box[i], verbose=verbose)
            n += 1
            if verbose:
                print()

    def write(self, outfile, format='ascii', **kwargs):
        """Write centers to output file

        Parameters
        ----------
        outfile : str
            Output file name
        format : ['ascii', 'fits'], optional
            Format of output file.
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
        **kwargs : dict, optional
            Other keyword arguments accepted by the `astropy.io.ascii.write`
            or `astropy.io.fits.HDUList.writeto`.
        """
        if format == 'ascii':
            outdata = np.ma.array(self.center,
                    mask=np.repeat(self.status[:,np.newaxis], 2, axis=1))
            yc = outdata[..., 0].flatten()
            xc = outdata[..., 1].flatten()
            box = self._box
            status = self.status.flatten()
            if self.file is None:
                out = table.Table([xc, yc, box, status.astype(int)],
                            names=['xc', 'yc', 'box', 'status'])
            else:
                file = self._file1d
                ext = self._ext
                out = table.Table([self._file1d, self._ext, xc, yc, box,
                            status.astype(int)],
                            names=['file', 'ext', 'xc', 'yc', 'box', 'status'])
            out.write(outfile, **kwargs)
        elif format == 'fits':
            out = fits.HDUList(fits.PrimaryHDU(self.center))
            out.append(fits.ImageHDU(self.status.astype(int), name='status'))
            if self.file is not None:
                files = fits.BinTableHDU(table.Table([self._file1d, self._ext,
                        self._box], names=['file', 'ext', 'box']), name='file')
                files.header['ndim'] = self.file.ndim
                for i in range(self.file.ndim):
                    files.header['axis{}'.format(i)] = self.file.shape[i]
                out.append(files)
            out.writeto(outfile, **kwargs)
        else:
            raise ValueError('unrecognized output format')

    def read(self, infile, format='ascii', **kwargs):
        """Read centers from input file

        Parameters
        ----------
        infile : str
            Input file name
        format : ['ascii', 'fits'], optional
            Format of input file.
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
        if format == 'ascii':
            intable = ascii.read(infile)
            if len(intable.keys()) > 3:
                self.file = np.array(intable['file'])
                self.ext = np.array(intable['ext'])
                self._ext = self.ext.reshape(-1)
                self._box = np.array(intable['box'])
                self.box = self._box[0] if np.all(self._box == self._box[0]) \
                        else self._box
                self._shape = self.file.shape
                self._size = self.file.size
                self.image = np.zeros(self._size, dtype='O')
                self.image[:] = None
                self._file1d = self.file.reshape(-1)
                self._image1d = self.image.reshape(-1)
            if self._size != len(intable['xc']):
                raise ValueError("input file doesn't match the number "
                        "of images")
            self.center = np.array([intable['yc'], intable['xc']]).T. \
                    reshape(self._shape+(2,))
            self.status = np.array(intable['status']).astype(bool). \
                    reshape(self._shape)
        elif format == 'fits':
            with fits.open(infile) as _f:
                if len(_f) == 3:
                    ndim = _f['file'].header['ndim']
                    shape = ()
                    for i in range(ndim):
                        shape = shape + \
                                (_f['file'].header['axis{}'.format(i)],)
                    self.file = _f['file'].data['file'].reshape(shape)
                    self._file1d = self.file.reshape(-1)
                    self.ext = _f['file'].data['ext'].reshape(shape)
                    self._ext = self.ext.reshape(-1)
                    self._box = _f['file'].data['box']
                    self.box = self._box[0] \
                            if np.all(self._box == self._box[0]) else \
                            self._box.reshape(shape)
                    self._shape = shape
                    self._size = self.file.size
                    self.image = np.zeros(self._shape, dtype='O')
                    self.image[:] = None
                    self._image1d = self.image.reshape(-1)
                if _f[0].data.shape[:-1] != self._shape:
                    raise ValueError("input file doesn't match the "
                            "number/shape of images")
                self.center = _f[0].data
                self.status = _f['status'].data.astype(bool)
        else:
            raise ValueError('unrecognized input format')

    @classmethod
    def from_fits(cls, infile):
        obj = cls('')
        obj.read(infile, format='fits')
        return obj


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
