"""Toolkit to support DDAP Ceres local spectrophotometry project"""

from glob import glob
from warnings import warn
import os, numpy as np, matplotlib.pyplot as plt, astropy.units as u
from astropy.time import Time
from astropy.io import ascii
from pysis import CubeFile
import spiceypy as spice

from ..core import rebin
from ..geometry import load_generic_kernels
from ..Photometry import PhotometricData
from ..projects.dawn import FCImage


class RegionalData:
    """Regional data object
    """

    def __init__(self, cubedir='.', maskdir='.', data_catalog=None,
            force_catalog_data=False, spatial_bin=None,
            roi_tags=[66, 185, 0, 255], mask_sfx='mask',
            outfile='{}_roi{}.fits'):
        """
        Parameters
        ----------
        cubedir : str
            Directory of input backplane data in ISIS cubes
        maskdir : str
            Directory of ROI masks
        data_catalog : str, optional
            Catalog file that provides the path name of image data file.
            Default is to take the image data from ISIS cubes.  If this
            parameter is provided (not None), then the image data will
            be taken from a separate file pointed by the catalog file.
            Catalog file must have columns 'ID', 'L1a', 'L1b', 'L1c',
            which stores the exposure ID, paths to level 1a, 1b, and l1c
            image files, all in strings.  The paths are assumed to be
            relative to the path of the catalog file.
        force_catalog_data : bool, optional
            By default, if no separate data found in `data_catalog`,
            then the embedded data in the ISIS cube will be used.  If
            set `True`, then the corresponding image will be dropped
            in this case.
        spatial_bin : num, optional
            Factor for spatial binning before extracting data.  This
            factor should be power of 2.  If not, then it will be changed
            to the closest power of 2, and a warning is issued.
        roi_tags : list of int, optional
            Values of ROI tags
        mask_sfx : str, optional
            Suffix to be added to data file names to make them mask file names
        outfile : str, optional
            Root name of output data files.  It has two {} to be filled in by
            filter name and ROI number.  ROI number starts from 1 in stead of 0.
        """
        self.cubedir = cubedir
        self.maskdir = maskdir
        self.data_catalog = data_catalog
        self.catalog = None if self.data_catalog is None \
                else ascii.read(self.data_catalog)
        self.force_catalog_data = force_catalog_data
        # check and process `spatial_bin` to make sure it's the power of 2
        spatial_bin_ = 2**int(np.log2(spatial_bin))
        if spatial_bin_ != spatial_bin:
            spatial_bin = spatial_bin_
            warn('spatial_bin of {} is not a power of 2, changed to {}'.
                format(spatial_bin, spatial_bin_))
        self.spatial_bin = spatial_bin
        self.mask_sfx = mask_sfx
        self.roi_tags = roi_tags
        self.outfile = outfile
        # calibration constants to convert from flux to reflectance
        self.iofcal = {'F1': 1.365,
                       'F2': 1.863,
                       'F3': 1.274,
                       'F4': 0.865,
                       'F5': 0.785,
                       'F6': 1.058,
                       'F7': 1.572,
                       'F8': 1.743}
        # Ceres SPK file
        self.ceres_spk = os.path.join(os.path.sep, 'Users', 'jyli',
                'Work', 'Ceres', 'spice', 'ceres_1900-2100_20151014.bsp')

    @property
    def n_roi(self):
        return len(self.roi_tags)

    def _output_filename(self, filter, roi, ext='fits', binned=False):
        base = self.outfile.format(filter, roi)
        name, _ = os.path.splitext(base)
        if binned:
            name = name + '_binned'
        return '.'.join([name, ext])

    def _search_catalog(self, f):
        """Search catalog to find calibrated image from input image name.

        If Level 1c calibration exists, then return it.  Otherwise return
        Level 1b calibration.
        """
        if self.data_catalog is None:
            raise ValueError('catalog file is not specified.')
        data_path = os.path.split(self.data_catalog)[0]
        tmp = os.path.basename(f).split('_')[0]
        ww = tmp.find('1B') + 2
        img_id = int(tmp[ww:])
        row = self.catalog[self.catalog['ID'] == img_id]
        if not row['L1c'].mask:
            img_file = os.path.join(data_path, row['L1c'][0])
        elif not row['L1b'].mask:
            img_file = os.path.join(data_path, row['L1b'][0])
        else:
            img_file = ''
        return img_file

    @property
    def cubefiles(self):
        return np.array(glob(os.path.join(self.cubedir, '*.cub')))

    def phodata_extract(self, overwrite=False):
        """Extract photometric data"""

        print('Extract photometric data')
        print('    ISIS cube directory: {}'.format(self.cubedir))
        print('    ROI mask directory: {}, suffix: {}'.format(self.maskdir,
                    self.mask_sfx))
        print('    ROI tags: ', self.roi_tags)
        if self.data_catalog is None:
            print('    Data imbedded in ISIS cube')
        else:
            print('    Data catalog file: {}'.format(self.data_catalog))
            print('        Catalog data use forced: {}'.format(
                    self.force_catalog_data))
        if self.spatial_bin is not None:
            print('    Spatial binning: {}'.format(self.spatial_bin))
        else:
            print('    No spatial binning.')
        print('    Output file name template: {}'.format(self.outfile))
        print()

        files = self.cubefiles
        get_filter = lambda f: os.path.splitext(os.path.basename(f))[0][-3:-1]
        filters = np.array([get_filter(f) for f in files])
        self.filter_list = np.unique(filters)

        load_generic_kernels()
        spice.furnsh(self.ceres_spk)

        # loop through filters
        for flt in self.filter_list:
            print(' '*80, end='\r')
            print('filter : {}'.format(flt), end='')
            ff = files[filters == flt]  # files of filter 'flt'
            ff = [x for x in ff if x.find('FC1') == -1] # filter out FC1
            print(', {} files'.format(len(ff)))
            iof = [[] for i in range(self.n_roi)]
            pha = [[] for i in range(self.n_roi)]
            emi = [[] for i in range(self.n_roi)]
            inc = [[] for i in range(self.n_roi)]
            lat = [[] for i in range(self.n_roi)]
            lon = [[] for i in range(self.n_roi)]
            pxl = [[] for i in range(self.n_roi)]
            # loop through files
            for j, f in enumerate(ff):
                print('    {}: {}'.format(j+1, f), end='\r')
                f_base = os.path.basename(f)
                # load backplane data
                datacube = CubeFile(f)
                data = datacube.apply_numpy_specials()
                im = data[0]
                # load image data if needed
                if self.catalog is not None:
                    img_file = self._search_catalog(f)
                    if img_file:
                        im = FCImage(img_file, quickload=True)
                    else:
                        if self.force_catalog_data:
                            warn('Image {} not found, skipped'.format(f_base))
                            continue
                        else:
                            warn('Image {} not found, embedded cube data used'.
                                    format(f_base))
                # calibrate to i/f
                if 'Instrument' not in datacube.label['IsisCube']:
                    utc = f_base[15:26]
                    utc = '20'+utc[:2] + '-' + utc[2:5] + 'T' + \
                            utc[5:7] + ':' + utc[7:9] + ':' + utc[9:]
                    utc = Time(utc).isot
                else:
                    utc = Time(datacube.label['IsisCube']['Instrument']\
                            ['StartTime']).isot
                et = spice.utc2et(utc)
                st, lt = spice.spkezr('sun', et, 'j2000', 'lt+s', 'ceres')
                rh = np.sqrt(st[0]*st[0] + st[1]*st[1] + st[2]*st[2]) * \
                        u.km.to('au')
                im = im * rh * rh * np.pi / self.iofcal[flt]
                # prepare mask
                maskfile = os.path.join(self.maskdir, self.mask_sfx + f_base)
                if not os.path.isfile(maskfile):
                    continue
                mask = CubeFile(maskfile)
                mask = np.squeeze(mask.apply_numpy_specials())
                mask[~np.isfinite(mask)] = -255
                mask = mask.astype('int16')
                # clean up noise values.  usually those are single pixels
                nn = 0
                while (len(np.unique(mask)) > len(self.roi_tags) + 1) \
                        and nn < 3:
                    for v in np.unique(mask):
                        if v not in list(self.roi_tags) + [-255]:
                            ww = np.where(mask == v)
                            ww1_ = ww[1] + 1
                            ww1_bad = np.where(ww1_ >= mask.shape[1])
                            if len(ww1_bad[0]) > 0:
                                ww1_[ww1_bad] = ww[1][ww1_bad] - 1
                            mask[ww] = mask[ww[0], ww1_]
                    nn += 1
                if nn >= 3:
                    warn('mask {} contains many noise values'.format(f_base))
                # spatially bin data and mask
                if (self.spatial_bin is not None) and (self.spatial_bin != 1):
                    im = rebin(im, (self.spatial_bin, self.spatial_bin),
                                mean=True)
                    data = rebin(data, (1, self.spatial_bin, self.spatial_bin),
                                mean=True)
                    mask = rebin_mask(mask,
                            (self.spatial_bin, self.spatial_bin))
                # extract data
                for i, t in enumerate(self.roi_tags):
                    ww = (mask == t)  # pixels within roi mask
                    ww = ww & (data[7] == 1) # illuminated by the Sun
                    for d in data[1:]:  # filter out nan values
                        ww = ww & np.isfinite(d)
                    if ww.any():
                        iof[i].append(im[ww])
                        pha[i].append(data[1][ww])
                        emi[i].append(data[2][ww])
                        inc[i].append(data[3][ww])
                        lat[i].append(data[4][ww])
                        lon[i].append(data[5][ww])
                        pxl[i].append(data[6][ww])
            print(' '*80, end='\r')
            # save data
            print('    saving data', end='\r')
            for i in range(self.n_roi):
                iof_ = np.concatenate(iof[i])
                pha_ = np.concatenate(pha[i])
                emi_ = np.concatenate(emi[i])
                inc_ = np.concatenate(inc[i])
                lat_ = np.concatenate(lat[i])
                lon_ = np.concatenate(lon[i])
                phodata = PhotometricData(iof=iof_, inc=inc_, emi=emi_,
                        pha=pha_, geolat=lat_, geolon=lon_)
                phodata.write(self._output_filename(flt, i+1),
                        overwrite=overwrite)

        spice.kclear()

    def plot_coverage(self, filter, savefig=False):
        """Plot ROI coverage of data"""
        phofiles = [self._output_filename(filter, i+1) for i in
                range(self.n_roi)]
        pho = [PhotometricData(f) for f in phofiles]
        plt.figure(num=plt.gcf().number)
        for p in pho:
            plt.plot(p.geolon, p.geolat, '.', ms=1)
        if savefig:
            plt.savefig(self._output_filename(filter, '', 'png'))

    def plot_phodata(self, filter, roi, savefig=False):
        """Plot photometric data for specified filter and ROI"""
        pho = PhotometricData(self._output_filename(filter, roi))
        pho.plot()
        if savefig:
            plt.savefig(self._output_filename(filter, roi, 'png'))

    def binning(self, bins, filter=None, roi=None, overwrite=False):
        if filter is None:
            filter = self.filter_list
        if isinstance(filter, str) or (not hasattr(filter, '__iter__')):
            filter = [filter]
        if roi is None:
            roi = range(1, self.n_roi+1)
        if not hasattr(roi, '__iter__'):
            roi = [roi]
        for flt in filter:
            print('filter {}:'.format(flt))
            for ii in roi:
                print(' '*80, end='\r')
                print('  roi: {}'.format(ii), end='\r')
                pho = PhotometricData(self._output_filename(flt, ii))
                phob = pho.bin(bins=bins)
                phob.write(self._output_filename(flt, ii, binned=True),
                        overwrite=overwrite)


def rebin_mask(mask, bin):
    """Rebin mask with specified binning factors

    Mask will be binned based on the specified factors.  The pixel value
    in the output mask is the value of the most pixels in each bin,
    rather than the arithmatic calculation of those pixel values.  This
    will avoid introducing new mask values in the binning.

    Parameters
    ----------
    mask : 2D array
        Input mask to be rebinned.
    bin : 2-element sequence of positive int
        Bin size in two dimentions

    Return
    ------
    2D array, binned mask
    """

    sz = np.shape(mask)
    out_sz = np.ceil(np.array(sz) / bin).astype(int)
    out = np.zeros(out_sz)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            y0 = i * bin[0]
            x0 = j * bin[1]
            sub = mask[y0:y0 + bin[0], x0:x0 + bin[1]]
            v, c = np.unique(sub, return_counts=True)
            out[i, j] = v[c.argmax()]

    return out
