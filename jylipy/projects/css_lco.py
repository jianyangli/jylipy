# CSS utility library
#

import jylipy as jp
import numpy as np
from os.path import join, isfile
import os
from astropy.coordinates import SkyCoord
from astropy.io import fits
import calviacat as cvc

datadir = '/Volumes/Pegasus/Work/Comet_2013A1/LCOGT/Data/'


def load_image(filename,datadir=datadir,header=False,original=False):
    '''Load WCS updated image in `datadir`+'WCS_Updated/' first, if not
    available, then load original data from `datadir`+'Archive/20170326/'
    in either `.fits` or `.fz`.
    '''
    from os.path import isfile, join
    if not original:
        fn = join(datadir, 'WCS_Updated', filename)
    else:
        fn = ''
    if not isfile(fn):
        fn = join(datadir, 'Archive', filename)
        print(fn)
    if not isfile(fn):
        fn = fn+'.fz'
    if not isfile(fn):
        raise IOError('file not found')
    if header:
        func = jp.headfits
    else:
        func = jp.readfits
    ext = jp.condition(fn.endswith('.fz'),1,0)
    return func(fn,verbose=False,ext=ext)


def load_wcs(filename, **kwargs):
    '''Return WCS of specific file'''
    from astropy.wcs import WCS
    dummy = kwargs.pop('header',None)
    hdr = load_image(filename, header=True, **kwargs)
    return WCS(hdr)


def adjust_orientation(filename, center=None, outdir=datadir+'Oriented/'):
    '''Adjust the orientation of images so it is north up and east to the left.
    The adjustment is based on the WCS keywords.  If the WCS is updated with
    astrometry.net, then the updated WCS is used.  Otherwise the WCS in the
    original image is used.
    '''
    from os.path import join, basename
    w = load_wcs(filename)
    im = load_image(filename).astype('float32')
    hdr = load_image(filename, header=True)
    if w.wcs.cd[0,0] > 0:
        im = im[:,::-1]
        hdr['CD1_1'] = -hdr['CD1_1']
        if center is not None:
            center[1] = im.shape[1]-center[1]-1
    if w.wcs.cd[1,1] < 0:
        im = im[::-1,:]
        hdr['CD2_2'] = -hdr['CD2_2']
        if center is not None:
            center[0] = im.shape[0]-center[0]-1
    jp.writefits(join(outdir,basename(filename)),im,hdr,overwrite=True)


def background(im):
    ys, xs = im.shape
    b1 = jp.background(im,region=[100,100,300,300],method='median')
    b2 = jp.background(im,region=[100,xs-300,300,xs-100],method='median')
    b3 = jp.background(im,region=[ys-300,100,ys-100,300],method='median')
    b4 = jp.background(im,region=[ys-300,xs-300,ys-100,xs-100],method='median')
    bg = np.mean([b1,b2,b3,b4])
    return bg


def save_image(outfile, imgtbl, subim, enhd, enhs):
    '''Save images to output file'''
    from jylipy.apext import fits
    from astropy.wcs import WCS
    hdrs = [jp.headfits(datadir+'Oriented/'+r['FileName'],verbose=False) for r in imgtbl]
    hdr = fits.Header()
    keys1 = ['siteid','site','telid','telescop','date-obs','day-obs','filter','instrume','instype']
    keys2 = ['utstart','utstop','exptime','moonfrac','moondist','moonalt','pixscale',]
    keys2a = ['utstart','utstop','exptime','mnfrac','mndist','mnalt']
    for i,r in enumerate(imgtbl):
        hdr['file'+str(i)] = r['FileName']
    for k in keys1:
        hdr[k] = hdrs[0][k]
    for k1,k2 in zip(keys2,keys2a):
        val = []
        for i,h in enumerate(hdrs):
            hdr[k2+str(i)] = h[k1]
            val.append(h[k1])
        if k1 == 'utstart':
            hdr[k2] = min(val)
        elif k1 == 'utstop':
            hdr[k2] = max(val)
        elif k1 == 'exptime':
            hdr[k2] = sum(val)
        else:
            hdr[k2] = np.mean(val)
    ts = jp.Time([hdr['date-obs'].split('T')[0]+'T'+x for x in [hdr['utstart'],hdr['utstop']]])
    hdr['duration'] = (ts[1]-ts[0]).sec
    hdr['utmid'] = (ts[0]+(ts[1]-ts[0])/2).isot.split('T')[1]
    #w = WCS(load_image(imgtbl['FileName'][0],header=True))
    #hdr.extend(w.to_header())
    jp.writefits(outfile,subim,hdr,overwrite=True)
    jp.writefits(outfile,enhd,name='div_1ovr',append=True)
    jp.writefits(outfile,enhs,name='sub_1ovr',append=True)


def show_image(aspect, ds9=None):
    if ds9 is None:
        ds9 = getds9('css')
    for r in aspect:
        im = jp.readfits(datadir+'Oriented/'+r['FileName'],verbose=False)
        ds9.imdisp(im)
        sz = np.array(im.shape)/2
        if isinstance(r['xc_2'], np.number) and isinstance(r['yc_2'], np.number):
            c = jp.CircularRegion(r['xc_2'],r['yc_2'],10)
            c.show(ds9)
        ds9.set('regions','image; text {0} {1} # text='.format(sz[1]-80,sz[0]-150)+'{'+r['FileName']+'}')
        ds9.set('regions','image; text {0} {1} # text='.format(sz[1]+200,sz[0]-150)+'{'+r['Filter']+'}')


def stack_group(group,datadir=datadir+'Oriented/'):
    from os.path import join
    ims = [jp.readfits(join(datadir,filename),verbose=False) for filename in group['FileName']]
    yref,xref = [int(np.round(x)) for x in group[('yc_2','xc_2')][0]]
    ims = [jp.shift(im,(yref-yc,xref-xc)) for im,(yc,xc) in zip(ims,group[('yc_2','xc_2')])]
    ims = np.array(ims)
    return np.median(ims,axis=0), (yref, xref)


def show_stacked(date, filt, ext=0, ds9=None):
    '''Show the stacked images in DS9
    '''
    from astropy.io import ascii
    aspect = jp.ascii_read(datadir+'Stacked/aspect.csv')
    bdr = {'B':2,'V':3,'R':19,'I':17}
    if date not in ['201410','201503']:
        print('date must be in ["201410", "201503"]')
        return
    if filt not in ['B','V','R','I']:
        print('filter must be in ["B","V","R","I"]')
        return
    if ext not in [0,1,2,3]:
        print('ext must be 0 - stacked image; 1 - 1/rho divided; 2 - 1/rho subtracted')
        return
    if ds9 is None:
        ds9 = jp.getds9('css')
    if date == '201410':
        for f in aspect.query('filter',filt)[:bdr[filt]]:
            ds9.imdisp(datadir+'Stacked/'+f['filename'],ext=ext)
    else:
        for f in aspect.query('filter',filt)[bdr[filt]:]:
            ds9.imdisp(datadir+'Stacked/'+f['filename'],ext=ext)


class PhotCal():
    """Photometric calibration for LCO images

    Photometrically calibrate the magnitude zero points using the '_cat'
    generated by LCO pipeline with the `calviacat` package.
    """
    _filter_mapping = {'skymapper': {'u': ['U'], 'v': [], 'r': ['R'],
                                     'i': ['I'], 'g': ['g'],
                                     'z': ['z']},
                       'ps1': {'g': ['V', 'gp'], 'r': ['R', 'rp'],
                               'i': ['I', 'ip'], 'z': ['z', 'zp'],
                               'y': ['y']},
                       'atlas': {'g': ['V', 'gp'], 'r': ['R', 'rp'],
                                 'i': ['I', 'ip'], 'z': ['z', 'zp']}}

    @property
    def cal_filter(self):
        for f in self._filter_mapping[self.catalog_name]:
            if self.filter in self._filter_mapping[self.catalog_name][f]:
                return f
        return None

    def __init__(self, imgname, catalog,
                 path=join(os.sep, 'Users', 'jyli', 'Work', 'Comet_2013A1',
                                                'LCOGT', 'Data', 'Archive')):
        """
        imgname - str
            The name of LCO image to be calibrated.  The image name has a
            general format of (site)(tel)-(instr)-YYYYMMDD-(frame).
        catalog : ['ps1', 'atlas', 'skymapper']
            catalog to be used for calibration
        path - str
            Path to data file
        """
        self.file = imgname
        self.path = path
        catfile = join(self.path, self.file) + '-e91.fits'
        pipeline = 'BANZAI'
        if not isfile(catfile):
            catfile = catfile+'.fz'
        if not isfile(catfile):
            catfile = join(self.path, self.file) + '-e90_cat.fits'
            pipeline = 'ORAC'
        if not isfile(catfile):
            catfile = catfile+'.fz'
        if not isfile(catfile):
            print(catfile)
            raise IOError("cat file not found")
        self._catfile = catfile
        self._pipeline = pipeline
        if pipeline == 'BANZAI':
            self.filter = fits.open(self._catfile)['sci'].header['filter']
        else:
            self.filter = fits.open(self._catfile)[0].header['filter']
        self.catalog_name = catalog
        if catalog == 'ps1':
            self.catalog = cvc.PanSTARRS1
        elif catalog == 'atlas':
            self.catalog = cvc.RefCat2
            from os import environ
            environ['CASJOBS_USERID'] = 'jyli'
            environ['CASJBOS_PW'] = 'Charac+ers12'
        elif catalog == 'skymapper':
            self.catalog = cvc.SkyMapper
        self._fetch_database = True

    def __call__(self):
        if self._pipeline == 'BANZAI':
            with fits.open(self._catfile) as hdu:
                phot = jp.Table(hdu['cat'].data)
                phot = phot[phot['FLAG'] == 0]
                lco = SkyCoord(phot['RA'], phot['DEC'], unit='deg')
                m_inst = -2.5 * np.log10(phot['FLUX'])
                m_err = phot['FLUXERR'] / phot['FLUX'] * 1.0857
        else:
            with fits.open(self._catfile) as hdu:
                phot = jp.Table(hdu[1].data)
                phot = phot[phot['FLAGS'] == 0]
                lco = SkyCoord(phot['ALPHA_J2000'], phot['DELTA_J2000'],
                               unit='deg')
                m_inst = -2.5 * np.log10(phot['FLUX_AUTO'])
                m_err = phot['FLUXERR_AUTO'] / phot['FLUX_AUTO'] * 1.0857

        cat = self.catalog(self.catalog_name+'_cat.db')
        if self._fetch_database:
            if len(cat.search(lco)[0]) < 500:
                cat.fetch_field(lco)
        self.zp_mean = 0.
        self.zp_median = 0.
        self.zp_uncertainty = 0.
        self.cal_error = ''
        self.cal_success = False
        try:
            match = cat.xmatch(lco)
            if match is None:
                self.cal_success = False
                self.cal_error = 'No matching stars'
            else:
                objids, distances = match
                zp_mean, zp_median, unc, m, gmi = cat.cal_constant(
                    objids, m_inst, self.cal_filter)
                self.cal_success = True
                self.zp_mean = zp_mean
                self.zp_median = zp_median
                self.zp_uncertainty = unc
        except Exception as e:
            self.cal_error = str(e)
