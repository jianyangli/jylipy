# script to stack CSS images
#

import jylipy as jp
import numpy as np

datadir = '/Volumes/LaCie/work/Comet_2013A1/LCOGT/Data/'


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
        fn = join(datadir, 'Archive/20170326', filename)
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
    ys, xs = readfits(im)[0].data.shape
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


groups = [['2014-09-30', 'R', 4],
             ['2014-09-30', 'V', 1],
             ['2014-09-30', 'I', 2],
             ['2014-09-30', 'R', 2],
             ['2014-09-30', 'V', 2],
             ['2014-09-30', 'B', 2],
             ['2014-10-01', 'I', 2],
             ['2014-10-01', 'R', 2],
             ['2014-10-01', 'V', 2],
             ['2014-10-10', 'B', 2],
             ['2014-10-09', 'R', 3],
             ['2014-10-09', 'I', 3],
             ['2014-10-11', 'R', 3],
             ['2014-10-11', 'I', 4],
             ['2014-10-12', 'R', 3],
             ['2014-10-12', 'I', 4],
             ['2014-10-12', 'R', 3],
             ['2014-10-12', 'I', 4],
             ['2014-10-13', 'R', 3],
             ['2014-10-13', 'I', 4],
             ['2014-10-14', 'R', 3],
             ['2014-10-14', 'I', 4],
             ['2014-10-14', 'R', 3],
             ['2014-10-15', 'R', 3],
             ['2014-10-15', 'I', 4],
             ['2014-10-16', 'R', 3],
             ['2014-10-16', 'I', 4],
             ['2014-10-17', 'R', 3],
             ['2014-10-17', 'I', 4],
             ['2014-10-17', 'R', 3],
             ['2014-10-17', 'I', 4],
             ['2014-10-20', 'R', 3],
             ['2014-10-20', 'I', 4],
             ['2014-10-21', 'R', 3],
             ['2014-10-21', 'I', 4],
             ['2014-10-22', 'R', 3],
             ['2014-10-22', 'I', 4],
             ['2014-10-23', 'R', 3],
             ['2014-10-23', 'I', 4],
             ['2014-10-24', 'R', 3],
             ['2014-10-24', 'I', 4],
             ['2015-03-06', 'V', 2],
             ['2015-03-06', 'B', 3],
             ['2015-03-06', 'R', 3],
             ['2015-03-06', 'I', 1],
             ['2015-03-07', 'V', 2],
             ['2015-03-07', 'B', 3],
             ['2015-03-07', 'R', 3],
             ['2015-03-07', 'I', 2],
             ['2015-03-10', 'V', 2],
             ['2015-03-10', 'B', 3],
             ['2015-03-10', 'R', 3],
             ['2015-03-10', 'I', 3],
             ['2015-03-11', 'B', 3],
             ['2015-03-11', 'I', 3],
             ['2015-03-11', 'R', 3],
             ['2015-03-11', 'V', 3],
             ['2015-03-12', 'B', 3],
             ['2015-03-12', 'I', 3],
             ['2015-03-12', 'R', 3],
             ['2015-03-12', 'V', 3],
             ['2015-03-14', 'B', 3],
             ['2015-03-14', 'I', 3],
             ['2015-03-14', 'R', 3],
             ['2015-03-14', 'V', 3],
             ['2015-03-23', 'B', 3],
             ['2015-03-23', 'I', 3],
             ['2015-03-23', 'R', 3],
             ['2015-03-23', 'V', 3],
             ['2015-03-23', 'B', 3],
             ['2015-03-23', 'R', 3],
             ['2015-03-24', 'B', 3],
             ['2015-03-24', 'I', 3],
             ['2015-03-24', 'R', 3],
             ['2015-03-24', 'V', 2],
             ['2015-03-24', 'B', 3],
             ['2015-03-24', 'R', 3],
             ['2015-03-26', 'B', 3],
             ['2015-03-26', 'R', 3],
             ['2015-03-26', 'V', 3],
             ['2015-03-26', 'I', 3],
             ['2015-03-26', 'B', 3],
             ['2015-03-26', 'I', 3],
             ['2015-03-26', 'R', 3],
             ['2015-03-26', 'V', 3],
             ['2015-03-27', 'B', 3],
             ['2015-03-27', 'I', 3],
             ['2015-03-27', 'R', 3],
             ['2015-03-27', 'V', 6],
             ['2015-03-27', 'I', 3],
             ['2015-03-28', 'V', 3],
             ['2015-03-28', 'I', 3],
             ['2015-03-28', 'B', 3],
             ['2015-03-28', 'I', 3],
             ['2015-03-28', 'R', 3],
             ['2015-03-28', 'V', 3],
             ['2015-03-28', 'B', 3],
             ['2015-03-28', 'R', 3],
             ['2015-03-29', 'B', 3],
             ['2015-03-29', 'I', 3],
             ['2015-03-29', 'R', 3],
             ['2015-03-29', 'V', 3],
             ['2015-03-29', 'B', 3],
             ['2015-03-29', 'R', 3],
             ['2015-03-29', 'V', 3],
             ['2015-03-29', 'I', 3]]


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

