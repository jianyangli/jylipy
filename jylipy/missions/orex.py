# OREx software package
#
# 1/5/2017, JYL @PSI

import warnings
import numpy as np, string, spiceypy as spice
from copy import copy
from jylipy.core import *
from jylipy.vector import xyz2sph, vecsep #Image, readfits, condition, xyz2sph, Time, num, findfile, writefits, CCDData, ImageMeasurement, CaseInsensitiveOrderedDict, findfile, ascii_read
from jylipy.apext import Table, Column, units, fits
from jylipy import Photometry, pplot, Table
from jylipy.saoimage import getds9
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from jylipy.Photometry import PhotometricData, PhotometricDataGrid


def spdif_mapping(data, coord, grid, method='nearest', reduce=False):
    '''Project SPDIF data to lon-lat grid

    data : array (n, b), PSDIF data with n spectra of b bands
    lon, lat : Corresponding (longitude, latitude)
    longrid, latgrid : Longitude and latitude grid of output cube
    reduce : Reduce the data by taken the mean of each (lat,lon) grid

    v1.0.0 : Jan 5, 2017, JYL @PSI
    '''
    from scipy.interpolate import griddata
    data = np.asarray(data)
    lon, lat = coord
    longrid, latgrid = grid
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if reduce:
        lon1 = []
        lat1 = []
        data1 = []
        for l1,l2 in zip(longrid,np.concatenate((longrid[1:],[360]))):
            goodlon = (lon >= l1) & (lon < l2)
            for m1,m2 in zip(latgrid[:-1],latgrid[1:]):
                good = goodlon & (lat >= m1) & (lat < m2)
                if good.any():
                    data1.append(data[good].mean(axis=0))
                    lon1.append((l1+l2)/2.)
                    lat1.append((m1+m2)/2.)
        data = np.asarray(data1)
        lon = np.asarray(lon1)
        lat = np.asarray(lat1)

    longrid = np.asarray(longrid)
    latgrid = np.asarray(latgrid)
    pt = np.concatenate([lon[:,np.newaxis], lat[:,np.newaxis]], axis=1)
    nbd = data.shape[1]
    grid = np.ones(latgrid.shape+longrid.shape)
    longd = longrid*grid
    latgd = (latgrid*grid.T).T
    out = np.zeros((nbd,)+latgrid.shape+longrid.shape)
    for b in range(nbd):
        out[b] = griddata(pt, data[:,b],(longd, latgd), method=method)
    return out


def load_phodata(datafile, band, iofmin=0, iofmax=1, incmin=0, incmax=90, emimin=0, emimax=90, phamin=0, phamax=180, norejection=False):
    '''Load photometric data from PDIF data file'''
    data = fits.open(datafile)
    iof = data[0].data[:,band]
    inc = data[3].data['INCIDANG']
    emi = data[3].data['EMISSANG']
    pha = data[3].data['PHASEANG']
    asp = Table(data[3].data)
    ioferr = data[1].data[:,band]
    if not norejection:
        good = (iof>=iofmin)& (iof<=iofmax )& (inc>=incmin)& (inc<=incmax) & (emi>=emimin)& (emi<=emimax) & (pha>=phamin)& (pha<=phamax)
        iof = iof[good]
        inc = inc[good]
        emi = emi[good]
        pha = pha[good]
        asp = asp[good]
        ioferr = ioferr[good]
    pho = Photometry.PhotometricData(iof=iof,inc=inc,emi=emi,pha=pha)
    return pho, asp, ioferr


def pick_spectrum(datafile, coord, radius=2., unit='radf'):
    '''Pick a spectrum within a circular footprint of radius `radius` and
    centered at (lat,lon)'''
    data = fits.open(datafile)
    lats = data[3].data['LAT']
    lons = data[3].data['LON']
    lat,lon = coord
    dist = vecsep([lon,lat], [lons,lats])
    within = (dist < radius) & (np.array([data[3].data['INCIDANG'],data[3].data['EMISSANG'],data[3].data['PHASEANG']]).max(axis=0) > 0)
    if within.any():
        s = data[0].data[within].mean(axis=0)
        return s
    else:
        return -np.ones_like(data[0].data[0])


def read_phomodel(phofile):
    f = fits.open(phofile)
    nm = len(f)
    pars = {}
    for i in range(1,nm,2):
        d = f[i].data
        p = {}
        p['wavelength'] = d[0]
        p['chisq'] = d[1]
        p['par'] = d[2:].T
        pars[f[i].header['extname']] = p
    return pars


def plot_model_quality(datafile):
    pdf = PdfPages(datafile.replace('.fits','.pdf'))
    q = fits.open(datafile)
    nm = len(q)-1
    mnames = [q[i].header['extname'] for i in range(1,1+nm)]
    plt.clf()
    f, ax = plt.subplots(nm, 1, sharex=True, num=plt.gcf().number)
    lbl = ['SLOPE','SLOPE_PHA','SLOPE_INC','SLOPE_EMI']
    for j in range(len(lbl)):
        for i in range(nm):
            good = np.array(q[i+1].data).view(('>f4',9))[:,1:].sum(axis=1) != 0
            ax[j].plot(q[i+1].data['WAV'][good], q[i+1].data[lbl[j]][good],'.')
        pplot(ax[j],ylabel=lbl[j])
    pplot(ax[3],xlabel='Wavelength ($\mu$m)')
    ax[0].legend(mnames,loc='lower center',ncol=nm//2)
    pdf.savefig()
    plt.clf()
    f, ax = plt.subplots(nm, 1, sharex=True, num=plt.gcf().number)
    lbl = ['CORR','CORR_PHA','CORR_INC','CORR_EMI']
    for j in range(len(lbl)):
        for i in range(nm):
            good = np.array(q[i+1].data).view(('>f4',9))[:,1:].sum(axis=1) != 0
            ax[j].plot(q[i+1].data['WAV'][good], q[i+1].data[lbl[j]][good],'.')
        pplot(ax[j],ylabel=lbl[j])
    pplot(ax[3],xlabel='Wavelength ($\mu$m)')
    ax[0].legend(mnames,loc='lower center',ncol=nm//2)
    pdf.savefig()
    pdf.close()


def plot_phomodel(phofile, model=None,filter=True,savefig=False,overplot=False,**kwargs):
    par = read_phomodel(phofile)
    if savefig:
        pdf = PdfPages(phofile.replace('.fits','.pdf'))
    if model is None:
        mds = par.keys()
    else:
        if isinstance(model,str):
            mds = [model]
        elif isinstance(model,list):
            mds = model
    for k in mds:
        npar = par[k]['par'].shape[1]
        if filter:
            flags = par[k]['par'].sum(axis=1) != 0
        else:
            flags = np.ones(par[k]['par'].shape[0],dtype=bool)
        if overplot:
            f = plt.gcf()
            ax = f.axes
        else:
            f,ax = plt.subplots(npar,1,sharex=True,num=plt.gcf().number)
        for i in range(npar):
            ax[i].plot(par[k]['wavelength'][flags],par[k]['par'][flags,i],**kwargs)
            pplot(ax[i])
        pplot(ax[-1],xlabel='Wavelength ($\mu$m)')
        pplot(ax[0],title=k)
        if savefig:
            pdf.savefig()
    if savefig:
        pdf.close()


def load_specdata(datafile):
    '''Load spectral data from PDIF data file'''
    data = fits.open(datafile)
    iof = data[0].data
    asp = Table(data[3].data)
    err = data[1].data
    wav = data[2].data
    return iof, err, wav, asp


def load_phomodels(datafile):
    '''Load photometric model generated by photmods.pro'''
    data = fits.open(datafile)
    models = {}
    for h in data[1::2]:
        models[h.header['EXTNAME']] = {}
        models[h.header['EXTNAME']]['wav'] = h.data[0]
        models[h.header['EXTNAME']]['chisq'] = h.data[1]
        models[h.header['EXTNAME']]['p'] = h.data[2:]
    return models



import urllib.request, urllib.error, urllib.parse, http.cookiejar, sys, json, os, threading, queue
from datetime import datetime

user_jyli = 'jyli'
pwd_jyli = 'VUy-cw6-mDA-VLz'


def session_login(host, username=None, password=None):
    if username is None:
        username = input('Username: ')
    if password is None:
        from getpass import getpass
        password = getpass('Password: ')
    url = 'https://%s.lpl.arizona.edu/session-login' % host
    #Create the data to be sent and set the request header
    params = ('{"username": "%s","password":"%s"}' % (username,password)).encode()
    header = {'Content-Type':'application/json'}
    #Attempt to login:
    request = urllib.request.Request(url,params,header)
    try:
        response = urllib.request.urlopen(request)
        #Save the cookie for later
        cookiejar = http.cookiejar.LWPCookieJar('%s_cookie' % host)
        cookiejar.extract_cookies(response,request)
        cookiejar.save('%s_cookie'%(host))
        print('Successfully logged in')
    except urllib.error.HTTPError as e:
        print('Failed to login.\nReason: %s' % e)
        raise urllib.error.HTTPError


def webapi_post(host, query, endpoint, username=None, password=None):
    url = 'https://%s.lpl.arizona.edu/%s' % (host,endpoint)
    header = {'Content-Type':'application/json'}
    request = urllib.request.Request(url,query.encode(),header)
    if host+'_cookie' not in os.listdir('.'):
        session_login(host, username=username, password=password)
    elif (datetime.now() - datetime.fromtimestamp(os.path.getmtime(host+'_cookie'))).days > 2:
        session_login(host, username=username, password=password)
    cookiejar = http.cookiejar.LWPCookieJar('%s_cookie' % host)
    cookiejar.load('%s_cookie' % host)
    cookiejar.add_cookie_header(request)
    try:
        response = urllib.request.urlopen(request)
        return response.read()
    except urllib.error.HTTPError as e:
        print('Failed to download. \nReason: %s\nQuery: %s' % (e.read(),query))


class DownloadHelper(threading.Thread):
    def __init__(self, download_queue, username=None, password=None):
        threading.Thread.__init__(self)
        self.download_queue = download_queue
        self.username=username
        self.password=password

    def run(self):
        while 1:
            (level,id) = self.download_queue.get()
            if level == 'l0':
                query = '{ "product": "ovirs_sci_frame.fits-l0-spot", "id": "ovirs_sci_frame.%s" }' % str(id)
                spot = webapi_post('spocflight',query,'data-get-product',username=self.username, password=self.password)
                with open('fits_l0_spot/ovirs_sci_frame_%s.fits' % str(id), 'wb') as f:
                    f.write(spot)
                    f.close()
            else:
                query = '{ "product": "ovirs_sci_level2_record.fits-l2-spot", "id": "ovirs_sci_level2_record.%s" }' % str(id)
                spot = webapi_post('spocflight',query,'data-get-product', username=self.username, password=self.password)
                with open('fits_l2_spot/ovirs_sci_level2_record_%s.fits' % str(id), 'wb') as f:
                    f.write(spot)
            self.download_queue.task_done()


def spoc_product_name(inst, level, group, cat='sci'):
    """Return the SPOC product name for the given instrument, level and group

    For OVIRS data:
        'l0_spot'      : 'ovirs_sci_frame'
        'l0_multipart' : 'ovirs_sci_level0'
        'l2_spot'      : 'ovirs_sci_level2_record'
        'l2_multipart' : 'ovirs_sci_level2'
        'l3a_spot'     : 'ovirs_sci_level3a_record'
        'l3a_multipart': 'ovirs_sci_level3a'
        'l3b_spot'     : 'ovirs_sci_level3b_record'
        'l3b_multipart': 'ovirs_sci_level3b'
        'l3e1'         : 'ovirs_sci_level3e_record'
        'l3e2'         : 'ovirs_sci_level3e'
    """
    instruments = ['ovirs']
    categories = ['sci']
    levels = ['l0', 'l2', 'l3a', 'l3b', 'l3c', 'l3e', 'l3f', 'l3g']
    groups = ['spot', 'multipart']

    if inst not in instruments:
        raise ValueError('instrument not recognized in {0}: {1}'.format(instruments, inst))
    if level not in levels:
        raise ValueError('level not recognized in {0}: {1}'.format(levels, level))
    if group not in groups:
        raise ValueError('group not recognized in {0}: {1}'.format(groups, group))
    if cat not in categories:
        raise ValueError('category not recognized in {0}: {1}'.format(categories, cat))

    if group.strip() == 'spot':
        grp_str = 'record'
    else:
        grp_str = ''

    lvl_str = level[0]+'evel'+level[1:]
    table_name = '_'.join([inst, cat, lvl_str, grp_str])

    if level == 'l0' and group == 'spot':
        table_name = '_'.join([inst, cat, 'frame'])
    table_name = '"'+table_name.strip('_')+'"'

    return table_name


class SPOC_Downloader():
    def __init__(self, inst, level, group, cat='sci', host='spocflight', user=user_jyli, passwd=pwd_jyli):
        self.table_name = spoc_product_name(inst, level, group, cat=cat)
        self.host = host
        self.instrument = inst
        self.level = level
        self.group = group
        self.category = cat
        self.username = user
        self.password = passwd

    def query(self):
        query_str = '{ "table_name": '+self.table_name+', "columnsaa": ["*"] }'
        result = webapi_post(self.host, query_str, 'data-get-values', username=self.username, password=self.password)
        if result is None:
            return False
        self.data = json.loads(result)['result']
        return True

    def download(self, block=False, nworker=8):
        download_queue = query.Queue()
        for i in range(nworker):
            helper = DownloadHelper(download_queue, username=self.username, password=self.password)
            helper.daemon = True
            helper.start()
        for d in self.data:
            download_queue.put((self.level, d['id']))
        if block:
            download_queue.join()


class Catalog(Table):
    """Catalog of sources"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file = None
        self.wcs = None

    @classmethod
    def from_file(cls, file):
        """Generate a Catalog class object from input `file`.  `file` has to be
        loadable by `astropy.io.ascii.read` function"""
        table = ascii_read(file)
        inst = cls(table)
        inst.file = file
        return inst

    def write(self, file, **kwargs):
        """Write Catalog class object to `file`"""
        format = kwargs.pop('format', 'ascii.fixed_width_two_line')
        super().write(file, format=format, **kwargs)

    def strip(self, colnames=['XWIN_IMAGE','YWIN_IMAGE','ERRX2WIN_IMAGE','ERRY2WIN_IMAGE','ERRXYWIN_IMAGE','BACKGROUND','FLUX_APER','MAG_APER','FLAGS']):
        """Return a Catalog class object with only columns specified by `colnames`"""
        ks = list(self.keys())
        for c in colnames:
            if c not in ks:
                warnings.warn(f"'{c}' not a valid column name, discarded")
                colnames.remove(c)
        return self[colnames]

    def clean(self, im):
        """Return a cleaned catalog"""

        nd0 = len(self)
        sz = im.shape

        reject = np.zeros(len(self),dtype=bool)
        ctmean = np.zeros_like(reject,dtype=np.float32)
        dist = np.sqrt((self['X_IMAGE']-self['XWIN_IMAGE'])**2+(self['Y_IMAGE']-self['YWIN_IMAGE'])**2)
        for i in range(len(self)):
            if dist[i]>1:
                reject[i] = True
            else:
                x,y = self['XWIN_IMAGE','YWIN_IMAGE'][i]
                xx,yy = int(round(x-1)), int(round(y-1))
                if xx>sz[1] or yy>sz[0] or xx<140 or yy<60:
                    reject[i] = True
                else:
                    ctpix = im[yy,xx]
                    ctim = im[yy-1:yy+2,xx-1:xx+2]
                    ctmean[i] = ctim.mean()
                    if (ctpix < self['BACKGROUND'][i]+np.sqrt(self['BACKGROUND'][i])*2) or ((ctim>4090).any()):
                        reject[i] = True

        good = ~reject
        cleaned = self[good]
        ctmean = ctmean[good]

        flags = np.zeros(len(cleaned),dtype=np.uint16)
        traced = np.sqrt(cleaned['X2WIN_IMAGE'] + cleaned['Y2WIN_IMAGE']) > .8
        flags[traced] = 1
        uncertain = ctmean < cleaned['BACKGROUND']+np.sqrt(cleaned['BACKGROUND']*2)
        flags[uncertain] |= 8
        if 'FLAGS' in cleaned.keys():
            cleaned.rename_column('FLAGS','FLAGS_ORIG')
        cleaned.add_column(Column(flags, name='FLAGS'))
        return cleaned

    def normalize(self, exptime):
        """Normalize any FLUX and MAG column(s) by exposure time"""
        for k in self.keys():
            normed = True
            if k.find('FLUX_')>=0:
                val = self[k]/exptime
            elif k.find('MAG_')>=0:
                val = self[k]+2.5*np.log10(exptime)
            else:
                normed = False
            if normed:
                self.add_column(Column(val, name=k+'_NORM'))

    def show(self, ds9=None, size=5, specs=None, flagging=True):
        """Show objects in the Catalog in DS9"""
        if d is None:
            ds9 = getds9('OREx')
        for o in self:
            r = CircularRegion(o['XWIN_IMAGE']-1,o['YWIN_IMAGE']-1,size)
            if specs is not None:
                for k,v in specs.items():
                    r.specs[k] = v
            if flagging and ('FLAGS' in list(self.keys())):
                if o['FLAGS'] == 1:
                    r.specs['color'] = 'blue'
                elif o['FLAGS'] > 7:
                    r.specs['color'] = 'yellow'
            r.show(ds9)

    def plot(self, x='XWIN_IMAGE', y='YWIN_IMAGE', flagging=True):
        """Plot objects in the Catalog"""
        if flagging and ('FLAGS' in list(self.keys())):
            flags = list(set(self['FLAGS']))
            flags.sort()
            for f in flags:
                subset = self[self['FLAGS'] == f]
                ax = plt.plot(subset[x],subset[y],'o')
        else:
            ax = plt.plot(self[x],self[y],'o')
        pplot(xlabel='X',ylabel='Y')
        plt.legend([str(x) for x in flags])
        return ax

    @property
    def wcs(self):
        return self._wcs
    @wcs.setter
    def wcs(self, wcs):
        self._wcs = wcs
        if wcs is None:
            for k in self.keys():
                if k.find('_WCS') >= 0:
                    self.remove_column(k)
        # update wcs columns


    def dupfind(self, x='XWIN_IMAGE', y='YWIN_IMAGE', tol=1.):
        """Find and return duplicates in the catalog as a Catalog object.

        Duplicates are identified by having one or more other entries at distances between (`x`, `y`) < `tol`.
        """

        nps = len(xx)
        dim = len(xx[0])
        reg_code = np.zeros(nps)
        nregs = int(ceil(nps/50))
        boundry = np.zeros()
        for p in xx:
            p

        score = np.ones(nps)

        for i in range(nps):
            for j in range(i+1,nps):
                dist = np.sqrt((cats[i][x]-cats[j][x])**2+(cats[i][y]-cats[j][y])**2)
                if dist<1:
                    score[i] = score[i]*0.5
                    score[j] = score[j]*0.5


        fcats = findfile('Catalogs/20190106/','_proc.cat')
        cats = table.vstack([ascii_read(f) for f in fcats])
        nps = len(cats)
        score = ones(nps)*0.5
        for i in range(nps):
            for j in range(i+1,nps):
                dist = np.sqrt((cats[i]['XWIN_IMAGE']-cats[j]['XWIN_IMAGE'])**2+(cats[i]['YWIN_IMAGE']-cats[j]['YWIN_IMAGE'])**2)
                if dist<1:
                    score[i] = score[i]*0.5
                    score[j] = score[j]*0.5
        figure(figsize=(8,6))
        plot(score,'o')


def show_map(ax, data, title='Map', vmin=None, vmax=None, origin='down', cmap='jet', colorbarticks=None):
#    plt.figure(figsize=(8,3.3))
    im = ax.imshow(data,vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto',origin=origin)
    sz = data.shape
    pplot(ax,xticks=np.linspace(0,sz[1],7),title=title,xlim=[0,sz[1]],ylim=[0,sz[0]])
#    ax.yticks(np.linspace(0,180,7),[str(x) for x in range(-90,91,30)])
    ax.set_yticks(np.linspace(0,sz[0],5))
    lbl = [str(x) for x in range(-90,92,45)]
    lbl[1] = ''
    lbl[3] = ''
    ax.set_yticklabels(lbl)
    ax.set_xticks(np.linspace(0,sz[1],7))
    lbl = [str(x) for x in range(0,361,60)]
    lbl[1] = ''
    lbl[2] = ''
    lbl[4] = ''
    lbl[5] = ''
    ax.set_xticklabels(lbl)
    ax.grid()
    plt.colorbar(mappable=im, ticks=colorbarticks, ax=ax)

def add_sample_site(ax, size):
    # sample site
    sites = Table({'name': ['DL15', 'DL6', 'TM11', 'BB22', 'DL9'],
                   'lat': [56.6, 11.0,-1.5,-19.2,24],
                   'lon': [42.6, 88.5, 300.3, 97.7, 206.7]})
    for s in sites:
        xx = s['lon']/360*size[1]
        yy = (s['lat']+90)/180*size[0]
        ax.plot(xx, yy, 'o',ms=10,mfc='w',mec='k')
        ax.text(xx+2,yy,s['name'],color='w')


# OCAMS resolved photometric mapping
class OCAMS_Photometry():

    def __init__(self, datadir=None, filter=['pan', 'b', 'v', 'w', 'x'],
        match_map=True, binsize=None, pho_datafile=None, grid_datafile=None,
        mesh_size=1, suffix='', model=None, overwrite=False, maxmem=10,
        verbose=True, **kwargs):
        """
        datadir : str
            Directory of input data
        filter : list of str
            The filters to be processed
        match_map : bool
            Bin PolyCam images by 5x5 to match the resolution of MapCam
        binsize : num
            The spatial bin size in pixels for images in ingestion
        pho_datafile : str
            The root file name to save `PhotometricData`.  The data will be
            saved to pho_datafile+'pan'
        grid_datafile : str
            The root file name to save `PhotometricDataGrid`.  Default is
            f'{pho_datafile}_grid_{mesh_size}deg'
        model_file : str
            The root file name to save best-fit models.  Default is the data
            file name appended with a f'_{model.name}'.
        mesh_size : number
            The size of lat-lon mesh grid in degrees.  Default is 1.
        suffix : str
            suffix to output file names
        model : `~astropy.modeling.Model` instance
            Model instance to be fitted
        overwrite : bool
            Overwrite existing output files
        maxmem : number
            Approximate maximum memory size in GB allowed for
            `PhotometricDataGrid` object.
        verbose : bool
            Verbose output
        **kwargs : Keywords accepted by fitting
        """
        self.datadir = datadir
        self.filter = filter
        self.match_map = match_map
        self.binsize = binsize
        self.pho_datafile = pho_datafile
        self._grid_datafile = grid_datafile
        self.mesh_size = mesh_size
        self.suffix = suffix
        self.model = model
        self.overwrite = overwrite
        self.maxmem = maxmem
        self.verbose = verbose
        self.fitting_kwargs = kwargs
        self.wavelength = \
            {'pan': 651.9, 'b': 469.7, 'v': 549.5, 'w': 700.9, 'x': 854.0}

    @property
    def grid_datafile(self):
        if self._grid_datafile is None:
            if self.pho_datafile is not None:
                return f'{self.pho_datafile}_grid{self.mesh_size}deg'
            else:
                return None
        else:
            return self._grid_datafile

    @grid_datafile.setter
    def grid_datafile(self, v):
        self._grid_datafile = v

    def _generate_phodata_filename(self):
        """Generate the file name for `PhotometricData`
        """
        if self.pho_datafile is None:
            raise ValueError('`PhotometricData` file name is not'
                ' specified.')
        else:
            phofile = [f'{self.pho_datafile}_{flt}' \
                for flt in self.filter]
            if self.suffix:
                phofile = [f'{s}_{self.suffix}.fits' for s in phofile]
            else:
                phofile = [f'{s}.fits' for s in phofile]
            return phofile

    def _generate_phogrid_filename(self):
        """Generate the file names for `PhotometricDataGrid`
        """
        if self.grid_datafile is None:
            raise ValueError('`PhotometricDataGrid` file name is not'
                ' specified.')
        else:
            gridfile = [f'{self.grid_datafile}_{flt}' for flt in self.filter]
            if self.suffix:
                gridfile = [f'{s}_{self.suffix}.fits' for s in gridfile]
            else:
                gridfile = [f'{s}.fits' for s in gridfile]
            return gridfile

    def _generate_model_filename(self, datafile=None):
        """Generate the file names to save fitted model grid
        """
        from os.path import splitext
        if self.model is None:
            model_suffix = 'model'
        else:
            model_suffix = self.model.__class__.__name__
        if datafile is None:
            datafile = self._generate_phogrid_filename()
        if isinstance(datafile, str):
            datafile = splitext(datafile)[0]
            return f'{datafile}_{model_suffix}.fits'
        else:
            datafile = [splitext(s)[0] for s in datafile]
            return [f'{f}_{model_suffix}.fits' for f in datafile]

    def ingest_phodata(self, datadir=None, filter=None, pho_datafile=None,
        match_map=None, binsize=None, overwrite=None, suffix=None,
        verbose=None):
        """Ingest photometric data from images and backplanes

        See `.__init__()` for arguments.
        Return : `PhotometricData` object or list of it

        Method will ingest the images and backplanes and return a
        `PhotometricData` object, and optionally write to an output file.
        """
        from jylipy.Photometry import PhotometricData
        from os.path import basename, isfile

        if datadir is None:
            datadir = self.datadir
        if filter is None:
            filter = self.filter
        if pho_datafile is None:
            pho_datafile = self.pho_datafile
        if match_map is None:
            match_map = self.match_map
        if binsize is None:
            binsize = self.binsize
        if overwrite is None:
            overwrite = self.overwrite
        if suffix is None:
            suffix = self.suffix
        if verbose is None:
            verbose = self.verbose

        if datadir is None:
            raise ValueError('Input data directory not specified.')
        if isinstance(datadir, str):
            datadir = [datadir]
        files = np.concatenate([findfile(x,'dn.fits') for x in datadir])

        out = []
        outfiles = self._generate_phodata_filename()
        for flt,of in zip(filter, outfiles):
            fs = [x for x in files if x.find(flt+'.')!=-1]
            if verbose:
                print(f'Processing filter {flt}: {len(fs)} files found.')
            pho_all = PhotometricData()
            for f in fs:
                iof = readfits(f, verbose=False)
                if isfile(f.replace('.dn.', '.linc.')):
                    inc = readfits(f.replace('.dn.', '.linc.'), verbose=False)
                else:
                    inc = readfits(f.replace('.dn.', '.inc.'), verbose=False)
                if isfile(f.replace('.dn.', '.lemis.')):
                    emi = readfits(f.replace('.dn.', '.lemis.'), verbose=False)
                else:
                    emi = readfits(f.replace('.dn.', '.emis.'), verbose=False)
                pha = readfits(f.replace('.dn.', '.phase.'), verbose=False)
                lat = readfits(f.replace('.dn.', '.lat.'), verbose=False)
                lon = readfits(f.replace('.dn.', '.lon.'), verbose=False)
                if iof.max()>0.2:
                    # filter out icycles
                    iof = iof[200:]
                    inc = inc[200:]
                    emi = emi[200:]
                    pha = pha[200:]
                    lat = lat[200:]
                    lon = lon[200:]
                if match_map:
                    # bin PolyCam images to match MapCam resolution
                    if basename(f).find('pol') != -1:
                        iof = rebin(iof, [5, 5], mean=True)
                        inc = rebin(inc, [5, 5], mean=True)
                        emi = rebin(emi, [5, 5], mean=True)
                        pha = rebin(pha, [5, 5], mean=True)
                        lat = rebin(lat, [5, 5], mean=True)
                        lon = rebin(lon, [5, 5], mean=True)
                if binsize is not None:
                    iof = rebin(iof, [binsize, binsize], mean=True)
                    inc = rebin(inc, [binsize, binsize], mean=True)
                    emi = rebin(emi, [binsize, binsize], mean=True)
                    pha = rebin(pha, [binsize, binsize], mean=True)
                    lat = rebin(lat, [binsize, binsize], mean=True)
                    lon = rebin(lon, [binsize, binsize], mean=True)
                good = (inc<90) & (inc>0) & (emi<90) & (emi>0)
                pho = PhotometricData(iof=iof[good], inc=inc[good],
                    emi=emi[good], pha=pha[good], geolat=lat[good],
                    geolon=lon[good])
                pho_all.append(pho)
                if verbose:
                    print(f'    {basename(f)}, {len(pho)} data points')
            out.append(pho_all)
            pho_all.write(of, overwrite=overwrite)
        if len(out) == 1:
            out = out[0]
        return out

    def mesh_phodata(self, phodata=None, pho_datafile=None, mesh_size=None,
        grid_datafile=None, overwrite=None, verbose=None, maxmem=None):
        """Generate photometric data mesh

        phodata : `PhotometricData` or list of
            Input photometric data to be ported to grid data.  If None, then
            load from `pho_datafile`.  If `pho_datafile` is stil None or not
            recognized, then load from default input files.
        pho_datafile : str or list of
            File names to load data
        See `.__init__()` for other arguments.

        Return : `PhotometricDataGrid` or a list of
        """
        if mesh_size is None:
            mesh_size = self.mesh_size
        if grid_datafile is None:
            grid_datafile = self.grid_datafile
        if overwrite is None:
            overwrite = self.overwrite
        if verbose is None:
            verbose = self.verbose
        if maxmem is None:
            maxmem = self.maxmem

        lat = np.linspace(-90,90,round(180/mesh_size)+1)
        lon = np.linspace(0,360,round(360/mesh_size)+1)

        if phodata is not None:
            if isinstance(phodata, PhotometricData):
                phodata = [phodata]
            elif isinstance(phodata[0], PhotometricData):
                pass
            else:
                raise TypeError('Input data not recognized.')
        else:
            if pho_datafile is not None:
                if isinstance(pho_datafile, str):
                    phofiles = [pho_datafile]
                elif isinstance(pho_datafile[0], str):
                    phofiles = pho_datafile
                else:
                    warnings.warn('Specified input files not recognized, load'
                        ' data from default')
                    pho_datafile = None
            if pho_datafile is None:
                phofiles = self._generate_phodata_filename()
            phodata = [PhotometricData(f) for f in phofiles]

        out = []
        outfiles = self._generate_phogrid_filename()
        for p, o in zip(phodata, outfiles):
            pg = PhotometricDataGrid(lat=lat,lon=lon, maxmem=maxmem)
            pg.file = o
            pg.port(p, verbose=verbose)
            pg.write(overwrite=True)
            out.append(pg)
        if len(out) == 1:
            out = out[0]
        return out

    def fit_phomesh(self, phodata=None, grid_datafile=None, model=None,
        verbose=None, overwrite=None, **kwargs):
        """Fit photometric model to photometric grid data

        phodata : `PhotometricDataGrid` of list of
            Data to be fitted.  If `None`, then load from data file
            `grid_datafile`.  If `grid_datafile` is None or invalide, then
            load from default grid data file
        grid_datafile : str or list of
            File names to load grid data
        See `.__init__()` for other arguments.
        **kwargs : keywords accepted by fitter

        Return : `astropy.modeling.Fitter` or list of
        """
        if model is None:
            model = self.model
        if verbose is None:
            verbose = self.verbose
        if overwrite is None:
            overwrite = self.overwrite

        if phodata is not None:
            if isinstance(phodata, PhotometricDataGrid):
                phodata = [phodata]
            elif isinstance(phodata[0], PhotometricDataGrid):
                pass
            else:
                raise TypeError('Input data not recognized.')
            model_file = None
        else:
            if grid_datafile is not None:
                if isinstance(grid_datafile, str):
                    gridfiles = [grid_datafile]
                elif isinstance(grid_datafile[0], str):
                    gridfiles = grid_datafile
                else:
                    warnings.warn('Specified input files not recognized, load'
                        ' data from default')
                    grid_datafile = None
            if grid_datafile is None:
                gridfiles = self._generate_phogrid_filename()
            phodata = [PhotometricDataGrid(datafile=f) for f in gridfiles]
            model_file = self._generate_model_filename(gridfiles)
            if isinstance(model_file, str):
                model_file = [model_file]

        out = []
        fitting_kwargs = self.fitting_kwargs
        fitting_kwargs.update(kwargs)
        for i,p in enumerate(phodata):
            fit = p.fit(model, **fitting_kwargs)
            if model_file is not None:
                fit.model.write(model_file[i])
            out.append(fit)
        return out

    def cube_model_pars(self, models=None, outfile=None, cube=True,
        overwrite=None):
        """Assemble model parameters into cubes

        models : dict, `astropy.table.Table`
            'filter' : str, name of filter
            'wavelength' : number, wavelength
            'file' : str, name of fits file that stores the model grid
        outfile : str
            Name of fits file to store the cubes
        """
        from jylipy import pysis_ext
        import os

        if overwrite is None:
            overwrite = self.overwrite

        if models is None:
            file = self._generate_model_filename()
            filter = self.filter
            wavelength = [self.wavelength[x] for x in self.filter]
            models = {'filter': filter, 'wavelength': wavelength, 'file': file}

        models = Table(models)
        models.sort('wavelength')

        if outfile is None:
            cs = [np.array([x for x in models['file'][i]]) \
                for i in range(len(models))]
            bd = np.min([len(x) for x in cs])
            co = cs[0][:bd] == cs[1][:bd]
            for i in range(2, len(cs)):
                co = co & (cs[0][:bd] == cs[i][:bd])
            bd = np.where(~co)[0].min()
            outfile = ''.join(cs[0][:bd]).strip('_')
            outfile = f'{outfile}_grid{self.mesh_size}deg'
            if self.suffix:
                outfile = f'{outfile}_{self.suffix}'
            if self.model is None:
                outfile = f'{outfile}_model'
            else:
                outfile = f'{outfile}_{self.model.__class__.__name__}'
            outfile = os.path.splitext(outfile)[0]

        if cube:
            # save cube files
            fparms = [fits.open(f['file']) for f in models]
            keys = [fparms[0][i].header['extname'] for i in range(1,
                len(fparms[0]))]
            for k in keys:
                flist = []
                for i,f in enumerate(fparms):
                    filename = f'temp_{i:02d}.fits'
                    writefits(filename, f[k].data[::-1], overwrite=True)
                    pysis_ext.fits2isis(filename,
                        filename.replace('.fits','.cub'))
                    flist.append(filename.replace('.fits','.cub'))
                pysis_ext.cubeit(flist, f'{outfile}_{k}.cub')
                for i in range(len(fparms)):
                    os.remove(f'temp_{i:02d}.fits')
                    os.remove(f'temp_{i:02d}.cub')
        else:
            # save fits file
            fparms = [fits.open(f['file']) for f in models]
            out = {fparms[0][i].header['extname']: [] \
                for i in range(1,len(fparms[0]))}
            for f in fparms:
                for i in range(1,len(f)):
                    name = f[i].header['extname']
                    out[name].append(f[i].data)
            hdu = fits.PrimaryHDU(models['wavelength'])
            hdu.header['filter'] = ', '.join([f for f in models['filter']])
            hdu.header['model'] = fparms[0][0].header['model']
            hdu.header['parnames'] = fparms[0][0].header['parnames']
            hdu.header['extra'] = fparms[0][0].header['extra']
            hdus = fits.HDUList()
            hdus.append(hdu)
            for k in out.keys():
                out[k] = np.array(out[k])
                hdus.append(fits.ImageHDU(np.array(out[k]), name=k))
            hdus.writeto(outfile+'.fits', overwrite=overwrite)


# OCAMS resolved photometric mapping
class OVIRS_Photometry():

    def __init__(self, datadir=None, pho_datafile=None, grid_datafile=None,
        mesh_size=1, suffix='', model=None, overwrite=False, maxmem=10,
        verbose=True, **kwargs):
        """
        datadir : str
            Directory of input data
        pho_datafile : str
            The root file name to save `PhotometricData`.  The data will be
            saved to pho_datafile+'pan'
        grid_datafile : str
            The root file name to save `PhotometricDataGrid`.  Default is
            f'{pho_datafile}_grid_{mesh_size}deg'
        model_file : str
            The root file name to save best-fit models.  Default is the data
            file name appended with a f'_{model.name}'.
        mesh_size : number
            The size of lat-lon mesh grid in degrees.  Default is 1.
        suffix : str
            suffix to output file names
        model : `~astropy.modeling.Model` instance
            Model instance to be fitted
        overwrite : bool
            Overwrite existing output files
        maxmem : number
            Approximate maximum memory size in GB allowed for
            `PhotometricDataGrid` object.
        verbose : bool
            Verbose output
        **kwargs : Keywords accepted by fitting
        """
        self.datadir = datadir
        self.filter = filter
        self.match_map = match_map
        self.binsize = binsize
        self.pho_datafile = pho_datafile
        self._grid_datafile = grid_datafile
        self.mesh_size = mesh_size
        self.suffix = suffix
        self.model = model
        self.overwrite = overwrite
        self.maxmem = maxmem
        self.verbose = verbose
        self.fitting_kwargs = kwargs
        self.wavelength = \
            {'pan': 651.9, 'b': 469.7, 'v': 549.5, 'w': 700.9, 'x': 854.0}

    @property
    def grid_datafile(self):
        if self._grid_datafile is None:
            if self.pho_datafile is not None:
                return f'{self.pho_datafile}_grid{self.mesh_size}deg'
            else:
                return None
        else:
            return self._grid_datafile

    @grid_datafile.setter
    def grid_datafile(self, v):
        self._grid_datafile = v

    def _generate_phodata_filename(self):
        """Generate the file name for `PhotometricData`
        """
        if self.pho_datafile is None:
            raise ValueError('`PhotometricData` file name is not'
                ' specified.')
        else:
            phofile = [f'{self.pho_datafile}_{flt}' \
                for flt in self.filter]
            if self.suffix:
                phofile = [f'{s}_{self.suffix}.fits' for s in phofile]
            else:
                phofile = [f'{s}.fits' for s in phofile]
            return phofile

    def _generate_phogrid_filename(self):
        """Generate the file names for `PhotometricDataGrid`
        """
        if self.grid_datafile is None:
            raise ValueError('`PhotometricDataGrid` file name is not'
                ' specified.')
        else:
            gridfile = [f'{self.grid_datafile}_{flt}' for flt in self.filter]
            if self.suffix:
                gridfile = [f'{s}_{self.suffix}.fits' for s in gridfile]
            else:
                gridfile = [f'{s}.fits' for s in gridfile]
            return gridfile

    def _generate_model_filename(self, datafile=None):
        """Generate the file names to save fitted model grid
        """
        from os.path import splitext
        if self.model is None:
            model_suffix = 'model'
        else:
            model_suffix = self.model.__class__.__name__
        if datafile is None:
            datafile = self._generate_phogrid_filename()
        if isinstance(datafile, str):
            datafile = splitext(datafile)[0]
            return f'{datafile}_{model_suffix}.fits'
        else:
            datafile = [splitext(s)[0] for s in datafile]
            return [f'{f}_{model_suffix}.fits' for f in datafile]

    def ingest_phodata(self, datadir=None, filter=None, pho_datafile=None,
        match_map=None, binsize=None, overwrite=None, suffix=None,
        verbose=None):
        """Ingest photometric data from images and backplanes

        See `.__init__()` for arguments.
        Return : `PhotometricData` object or list of it

        Method will ingest the images and backplanes and return a
        `PhotometricData` object, and optionally write to an output file.
        """
        from jylipy.Photometry import PhotometricData
        from os.path import basename, isfile

        if datadir is None:
            datadir = self.datadir
        if filter is None:
            filter = self.filter
        if pho_datafile is None:
            pho_datafile = self.pho_datafile
        if match_map is None:
            match_map = self.match_map
        if binsize is None:
            binsize = self.binsize
        if overwrite is None:
            overwrite = self.overwrite
        if suffix is None:
            suffix = self.suffix
        if verbose is None:
            verbose = self.verbose

        if datadir is None:
            raise ValueError('Input data directory not specified.')
        if isinstance(datadir, str):
            datadir = [datadir]
        files = np.concatenate([findfile(x,'dn.fits') for x in datadir])

        out = []
        outfiles = self._generate_phodata_filename()
        for flt,of in zip(filter, outfiles):
            fs = [x for x in files if x.find(flt+'.')!=-1]
            if verbose:
                print(f'Processing filter {flt}: {len(fs)} files found.')
            pho_all = PhotometricData()
            for f in fs:
                iof = readfits(f, verbose=False)
                if isfile(f.replace('.dn.', '.linc.')):
                    inc = readfits(f.replace('.dn.', '.linc.'), verbose=False)
                else:
                    inc = readfits(f.replace('.dn.', '.inc.'), verbose=False)
                if isfile(f.replace('.dn.', '.lemis.')):
                    emi = readfits(f.replace('.dn.', '.lemis.'), verbose=False)
                else:
                    emi = readfits(f.replace('.dn.', '.emis.'), verbose=False)
                pha = readfits(f.replace('.dn.', '.phase.'), verbose=False)
                lat = readfits(f.replace('.dn.', '.lat.'), verbose=False)
                lon = readfits(f.replace('.dn.', '.lon.'), verbose=False)
                if iof.max()>0.2:
                    # filter out icycles
                    iof = iof[200:]
                    inc = inc[200:]
                    emi = emi[200:]
                    pha = pha[200:]
                    lat = lat[200:]
                    lon = lon[200:]
                if match_map:
                    # bin PolyCam images to match MapCam resolution
                    if basename(f).find('pol') != -1:
                        iof = rebin(iof, [5, 5], mean=True)
                        inc = rebin(inc, [5, 5], mean=True)
                        emi = rebin(emi, [5, 5], mean=True)
                        pha = rebin(pha, [5, 5], mean=True)
                        lat = rebin(lat, [5, 5], mean=True)
                        lon = rebin(lon, [5, 5], mean=True)
                if binsize is not None:
                    iof = rebin(iof, [binsize, binsize], mean=True)
                    inc = rebin(inc, [binsize, binsize], mean=True)
                    emi = rebin(emi, [binsize, binsize], mean=True)
                    pha = rebin(pha, [binsize, binsize], mean=True)
                    lat = rebin(lat, [binsize, binsize], mean=True)
                    lon = rebin(lon, [binsize, binsize], mean=True)
                good = (inc<90) & (inc>0) & (emi<90) & (emi>0)
                pho = PhotometricData(iof=iof[good], inc=inc[good],
                    emi=emi[good], pha=pha[good], geolat=lat[good],
                    geolon=lon[good])
                pho_all.append(pho)
                if verbose:
                    print(f'    {basename(f)}, {len(pho)} data points')
            out.append(pho_all)
            pho_all.write(of, overwrite=overwrite)
        if len(out) == 1:
            out = out[0]
        return out

    def mesh_phodata(self, phodata=None, pho_datafile=None, mesh_size=None,
        grid_datafile=None, overwrite=None, verbose=None, maxmem=None):
        """Generate photometric data mesh

        phodata : `PhotometricData` or list of
            Input photometric data to be ported to grid data.  If None, then
            load from `pho_datafile`.  If `pho_datafile` is stil None or not
            recognized, then load from default input files.
        pho_datafile : str or list of
            File names to load data
        See `.__init__()` for other arguments.

        Return : `PhotometricDataGrid` or a list of
        """
        if mesh_size is None:
            mesh_size = self.mesh_size
        if grid_datafile is None:
            grid_datafile = self.grid_datafile
        if overwrite is None:
            overwrite = self.overwrite
        if verbose is None:
            verbose = self.verbose
        if maxmem is None:
            maxmem = self.maxmem

        lat = np.linspace(-90,90,round(180/mesh_size)+1)
        lon = np.linspace(0,360,round(360/mesh_size)+1)

        if phodata is not None:
            if isinstance(phodata, PhotometricData):
                phodata = [phodata]
            elif isinstance(phodata[0], PhotometricData):
                pass
            else:
                raise TypeError('Input data not recognized.')
        else:
            if pho_datafile is not None:
                if isinstance(pho_datafile, str):
                    phofiles = [pho_datafile]
                elif isinstance(pho_datafile[0], str):
                    phofiles = pho_datafile
                else:
                    warnings.warn('Specified input files not recognized, load'
                        ' data from default')
                    pho_datafile = None
            if pho_datafile is None:
                phofiles = self._generate_phodata_filename()
            phodata = [PhotometricData(f) for f in phofiles]

        out = []
        outfiles = self._generate_phogrid_filename()
        for p, o in zip(phodata, outfiles):
            pg = PhotometricDataGrid(lat=lat,lon=lon, maxmem=maxmem)
            pg.file = o
            pg.port(p, verbose=verbose)
            pg.write(overwrite=True)
            out.append(pg)
        if len(out) == 1:
            out = out[0]
        return out

    def fit_phomesh(self, phodata=None, grid_datafile=None, model=None,
        verbose=None, overwrite=None, **kwargs):
        """Fit photometric model to photometric grid data

        phodata : `PhotometricDataGrid` of list of
            Data to be fitted.  If `None`, then load from data file
            `grid_datafile`.  If `grid_datafile` is None or invalide, then
            load from default grid data file
        grid_datafile : str or list of
            File names to load grid data
        See `.__init__()` for other arguments.
        **kwargs : keywords accepted by fitter

        Return : `astropy.modeling.Fitter` or list of
        """
        if model is None:
            model = self.model
        if verbose is None:
            verbose = self.verbose
        if overwrite is None:
            overwrite = self.overwrite

        if phodata is not None:
            if isinstance(phodata, PhotometricDataGrid):
                phodata = [phodata]
            elif isinstance(phodata[0], PhotometricDataGrid):
                pass
            else:
                raise TypeError('Input data not recognized.')
            model_file = None
        else:
            if grid_datafile is not None:
                if isinstance(grid_datafile, str):
                    gridfiles = [grid_datafile]
                elif isinstance(grid_datafile[0], str):
                    gridfiles = grid_datafile
                else:
                    warnings.warn('Specified input files not recognized, load'
                        ' data from default')
                    grid_datafile = None
            if grid_datafile is None:
                gridfiles = self._generate_phogrid_filename()
            phodata = [PhotometricDataGrid(datafile=f) for f in gridfiles]
            model_file = self._generate_model_filename(gridfiles)
            if isinstance(model_file, str):
                model_file = [model_file]

        out = []
        fitting_kwargs = self.fitting_kwargs
        fitting_kwargs.update(kwargs)
        for i,p in enumerate(phodata):
            fit = p.fit(model, **fitting_kwargs)
            if model_file is not None:
                fit.model.write(model_file[i])
            out.append(fit)
        return out

    def cube_model_pars(self, models=None, outfile=None, cube=True,
        overwrite=None):
        """Assemble model parameters into cubes

        models : dict, `astropy.table.Table`
            'filter' : str, name of filter
            'wavelength' : number, wavelength
            'file' : str, name of fits file that stores the model grid
        outfile : str
            Name of fits file to store the cubes
        """
        from jylipy import pysis_ext
        import os

        if overwrite is None:
            overwrite = self.overwrite

        if models is None:
            file = self._generate_model_filename()
            filter = self.filter
            wavelength = [self.wavelength[x] for x in self.filter]
            models = {'filter': filter, 'wavelength': wavelength, 'file': file}

        models = Table(models)
        models.sort('wavelength')

        if outfile is None:
            cs = [np.array([x for x in models['file'][i]]) \
                for i in range(len(models))]
            bd = np.min([len(x) for x in cs])
            co = cs[0][:bd] == cs[1][:bd]
            for i in range(2, len(cs)):
                co = co & (cs[0][:bd] == cs[i][:bd])
            bd = np.where(~co)[0].min()
            outfile = ''.join(cs[0][:bd]).strip('_')
            outfile = f'{outfile}_grid{self.mesh_size}deg'
            if self.suffix:
                outfile = f'{outfile}_{self.suffix}'
            if self.model is None:
                outfile = f'{outfile}_model'
            else:
                outfile = f'{outfile}_{self.model.__class__.__name__}'
            outfile = os.path.splitext(outfile)[0]

        if cube:
            # save cube files
            fparms = [fits.open(f['file']) for f in models]
            keys = [fparms[0][i].header['extname'] for i in range(1,
                len(fparms[0]))]
            for k in keys:
                flist = []
                for i,f in enumerate(fparms):
                    filename = f'temp_{i:02d}.fits'
                    writefits(filename, f[k].data[::-1], overwrite=True)
                    pysis_ext.fits2isis(filename,
                        filename.replace('.fits','.cub'))
                    flist.append(filename.replace('.fits','.cub'))
                pysis_ext.cubeit(flist, f'{outfile}_{k}.cub')
                for i in range(len(fparms)):
                    os.remove(f'temp_{i:02d}.fits')
                    os.remove(f'temp_{i:02d}.cub')
        else:
            # save fits file
            fparms = [fits.open(f['file']) for f in models]
            out = {fparms[0][i].header['extname']: [] \
                for i in range(1,len(fparms[0]))}
            for f in fparms:
                for i in range(1,len(f)):
                    name = f[i].header['extname']
                    out[name].append(f[i].data)
            hdu = fits.PrimaryHDU(models['wavelength'])
            hdu.header['filter'] = ', '.join([f for f in models['filter']])
            hdu.header['model'] = fparms[0][0].header['model']
            hdu.header['parnames'] = fparms[0][0].header['parnames']
            hdu.header['extra'] = fparms[0][0].header['extra']
            hdus = fits.HDUList()
            hdus.append(hdu)
            for k in out.keys():
                out[k] = np.array(out[k])
                hdus.append(fits.ImageHDU(np.array(out[k]), name=k))
            hdus.writeto(outfile+'.fits', overwrite=overwrite)
