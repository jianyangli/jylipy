# OREx software package
#
# 1/5/2017, JYL @PSI

import numpy as np, string, spiceypy as spice
from copy import copy
import ccdproc
from .core import *
from .vector import xyz2sph, vecsep #Image, readfits, condition, xyz2sph, Time, num, findfile, writefits, CCDData, ImageMeasurement, CaseInsensitiveOrderedDict, findfile, ascii_read
from .apext import Table, Column, units, fits
from jylipy import Photometry, pplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



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
    for i in range(1,nm):
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
            ax[j].plot(q[i+1].data['WAV'], q[i+1].data[lbl[j]])
        pplot(ax[j],ylabel=lbl[j])
    pplot(ax[3],xlabel='Wavelength ($\mu$m)')
    ax[0].legend(mnames,loc='lower center',ncol=nm//2)
    pdf.savefig()
    plt.clf()
    f, ax = plt.subplots(nm, 1, sharex=True, num=plt.gcf().number)
    lbl = ['CORR','CORR_PHA','CORR_INC','CORR_EMI']
    for j in range(len(lbl)):
        for i in range(nm):
            ax[j].plot(q[i+1].data['WAV'], q[i+1].data[lbl[j]])
        pplot(ax[j],ylabel=lbl[j])
    pplot(ax[3],xlabel='Wavelength ($\mu$m)')
    ax[0].legend(mnames,loc='lower center',ncol=nm//2)
    pdf.savefig()
    pdf.close()


def plot_phomodel(phofile):
    par = read_phomodel(phofile)
    pdf = PdfPages(phofile.replace('.fits','.pdf'))
    for k in par.keys():
        np = par[k]['par'].shape[1]
        plt.clf()
        f,ax = plt.subplots(np,1,sharex=True,num=plt.gcf().number)
        for i in range(np):
            ax[i].plot(par[k]['wavelength'],par[k]['par'][:,i])
            pplot(ax[i])
        pplot(ax[-1],xlabel='Wavelength ($\mu$m)')
        pplot(ax[0],title=k)
        pdf.savefig()
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

