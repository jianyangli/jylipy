# OREx software package
#
# 1/5/2017, JYL @PSI

import numpy as np, string, spice
from copy import copy
import ccdproc
from .core import *
from .vector import xyz2sph, vecsep #Image, readfits, condition, xyz2sph, Time, num, findfile, writefits, CCDData, ImageMeasurement, CaseInsensitiveOrderedDict, findfile, ascii_read
from .apext import Table, Column, units, fits
from jylipy import Photometry, pplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def spdif_mapping(data, xxx_todo_changeme, xxx_todo_changeme1, method='nearest', reduce=False):
	'''Project SPDIF data to lon-lat grid

	data : array (n, b), PSDIF data with n spectra of b bands
	lon, lat : Corresponding (longitude, latitude)
	longrid, latgrid : Longitude and latitude grid of output cube
	reduce : Reduce the data by taken the mean of each (lat,lon) grid

	v1.0.0 : Jan 5, 2017, JYL @PSI
	'''
	(lon, lat) = xxx_todo_changeme
	(longrid, latgrid) = xxx_todo_changeme1
	from scipy.interpolate import griddata
	data = np.asarray(data)
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


def load_specdata(datafile, band, iofmin=0, iofmax=1, incmin=0, incmax=90, emimin=0, emimax=90, phamin=0, phamax=180, norejection=False):
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


def pick_spectrum(datafile, xxx_todo_changeme2, radius=2., unit='radf'):
	'''Pick a spectrum within a circular footprint of radius `radius` and
	centered at (lat,lon)'''
	(lat,lon) = xxx_todo_changeme2
	data = fits.open(datafile)
	lats = data[3].data['LAT']
	lons = data[3].data['LON']
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
	for i in range(nm):
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
	ax[0].legend(mnames,loc='lower center',ncol=nm/2)
	pdf.savefig()
	plt.clf()
	f, ax = plt.subplots(nm, 1, sharex=True, num=plt.gcf().number)
	lbl = ['CORR','CORR_PHA','CORR_INC','CORR_EMI']
	for j in range(len(lbl)):
		for i in range(nm):
			ax[j].plot(q[i+1].data['WAV'], q[i+1].data[lbl[j]])
		pplot(ax[j],ylabel=lbl[j])
	pplot(ax[3],xlabel='Wavelength ($\mu$m)')
	ax[0].legend(mnames,loc='lower center',ncol=nm/2)
	pdf.savefig()
	pdf.close()


def plot_phomodel(phofile):
	par = read_phomodel(phofile)
	pdf = PdfPages(phofile.replace('.fits','.pdf'))
	for k in list(par.keys()):
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


