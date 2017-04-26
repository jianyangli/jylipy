import numpy as np, visvis as vv
from astropy import units as u, table
from ..core import *


def meshRead(fname):

	'''
	'''

	f = open(fname, 'r')

	# Skip comment lines and empty lines
	line = f.next()
	while line[0] in ['#', '\n']:
		line = f.next()

	# Number of vertices and triangles
	nvert, ntri = [int(x) for x in line.split()]

	# Vertices
	vert = np.empty((nvert,3), dtype=np.float32)
	for i in range(nvert):
		line = f.next()
		while line[0] in ['#', '\n']:
			line = f.next()
		vert[i] = [float(x) for x in line.split()]

	# Triangles
	tri = np.empty((ntri,3), dtype=np.int32)
	for i in range(ntri):
		line = f.next()
		while line[0] in ['#', '\n']:
			line = f.next()
		tri[i] = [int(x) for x in line.split()]

	f.close()

	return vert, tri


def meshWrite(fname, vert, tri, comment=None):
	'''
	'''

	# Test the validity of verticies
	try:
		vv.wobjects.polygonalModeling.checkDimsOfArray(vert, 3)
	except ValueError:
		raise ValueError('vertices must be Nx3 array.')

	# Test the validity of triangles
	try:
		vv.wobjects.polygonalModeling.checkDimsOfArray(tri, 3)
	except ValueError:
		raise ValueError('triangles must be Nx3 array.')

	f = open(fname, 'w')

	# Write comment lines
	if comment is not None:
		if not hasattr(comment,'__iter__'):
			comment = [comment]
		for c in comment:
			print c
			f.write('# '+c+'\n')
		f.write('\n')

	# Write plate shape model
	f.write('%d   %d\n' % (np.asarray(vert).shape[0], np.asarray(tri).shape[0]))
	for v in vert:
		f.write('%f   %f   %f\n' % tuple(v))
	for t in tri:
		f.write('%d   %d   %d\n' % tuple(t))

	f.close()


def vrmlRead(fname):

	'''
	'''

	f = open(fname, 'r')

	# Read in vertices
	line = f.next()
	while line.find('point') < 0:
		line = f.next()
	vert = []
	line = f.next()
	while True:
		line = line.strip().split('#')[0]
		if line != '':
			line = line.strip(' ,.;{[]}\n')
			if line != '':
				try:
					v = [float(x.strip(' ,;.')) for x in line.split()]
				except:
					break
				vert.append(v)
		line = f.next()
	vert = np.asarray(vert)

	# Read in triangles
	while line.find('coordIndex') < 0:
		line = f.next()
	tri = []
	line = f.next()
	while True:
		line = line.strip().split('#')[0]
		if line != '':
			line = line.strip(' ,.;{[]}\n')
			if line != '':
				try:
					ind = [int(x.strip(' ,;.')) for x in line.strip(' ,.;\n]}').split()]
				except:
					break
				tri.append(ind[:-1])
		line = f.next()
	tri = np.asarray(tri)

	f.close()

	return vert, tri


class BaseCCD(object):
	'''
 Generic CCD base class
	'''

	def __init__(self, npix=[1024,1024], pixsize=None, ifov=None, bias=0., dark=0., gain=1., noise=0., saturation=None, adc_depth=16, photcal=1., flat=None):

		self._npix = tuple(duplicate(npix))
		if pixsize is not None:
			self._pixsize = tuple(duplicate(pixsize, 2, u.um))
		else:
			self._pixsize = None
		if ifov is not None:
			self._ifov = tuple(duplicate(ifov, 2, u.rad))
		else:
			self._ifov = None
		self._bias = duplicate(bias, 1, u.electron)
		self._dark = duplicate(dark, 1, u.Unit('electron/s'))
		self._gain = duplicate(gain, 1, u.Unit('electron/adu'))
		self._noise = duplicate(noise, 1, u.electron)
		if saturation is not None:
			self._saturation = duplicate(saturation, 1, u.electron)
		else:
			self._saturation = None
		self._adc_depth = duplicate(adc_depth, 1)
		self._photcal = duplicate(photcal, 1, u.Unit('(W m-2 um-1 sr-1)/(adu s-1)'))
		if flat is not None:
			value = np.asarray(value)
			assert value.shape == self.npix
			self._flat = value/value.mean()
		else:
			self._flat = np.ones(self._npix)
		self._data = np.zeros(self._npix)*u.adu

	def __call__(self):
		return self.data

	def __str__(self):
		keys = [('npix', self._npix),
				('pixsize', self._pixsize),
				('ifov', self._ifov),
				('bias', self._bias),
				('dark', self._dark),
				('gain', self._gain),
				('noise', self._noise),
				('saturation', self._saturation),
				('adc_depth', self._adc_depth),
				('photcal', self._photcal)]

		part = ['{0}: {1}'.format(k,v) for k, v in keys]
		return '\n'.join(part)

	@property
	def npix(self):
		'''
		Size of CCD (height, width), integers.  Default size is (1024, 1024)
		'''
		return self._npix
	@npix.setter
	def npix(self, value):
		self._npix = tuple(duplicate(value))

	@property
	def pixsize(self):
		'''
		The physical size of pixels (y, x) in astropy quantity
		'''

		return self._pixsize
	@pixsize.setter
	def pixsize(self, value):
		self._pixsize = tuple(duplicate(value, 2, u.um))

	@property
	def ifov(self):
		'''
		The instantaneous FOV of pixels (angular size) (y, x) in astropy
		quantity
		'''

		return self._ifov
	@ifov.setter
	def ifov(self, value):
		self._ifov = tuple(duplicate(value, 2, u.rad))

	@property
	def bias(self):
		return self._bias
	@bias.setter
	def bias(self, value):
		self._bias = duplicate(value, 1, u.electron)

	@property
	def dark(self):
		'''
		Dark current, in astropy quantity with unit like 'electron/s'
		'''
		return self._dark
	@dark.setter
	def dark(self, value):
		self._dark = duplicate(value, 1, u.Unit('electron/s'))

	@property
	def gain(self):
		'''
		CCD gain, in astropy quantity with unit like 'electron/adu'
		'''
		return self._gain
	@gain.setter
	def gain(self, value):
		self._gain = duplicate(value, 1, u.Unit('electron/adu'))

	@property
	def noise(self):
		return self._noise
	@noise.setter
	def noise(self, value):
		self._noise = duplicate(value, 1, u.electron)

	@property
	def saturation(self):
		return self._saturation
	@saturation.setter
	def saturation(self, value):
		self._saturation = duplicate(value, 1, u.electron)

	@property
	def adc_depth(self):
		return self._adc_depth
	@adc_depth.setter
	def adc_depth(self, value):
		self._adc_depth = duplicate(value, 1)

	@property
	def photcal(self):
		'''
		Photometric calibration coefficient, in astropy with unit like
		spectral radiance per (adu/s)
		'''
		return self._photcal
	@photcal.setter
	def photcal(self, value):
		self._photcal = duplicate(value, 1, u.Unit('(W m-2 um-1 sr-1)/(adu s-1)'))

	@property
	def flat(self):
		'''
		A 2D numpy array of the same size as CCD, the flatfield of the CCD
		'''
		return self._flat
	@flat.setter
	def flat(self, value):
		assert value.shape == self.npix
		self._flat = value/value.mean()

	@property
	def data(self):
		'''
		A 2D numpy array, the image on the CCD
		'''
		return self._data

	def exposure(self, scene, time):
		'''
		Expose the CCD to `scene` for exposure time `time`
		'''
		assert isinstance(scene, u.quantity.Quantity) and (scene.shape == self._npix)
		time = duplicate(time, 1, u.s)
		im = ((scene/self._photcal*self._gain+self._dark)*time).to(u.electron)
		im = self._flat*np.random.poisson(im.value)*im.unit+self._bias
		if self._saturation is not None:
			im = np.clip(im, 0, self._saturation)
		else:
			im = np.clip(im, 0, im.max())
		if self._noise != 0:
			rn = np.random.normal(0., self._noise.value, np.size(self._data)).reshape(self._npix)*self._noise.unit
		else:
			rn = np.zeros_like(im)
		im += rn
		im /= self._gain
		self._data = np.clip(im, 0, 2**self._adc_depth*u.adu)


class BaseLens(vv.cameras.ThreeDCamera):
	'''
	'''

	def __init__(self):
		pass


	def __call__(self, scene):
		'''
		Lens transformation for given scene
		'''
		return scene

	def __str__(self):
		keys = (['fnumber', self._fnumber],
				['focal_length', self._focal_length])
		part = ['{0}: {1}'.format(k,v) for k, v in keys]
		return '\n'.join(part)

	@property
	def fnumber(self):
		return self._fnumber
	@fnumber.setter
	def fnumber(self, value):
		self._fnumber = value

	@property
	def focal_length(self):
		return self._focal_length
	@focal_length.setter
	def focal_length(self, value):
		self._focal_length = value


class FilterWheel(object):
	'''
	'''

	def __init__(self):
		'''
		'''


	def __str__(self):
		keys = (['name', self._name],
				['npos', self._npos],
				['current', self._current],
				['filters', self._filters])
		part = ['{0}: {1}'.format(k, v) for k, v in keys]
		return '\n'.join(part)

	@property
	def name(self):
		return self._name
	@name.setter
	def name(self, value):
		self._name = value

	@property
	def npos(self):
		return self._npos
	@npos.setter
	def npos(self, value):
		self._npos = value

	@property
	def curent(self):
		return self._curent
	@curent.setter
	def curent(self, value):
		self._curent = value

	@property
	def filters(self):
	    return self._filters
	@filters.setter
	def filters(self, value):
	    self._filters = value


class BaseCamera(object):
	'''
	'''

	_lens = BaseLens()
	_filter_wheel = FilterWheel()
	_CCD = BaseCCD()

	_ifov = 0.
	_photcal = 0.

	def __init__(self):
		'''
		'''

	def __str__(self):
		'''
		'''

	def __call__(self):
		'''
		'''


class Camera(vv.cameras.ThreeDCamera):
	'''
	'''

	# Pixel scale, default unit deg
	ifov = None
	# Number of pixels
	npix = None
	#

	def __init__(self, ifov=None, npix=None):

		super(Camera, self).__init__()

		if ifov is not None:
			self.ifov = ifov
			if type(self.ifov) is not u.quantity.Quantity:
				self.ifov *= u.deg

		if npix is not None:
			self.npix = npix

		if self.ifov is not None and self.npix is not None:
			self.fov = (self.ifov*self.npix).to(u.deg).value

		self.loc = (0., 0., 0.)

