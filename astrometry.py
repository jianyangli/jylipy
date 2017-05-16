# astrometry.net wrapper for python
#
# 3/21/2017, JYL @PSI

from astropy.io import fits
from astropy import wcs
import subprocess

class field(object):

	def __init__(self, imagefile, **options):
		self.file = imagefile
		self.options = {}
		image = fits.open(imagefile)
		w = wcs.WCS(image[0].header)
		self.options['ra'] = w.wcs.crval[0]
		self.options['dec'] = w.wcs.crval[1]
		for k,v in list(options.items()):
			self.options[k] = v

	def solve(self, **options):
		args = ['solve-field', self.file]
		keys = list(options.keys())
		for k,v in list(self.options.items()):
			if k not in keys:
				options[k] = self.options[k]
		for k,v in list(options.items()):
			k = k.replace('_','-')
			ndashes = 1 if len(k) == 1 else 2
			args.append('{0}{1}'.format(ndashes * '-', k))
			if v is not None:
				args.append(str(v))

		subprocess.call(args)
