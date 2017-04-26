from .core import Image
from .apext import fits


class HSTImage(Image):

	def __new__(cls, inputfile, dtype=None):
		fitsfile = fits.open(inputfile)
		name, hdr, data, dtype, shape = [], [], [], [], []
		for hdu in fitsfile:
			hdr.append(hdu.header)
			if hdu.data is not None:
				if isinstance(hdu.data, fits.fitsrec.FITS_rec):
					shape.append(None)
				else:
					name.append(hdu.name)
					data.append(hdu.data)
					dtype.append(hdu.data.dtype)
					shape.append(hdu.data.shape)

		data = fitsfile[0].data
		obj = Image(data, dtype=dtype).view(HRCImage)
		obj.header = fitsfile[0].header
		obj.source = inputfile
		obj.geometry = {}
		obj.calibration = {}
		return obj

	def __finalize_array__(self, obj):
		super(HRCImage, self).__array_finalize__(obj)
		if obj is None: return
		self.source = getattr(obj, 'source', None)
		self.header = getattr(obj, 'header', None)
		self.geometry = getattr(obj, 'geometry', {})
		self.calibration = getattr(obj, 'calibration', {})
		if hasattr(obj, 'records'):
			self.records = copy(obj.records)
			for k in obj.records:
				self.__dict__[k] = copy(getattr(obj, k))

