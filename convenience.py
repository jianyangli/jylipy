# Convenience package
#

class dir(object):

	work = '/Users/jyli/Work/'
	spice_generic_kernel = work+'naif/generic_kernels/'
	dawn = work+'Dawn/'
	c13a1 = work+'comet_c2013a1/'

class spice(object):

	@staticmethod
	def load_generic():
		import spice
		kernels = [dir.spice_generic_kernel+x for x in 'lsk/naif0010.tls pck/pck00010.tpc spk/planets/de430.bsp names.ker'.split()]
		for k in kernels:
			spice.furnsh(k)

