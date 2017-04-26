'''pysis extension including some convenience tools for using pysis

Modifications to the original pysis:

1. Reverse the order of data along vertical axis to put the origin
  (0,0) at lower-left corner, to be consistent with FITS format and
  DS9 display
2. Import all the isis command to the top level
3. Add a function `iter_isis` to execute ISIS commands iteratively
  into subdirectories

v0.0.1 : 5/26/2015, JYL @PSI
v0.0.2 : 1/11/2016, JYL @PSI
  Rename `iter_exec` to `iter_isis` to avoid name conflict with
  `iter_exec` in jylipy.core
v0.0.3 : 3/1/2016, JYL @PSI
  Define a wrapper `EZWrapper` to change the API of ISIS command calls
v0.0.4 : 5/17/2016, JYL @PSI
  Include all ISIS commands in the wrapped API
'''

import os
import pysis

#_to_load = ['explode', 'ratio', 'catlab', 'cubeit', 'mosrange', 'fits2isis', 'cam2map', 'dawnfc2isis', 'dawnvir2isis']
_to_load = pysis.isis.__all__

class CubeFile(pysis.cubefile.CubeFile):
	'''Modified CubeFile class that reverse the data in vertical
	direction to be consistent with FITS format and DS9 display,
	where the origin (0,0) is at lower-left and vertical direction
	goes up.
	'''
	def __init__(self, *args, **kwargs):
		super(CubeFile, self).__init__(*args, **kwargs)
		self.data = self.data[:,::-1,:]


def iter_isis(func, indata, *args, **kwargs):
	'''Iterate a command in subdirectories

	Parameters
	----------
	func : Function to execute
	indata : str
	  A single file, or a directory.  If directory, then all files in
	  it, or all subdirectories, will be processed iteratively.
	outdata : str, optional
	  Output file or directory corresponding to `indata`.  I.e., if
	  `indata` is a file/dir, then `outdata` is considered a file/dir.
	  It can be absent if the isis command `func` doesn't require a
	  to= parameter
	verbose : bool, optional
	  Verbose mode.  Default is True
	**kwargs : the input parameters for isis command `func`

	v1.0.0, 05/26/2015, JYL @PSI
	'''
	from .core import findfile

	verbose = kwargs.pop('verbose', True)
	if len(args) == 1:
		outdata = args[0]
	elif len(args) == 0:
		outdata = None
	else:
		raise TypeError('iter_isis() takes 2 or 3 arguments, {0} received'.format(2+len(args)))

	isiskeys = {}
	if os.path.isfile(indata):
		if verbose:
			print 'processing file ', os.path.basename(indata)
		isiskeys['from'] = indata
		if outdata is not None:
			outpath = os.path.dirname(outdata)
			outfile = os.path.basename(outdata)
			if not os.path.isdir(outpath):
				os.makedirs(outpath)
			if outdata.split('.')[-1].lower() != 'cub':
				outdata = '.'.join(outdata.split('.')[:-1]+['cub'])
			isiskeys['to'] = outdata
		for k in kwargs.keys():
			isiskeys[k] = kwargs[k]
		func(**isiskeys)
	elif os.path.isdir(indata):
		insidedir = findfile(indata, dir=True)
		insidefile = findfile(indata)
		insidefile = [x for x in insidefile if x.lower().endswith('.img') or x.lower().endswith('.pds') or x.lower().endswith('.cub') or x.lower().endswith('.fit') or x.lower().endswith('.fits')]
		if len(insidefile) > 0:
			if verbose:
				print 'directory {0}: {1} files found'.format(os.path.basename(indata), len(insidefile))
			if outdata is not None:
				if not os.path.isdir(outdata):
					os.makedirs(outdata)
				for fi in insidefile:
					iter_isis(func, fi, os.path.join(outdata, os.path.basename(fi)), verbose=verbose, **kwargs)
			else:
				for fi in insidefile:
					iter_isis(func, fi, verbose=verbose, **kwargs)
		if len(insidedir) > 0:
			if verbose:
				print 'directory {0}: {1} subdirectories found'.format(os.path.basename(indata), len(insidedir))
			if outdata is not None:
				if not os.path.isdir(outdata):
					os.makedirs(outdata)
				for di in insidedir:
					print 'processing directory ', os.path.basename(di)
					iter_isis(func, di, os.path.join(outdata, os.path.basename(di)), verbose=verbose, **kwargs)
			else:
				for di in insidedir:
					iter_isis(func, di, verbose=verbose, **kwargs)
	else:
		raise ValueError('input not found: {0}'.format(indata))


def listgen(outfile, strlist, overwrite=True):
	'''Generate a list file with the strings in `strlist`'''

	if not overwrite:
		if os.path.isfile(outfile):
			raise IOError('output file exists')

	f = open(outfile, 'w')
	for s in strlist:
		f.write(s)
		if not s.endswith('\n'):
			f.write('\n')
	f.close()


class EZWrapper(object):
	'''Wrapper class for ISIS functions.  It converts the API of pysis.isis
	commands to the usual parameter form from the dictionary form.

	For example:

		Original API:
		  pysis.isis.cam2map(**{'from': infile, 'to': outfile, 'map': mapfile, ...})

		New API:
		  pysis_ext.cam2map()  # Open GUI
		  pysis_ext.catlab(input)  # commands that don't take `to=` argument
		  pysis_ext.cam2map(input, output, log=logfile, ...)
		  pysis_ext.cubeit(input_list, output, **kwargs)

	The new API automatically check whether the input argument is a str
	or a list.  If it's a list, then it will use `fromlist=`, otherwise
	use `from=`

	To define an ISIS command with the new API:

		cmd_name = EZWrapper(isis_name, to='to', **kwargs)

		isis_name : str, the name of ISIS command
		fromlist : bool, optional, whether the ISIS command takes 'from='
		  or 'fromlist=' input
		to : str, optional, the output signature of the ISIS command.
		  E.g., automos takes 'mosaic=' as the output rather than 'to='
		**kwargs : The default values of other arguments to all calls
		  to the newly defined function
		cmd_name : the new command defined

	Calling sequence:

		cmd_name(input, output, log=None, tempdir='.', listfile=None, **kwargs)

		input : str or list of str, the input for ISIS command
		output : str, output for ISIS command
		log : str, optional, name of log file
		tempdir : str, optional, working directory
		listfile : str, optional, the name of intermediate list file for
		  commands that takes a list file as input.  If not provided,
		  then the temporary list file will be removed after the call
		**kwargs : other key=value for the ISIS command.  The values set
		  here will override the default values set in the initialization
		  of functions.

	v1.0.0 : JYL @PSI, 3/1/2016
	'''

	def __init__(self, func, to='to', **kwargs):
		self.func = func
		self.to = to
		self.kwargs = kwargs

	def __call__(self, *args, **kwargs):

		tempdir = kwargs.pop('tempdir', '.')

		fromlist = False
		if len(args) == 0:
			parms = {}
		else:
			infile = args[0]
			if hasattr(infile, '__iter__'):
				fromlist = True
				listfile = kwargs.pop('listfile', None)
				if listfile is None:
					lstfile = os.path.join(tempdir, self.func.name.split('/')[-1]+'.lst')
				else:
					lstfile = listfile
				listgen(lstfile, infile)
				parms = {'fromlist': lstfile}
			else:
				parms = {'from': infile}
			if len(args)>1:
				parms[self.to] = args[1]

		parms.update(self.kwargs)
		log = kwargs.pop('log', None)
		parms.update(kwargs)

		if len(parms)>0:
			if log is None:
				logfile = os.path.join(tempdir, 'temp.log')
			else:
				logfile = log
			parms['-log'] = logfile

		self.func(**parms)

		if fromlist and (listfile is None):
			os.remove(lstfile)
		if (len(parms)>0) & (log is None) and ('-log' in parms.keys()) and os.path.isfile(logfile):
			os.remove(logfile)


for c in _to_load:
	exec(c+' = EZWrapper(pysis.isis.'+c+')')


automos = EZWrapper(pysis.isis.automos, to='mosaic', priority='average')
