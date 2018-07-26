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
Stopped version tracking here, use git for change control
'''

from pysis import isis
from pysis.cubefile import CubeFile
from os.path import splitext, isfile, isdir, basename, dirname, join

__all__ = isis.__all__ + ['CubeFile', 'EZWrapper', 'iter_isis', 'listgen']

_to_load = isis.__all__


def iter_isis(func, indata, *args, **kwargs):
    '''Iterate a command in subdirectories

    Parameters
    ----------
    func : Function to execute
    indata : str or iterable of str
      A single file name, a list of file name, or a directory name.  If
      directory, then all files in it, and all subdirectories, will be
      processed iteratively.
    outdata : str or iterable of str, optional
      Output file name(s) or directory name corresponding to `indata`.
      If `indata` is a file/dir, then `outdata` is considered a file/dir.
      If `indata` is a list of files/dirs, then: if `outdata` is a single
      string, then it's considered a directory name to store all output;
      if `outdata` is a list of string, then it is individual output
      name corresponding to input files
      It can be absent if the isis command `func` doesn't require a
      to= parameter
    verbose : bool, taggle on verbose mode
    ext : str, the extension or ending of files to be processed.  Can be used
      to filter out unwanted files in the same directory
    suffix : str, if provided, then the extension of input file name will be
      replaced by this string.  Note that if use this keyword to change the
      extension of input file name, then the . (dot) has to be included here
    **kwargs : the input parameters for isis command `func`

    v1.0.0, 05/26/2015, JYL @PSI
    '''
    from .core import findfile, is_iterable
    from os import makedirs

    if len(args) == 1:
        outdata = args[0]
    elif len(args) == 0:
        outdata = None
    else:
        raise TypeError('iter_isis() takes 2 or 3 arguments, {0} received'.format(2+len(args)))

    if is_iterable(indata):
        # if input is a string of file/dir names, loop through them
        if outdata is not None:
            if is_iterable(outdata):
                for fi, fo in zip(indata, outdata):
                    iter_isis(func, fi, fo, **kwargs)
            else:
                # `outdata` is a directory name
                makedirs(outdata, exist_ok=True)
                for fi in indata:
                    iter_isis(func, fi, join(outdata, basename(fi)), **kwargs)
        else:
            for fi in indata:
                iter_isis(func, fi, **kwargs)
    elif isdir(indata):
        # if input is a directory
        verbose = kwargs.get('verbose', True)
        ext = kwargs.get('ext', None)
        insidedir = findfile(indata, dir=True)
        insidefile = findfile(indata)
        if ext is not None:
            insidefile = [x for x in insidefile if x.lower().endswith(ext)]
        if len(insidefile) > 0:
            if verbose:
                print('directory {0}/: {1} files found'.format(indata, len(insidefile)))
            if outdata is not None:
                makedirs(outdata, exist_ok=True)
                for fi in insidefile:
                    fo = join(outdata, basename(fi))
                    iter_isis(func, fi, fo, **kwargs)
            else:
                for fi in insidefile:
                    iter_isis(func, fi, **kwargs)
        if len(insidedir) > 0:
            if verbose:
                print('directory {0}/: {1} subdirectories found'.format(indata, len(insidedir)))
            if outdata is not None:
                makedirs(outdata, exist_ok=True)
                for di in insidedir:
                    print('processing directory {0}/', basename(di))
                    iter_isis(func, di, join(outdata, basename(di)), **kwargs)
            else:
                for di in insidedir:
                    iter_isis(func, di, **kwargs)
    elif isfile(indata):
        # if input is a single file
        verbose = kwargs.pop('verbose', True)
        suffix = kwargs.pop('suffix', None)
        ext = kwargs.pop('ext', None)
        if verbose:
            print('processing file {0}'.format(basename(indata)))
        isiskeys = {}
        isiskeys['from'] = indata
        if outdata is not None:
            makedirs(dirname(outdata), exist_ok=True)
            if suffix is not None:
                fo = splitext(outdata)[0]+suffix
            isiskeys['to'] = fo
        isiskeys.update(kwargs)
        func(**isiskeys)
    else:
        raise ValueError('input not found: {0}'.format(indata))


def listgen(outfile, strlist, overwrite=True):
    '''Generate a list file with the strings in `strlist`'''

    if not overwrite:
        if isfile(outfile):
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
          isis.cam2map(**{'from': infile, 'to': outfile, 'map': mapfile, ...})

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
          of functions.  The 'yes' or 'no' value keys can be set by boolean
          values True or False.

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
            if isinstance(infile, str) or isinstance(infile, bytes):
                parms = {'from': infile}
            elif (not isinstance(infile, (str,bytes))) and hasattr(infile, '__iter__'):
                fromlist = True
                listfile = kwargs.pop('listfile', None)
                if listfile is None:
                    lstfile = join(tempdir, self.func.name.split('/')[-1]+'.lst')
                else:
                    lstfile = listfile
                listgen(lstfile, infile)
                parms = {'fromlist': lstfile}
            else:
                raise TypeError('str/bytes or list of str/bytes expected, {0} received'.formate(type(infile)))
            if len(args)>1:
                parms[self.to] = args[1]

        parms.update(self.kwargs)
        log = kwargs.pop('log', None)
        parms.update(kwargs)

        if len(parms)>0:
            if log is None:
                logfile = join(tempdir, 'temp.log')
            else:
                logfile = log
            parms['-log'] = logfile

        for k in parms:
            if parms[k] == True:
                parms[k] = 'yes'
            elif parms[k] == False:
                parms[k] = 'no'

        self.func(**parms)

        if fromlist and (listfile is None):
            os.remove(lstfile)
        if (len(parms)>0) & (log is None) and ('-log' in list(parms.keys())) and isfile(logfile):
            os.remove(logfile)


for c in _to_load:
    exec(c+' = EZWrapper(isis.'+c+')')


automos = EZWrapper(isis.automos, to='mosaic', priority='average')
