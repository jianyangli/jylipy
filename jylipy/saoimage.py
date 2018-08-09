'''Extended DS9 class moduel'''


import numpy as np
__all__ = ['Region', 'CircularRegion', 'EllipseRegion', 'BoxRegion', 'AnnulusRegion', 'DS9', 'getds9']

class Region(object):
    '''Base class for DS9 regions'''

    parname = ('x', 'y')
    specs = {}
    _shape = None
    zerobased = True
    size = None

    def __init__(self, *args, **kwargs):
        '''The initialization of Region class depends on its shape.
        The general form is:

        r = Region(shape, *par, frame=None, ds9=None)

        shape : str
          The shape of region, same as defined in DS9
        *par : numbers
          The parameters for shape, same definition and order as in
          DS9.get('regions -format saoimage')

        To subclass Region for specific shape:

        class BoxRegion(Region):
            parname = ('x', 'y', 'a', 'b', 'angle')
            _shape = 'box'
            size = ('a', 'b')

        The initialization of subclass is now simply:

        r = BoxRegion(x, y, a, b, angle)

        v1.0.0 : JYL @PSI, 3/6/2015
        v1.0.1 : JYL @PSI, 1/5/2017
          Added property .specs to store the properties of the region, such as
          color, width, etc.
        '''

        if len(self.parname) != len(args):
            raise TypeError('{0}.__init__() takes {1} arguments ({2} given)'.format(type(self),len(self.parname)+1,len(args)+1))
        if self._shape is None:
            self._shape = kwargs.pop('shape', None)
        self.frame = kwargs.pop('frame', None)
        self.ds9 = kwargs.pop('ds9', None)
        self.zerobased = kwargs.pop('zerobased', True)
        for i in range(len(args)):
            self.__dict__[self.parname[i]] = args[i]
        if self._shape is None:
            raise Warning('region shape is not defined')
        self.specs['color'] = kwargs.pop('color', 'green')
        self.specs['width'] = kwargs.pop('width', 1)

    @property
    def shape(self):
        return self._shape

    @property
    def par(self):
        v = []
        for k in self.parname:
            v.append(self.__dict__[k])
        return v

    @property
    def xmin(self):
        if self.size is None:
            raise ValueError('size of region not defined')
        return self.x-self.__dict__[self.size[0]]/2.

    @property
    def xmax(self):
        if self.size is None:
            raise ValueError('size of region not defined')
        return self.x+self.__dict__[self.size[0]]/2.

    @property
    def ymin(self):
        if self.size is None:
            raise ValueError('size of region not defined')
        return self.y-self.__dict__[self.size[1]]/2.

    @property
    def ymax(self):
        if self.size is None:
            raise ValueError('size of region not defined')
        return self.y+self.__dict__[self.size[1]]/2.

    def __repr__(self):
        return '<'+super(Region, self).__repr__().split()[0].split('.')[-1]+'('+self.__str__().split('(')[1]+'>'

    def __str__(self):
        par = []
        for k in self.parname:
            par.append(self.__dict__[k])
        return str(self.shape)+str(tuple(par))

    def show(self, ds9=None, frame=None):
        if ds9 is None:
            ds9 = self.ds9
        if ds9 is None:
            raise ValueError('DS9 window is not specificed')
        ds9 = getds9(ds9)
        if frame is None:
            frame = self.frame
        if frame is None:
            frame = ds9.get('frame')
        ds9.set('frame '+frame)
        par = []
        for k in self.parname:
            par.append(self.__dict__[k])
            if self.zerobased:
                if k in ['x','y']:
                    par[-1] += 1
        propstr = ''
        for k in list(self.specs.keys()):
            propstr = propstr+' '+k+'='+str(self.specs[k])
        propstr = '#'+propstr
        ds9.set('regions', 'image; '+self.shape+' '+' '.join(str(par)[1:-1].split(','))+propstr)


class CircularRegion(Region):
    '''DS9 circular region class'''
    parname = ('x','y','r')
    _shape = 'circle'
    size = ('r', 'r')


class EllipseRegion(Region):
    '''DS9 ellipse region class'''
    parname = ('x','y','a','b','angle')
    _shape = 'ellipse'
    size = ('a','b')


class BoxRegion(Region):
    '''DS9 box region class'''
    parname = ('x', 'y', 'a', 'b', 'angle')
    _shape = 'box'
    size = ('a','b')


class AnnulusRegion(Region):
    '''DS9 annulus region class'''
    parname = ('x', 'y', 'r_in', 'r_out')
    _shape = 'annulus'
    size = ('r_out', 'r_out')


import pyds9
class DS9(pyds9.DS9):
    '''Extended pyds9.DS9 class.'''

    def __init__(self, restore=None, **kwargs):
        super(DS9, self).__init__(**kwargs)
        if restore is not None:
            from os.path import isfile
            if not isfile(restore):
                raise Warning('restoration file '+restore+' not found')
            else:
                self.restore(restore)

    @property
    def frames(self):
        return self.get('frame all').split()

    @property
    def actives(self):
        return self.get('frame active').split()

    def cursor(self, coord='image', value=False):
        '''Return cursor position (y, x) in 0-based indices

        x, y = cursor()'''
        x, y = self.get('imexam coordinate '+coord).split()
        if value:
            return float(x)-1, float(y)-1, float(self.get(' '.join(['data', coord, x, y, '1 1 yes'])))
        else:
            return float(x)-1, float(y)-1

    def get_arr2np(self):
        '''Replacement of the original pyds9.DS9.get_arr2np(), which seems
        to return a float32 array with bytes swapped, and the image size
        corrected.'''
        im = super(DS9, self).get_arr2np().byteswap()
        return im.reshape(*im.shape[::-1])

    def xpa(self):
        '''Interactive XPA command session

        Example:

        >>> Enter XPA command: get frame #  # print current frame number
        >>> Enter XPA command: set frame next  # set next frame active
        >>> Enter XPA command: quit   # or q, quick XPA session'''
        import sys
        while True:
            sys.stdout.write('>>> Enter XPA command: ')
            xpa = sys.stdin.readline().strip('\n')#('XPA command: ')
            if xpa in ['quit', 'q']:
                break
            elif xpa.startswith('get'):
                cmd = xpa[xpa.find(' ')+1:]
                try:
                    print((self.get(cmd)))
                except:
                    print('Invalid XPA command')
            elif xpa.startswith('set'):
                cmd = xpa[xpa.find(' ')+1:]
                try:
                    r = self.set(cmd)
                except:
                    print('Invalid XPA command')
                if r != 1:
                    print(('Error in executing XPA command "'+xpa+'"'))
            else:
                print('Invalid XPA command')

    def _collect_pars(self):
        fno0 = self.get('frame')
        fnos = []
        if not hasattr(self, 'data'):
            self.data = {}
        self.set('frame first')
        while True:
            n = self.get('frame')
            if n in fnos:
                break
            fnos.append(n)
            if not hasattr(self.data, n):
                self.data[n] = {}
            self.data[n]['data'] = self.get_arr2np()
            self.data[n]['shift'] = [0.,0.]
            self.data[n]['rotate'] = 0.
            self.set('frame next')
        self.set('frame '+str(fno0))

    def interactive(self):
        '''Start an interactive session

        Commands:
          c : create new frame
          d : delete current frame
          f : zoom to fit
          option h : open header dialog window
          i : zoom in by a factor of 2
          m : match image coordinate, scale, and colorbar with current frame
          n : next frame
          o : zoom out by a factor of 2
          p : previous frame
          option p : open pan zoom rotate dialog window
          q : quit interactive session
          r : rotate image 1 deg in ccw direction
          shift r : rotate image 1 deg in cw direction
          option s : open scale dialog window
          shift x : XPA command window in Python
          arrow keys : shift image by 1 pixel
        '''
        self._collect_pars()
        shift = False
        option = False
        while True:
            k = self.get('imexam any coordinate image').split()[0]
            if k.startswith('Shift'):
                shift = True
            elif k.startswith('Mode'):
                option = True
            elif k == 'c':
                self.set('frame new')
            elif k == 'd':
                self.set('frame delete')
            elif k == 'f':
                self.set('zoom to fit')
            elif option and k == 'h':
                self.set('header')
                option = False
            elif k == 'i':
                self.set('zoom 2')
            elif k == 'm':
                self.set('match frame image')
                self.set('match scale')
                self.set('match colorbar')
            elif k == 'n':
                self.set('frame next')
            elif k == 'o':
                self.set('zoom 0.5')
            elif not option and k == 'p':
                self.set('frame prev')
                option = False
            elif option and k == 'p':
                self.set('pan open')
                option = False
            elif k == 'q':
                break
            elif not shift and k == 'r':
                self.set('rotate +1')
                self.data[self.get('frame')]['rotate'] += 1
            elif shift and k == 'r':
                self.set('rotate -1')
                self.data[self.get('frame')]['rotate'] -= 1
                shift = False
            elif option and k == 's':
                self.set('scale open')
                option = False
            elif shift and k == 'x':
                self.xpa()
                shift = False
            elif k == 'Right':
                self.set_np2arr(shift(self.get_arr2np(),(0,1)))
                self.data[self.get('frame')]['shift'][1] += 1
            elif k == 'Left':
                self.set_np2arr(shift(self.get_arr2np(),(0,-1)))
                self.data[self.get('frame')]['shift'][1] -= 1
            elif k == 'Up':
                self.set_np2arr(shift(self.get_arr2np(),(1,0)))
                self.data[self.get('frame')]['shift'][0] += 1
            elif k == 'Down':
                self.set_np2arr(shift(self.get_arr2np(),(-1,0)))
                self.data[self.get('frame')]['shift'][0] -= 1

    def imdisp(self, im, ext=None, par=None, newframe=True, verbose=True):
        '''Display images.

        Parameters
        ----------
        im : string or string sequence, 2-D or 3-D array-like numbers
          File name, sequence of file names, image, or stack of images.  For
          3-D array-like input, the first dimension is the dimension of stack
        ext : non-negative integer, optional
          The extension to be displayed
        newframe : bool, optional
          If set `False`, then the image will be displayed in the currently
          active frame in DS9, and the previous image will be overwritten.
          By default, a new frame will be created to display the image.
        par : string, or list of string, optional
          The display parameters for DS9.  See DS9 document.
        verbose : bool, optional
          If `False`, then all print out is suppressed.

        Returns: int or list(int)
          The status code:
          0 - no error
          1 - image file not found
          2 - extension not existent
          3 - invalid image extension
          13 - invalid PDS format
          14 - invalid FITS format
          15 - Unrecognized extension

        v1.0.0 : JYL @PSI, 2/14/2015, adopted from the standalone imdisp()
        '''

        # Pre-process for the case of a single image
        from astropy import nddata
        if isinstance(im, (str,bytes)):
            ims = [im]
        elif isinstance(im, np.ndarray):
            if im.ndim == 2:
                ims = [im]
            else:
                ims = im
        elif isinstance(im, nddata.NDData):
            ims = [im]
        else:
            ims = im

        # Loop through all images
        if len(ims) > 1:
            self.set('tile')
        st = []
        for im in ims:
            if newframe:
                self.set('frame new')
            # Display image(s)
            if isinstance(im, str):
                if im.split('[')[0].lower().endswith(('.fits','.fit','fz')):
                    try:
                        if ext is None:
                            print('fits {0}'.format(im))
                            tmp = self.set('fits {0}'.format(im))
                        else:
                            print('fits {0}[{1}]'.format(im,ext))
                            tmp = self.set('fits {0}[{1}]'.format(im,ext))
                        st.append(0)
                    except ValueError:
                        from os.path import isfile
                        if not isfile(im):
                            if verbose:
                                print()
                                print('File does not exist: {0}'.format(im))
                            st.append(1)
                        elif ext is None:
                            if verbose:
                                print()
                                print('Invalid FITS format')
                            st.append(14)
                        else:
                            from astropy.io import fits
                            info = fits.info(im,output=False)
                            if ext >= len(info):
                                if verbose:
                                    print()
                                    print(('Error: Extension '+repr(ext)+' does not exist!'))
                                st.append(2)
                            elif (info[ext][3] in ('ImageHDU','CompImageHDU')) and (len(info[ext][5])>1):
                                if verbose:
                                    print()
                                    print(('Error: Extension '+repr(ext)+' contains no image!'))
                                    print()
                                st.append(3)
                            print((fits.info(im)))
                elif im.lower().endswith('.img'):
                    from .PDS import readpds
                    try:
                        self.set_np2arr(np.asarray(readpds(im)).astype('f4'))
                        st.append(0)
                    except:
                        if verbose:
                            print()
                            print('Invalid PDS format')
                        st.append(13)
                else:
                    if verbose:
                        print()
                        print('Unrecognized extension')
                    st.append(15)
            else:
                self.set_np2arr(np.asarray(im).astype('f4'))
                st.append(0)

            # set DS9 parameters
            if par is not None:
                self.sets(par)

        if len(st) == 1:
            st = st[0]
        return st


    def multiframe(self, fitsfile):
        '''Display multiframe FITS'''
        self.set('multiframe '+fitsfile)


    def region(self, frame=None, system='image', zerobased=True):
        '''Returns a list of regions already defined in the frame

        Note: the keyword `zerobased` controls the coordinate indexing
        convention.  DS9 convention is 1-based, but Python convention
        is 0-based!'''
        if frame is not None:
            fno0 = self.get('frame')
            self.set('frame '+str(frame))
        cf = self.get('frame')
        sys0 = self.get('region system')
        if sys0 != system:
            self.set('region system '+system)
        regstr = self.get('region -format saoimage')
        if regstr == []:
            return []
        else:
            regstr = regstr.split()
        reg = []
        for r in regstr:
            shape = r[:r.find('(')]
            pars = np.fromstring(r[r.find('('):].strip('()'),sep=',',dtype=float)
            if zerobased:
                pars[:2] -= 1
            if shape == 'circle':
                reg.append(CircularRegion(*pars, ds9=self, frame=cf, zerobased=zerobased))
            elif shape == 'ellipse':
                reg.append(EllipseRegion(*pars, ds9=self, frame=cf, zerobased=zerobased))
            elif shape == 'box':
                reg.append(BoxRegion(*pars, ds9=self, frame=cf, zerobased=zerobased))
            elif shape == 'annulus':
                reg.append(AnnulusRegion(*pars, ds9=self, frame=cf, zerobased=zerobased))
            else:
                reg.append({'shape': shape, 'pars': pars})
        if frame is not None:
            self.set('frame '+fno0)
        self.set('region system '+sys0)
        return reg

    def aperture(self, frame=None, zerobased=True):
        '''Extract apertures from circular or annulus regions in
        a list of photutils.Aperture instances'''
        reg = self.region(frame=frame, system='image', zerobased=zerobased)
        if reg == []:
            return None
        apts = []
        for r in reg:
            if isinstance(r, Region):
                if r.shape == 'circle':
                    apts.append(P.CircularAperture((r.x,r.y), r.r))
                elif r.shape == 'annulus':
                    apts.append(P.CircularAnnulus((r.x,r.y), r.r_in, r.r_out))
                else:
                    pass
            else:
                pass
        return apts

    def define_aperture(self, radius=3., centroid=False, **kwargs):
        '''Interactively define apertures

        Method takes the keywords accepted by centroid(), except for `center',
        for centroiding.
        '''
        import jylipy
        import photutils as P
        tmp = kwargs.pop('center', None)
        verbose = kwargs.pop('verbose', True)
        if not hasattr(radius, '__iter__'):
            radius = [radius]
        nr = len(radius)
        aperture = []
        i = 0
        while i<nr:
            print()
            print('Press q to exit')
            print('Left click in the image to define aperture center')
            print()
            key = self.get('imexam any coordinate image').split()
            if key[0] == 'q':
                break
            if len(key) == 3:
                x, y = key[1:]
                x, y = float(x), float(y)
                if centroid:
                    y, x = jylipy.centroid(self.get_arr2np(),center=[y,x],verbose=verbose)
                aperture.append(P.CircularAperture((x,y), radius[i%nr]))
                self.set('regions','image; circle('+','.join([str(x),str(y),str(radius[i%nr])])+')')
                i += 1
                print(('Aperture 1: ({0}, {1})'.format(x, y)))
                print()
        return aperture

    def apphot(self, **kwargs):
        from .core import apphot
        return apphot(ds9=self, **kwargs)

    def show_aperture(self, aperture, frame=None, zerobased=True):
        '''Show aperture as region'''

        if frame is not None:
            fno0 = self.get('frame')
            if str(frame) != fno0:
                self.set('frame '+str(frame))

        if not isinstance(aperture, list):
            if isinstance(aperture, P.Aperture):
                pos = aperture.positions
                if hasattr(aperture, 'r'):  # circular aperture
                    for x, y in pos:
                        if zerobased:
                            x, y = x+1, y+1
                        self.set('regions', 'image; circle('+','.join([str(x),str(y),str(aperture.r)])+')')
                elif hasattr(aperture, 'r_in'):  # annulus aperture
                    for x, y in pos:
                        if zerobased:
                            x, y = x+1, y+1
                        self.set('regions', 'image; annulus('+','.join([str(x),str(y),str(aperture.r_in),str(aperture.r_out)])+')')
                else:
                    pass
            else:
                l = len(aperture)
                if l == 3:  # circular aperture
                    x,y,r = aperture
                    self.set('regions', 'image; circle('+','.join([str(x),str(y),str(r)])+')')
                elif l == 4:  # annulus aperture
                    x,y,r1,r2 = aperture
                    self.set('regions', 'image; annulus('+','.join([str(x),str(y),str(r1),str(r2)])+')')
                else:
                    pass
        else:
            for apt in aperture:
                self.show_aperture(apt)

        if frame is not None:
            self.set('frame '+fno0)

    def sets(self, par, buf=None, blen=-1):
        '''XPA set that accepts a single command line or an array of lines'''
        if isinstance(par, str):
            return self.set(par, buf=buf, blen=blen)
        else:
            st = []
            if not hasattr(buf, '__iter__'):
                buf = [buf]*len(par)
            if not hasattr(blen, '__iter__'):
                blen = [blen]*len(par)
            for p, b, l in zip(par, buf, blen):
                st.append(self.set(p, b, l))
            return st

    def gets(self, par=None):
        '''XPA get that accepts a single command line or an array of lines'''
        if isinstance(par, str) or (par is None):
            return self.get(par)
        else:
            out = []
            for p in par:
                out.append(self.get(p))
            return out

    def backup(self, bckfile):
        '''Backup the current session'''
        self.set('backup '+bckfile)

    def restore(self, bckfile):
        '''Restore DS9 session'''
        self.set('restore '+bckfile)

    def saveimage(self, outfile, all=False):
        '''Save frame(s) to images.

        Parameters
        ----------
        outfile : str
          The full name of output file.  If multiple frames
          are to be saved, then the sequence number will be
          inserted right before the name extension, starting
          from 0.

        If `saveall` = True, then all active frames will be saved,
        with the current frame the first.

        v1.0.0 : 5/8/2015, JYL @PSI
        '''
        if len(outfile.split('.')) < 2:
            raise ValueError('The format of image file is not specified.  Please include an extension in the file name.')

        from os.path import basename
        if all:
            nfm = len(self.n_actives)
            tmp = outfile.split('.')
            fmtstr = '.'.join(tmp[:-1])+'_%0'+repr(int(np.ceil(np.log10(nfm))))+'d'+'.'+tmp[-1]
            for i in range(nfm):
                self.set('saveimage '+fmtstr % i)
                self.set('frame next')
        else:
            self.set('saveimage '+outfile)

    def setall(self, cmd, all=False):
        '''Set XPA command(s) to all active frames

        v1.0.0 : 5/8/2015, JYL @PSI
        '''
        if all:
            frm = self.frames
        else:
            frm = self.actives
        cf = self.get('frame')
        for f in frm:
            self.set('frame '+f)
            self.sets(cmd)
        self.set('frame '+cf)


def getds9(ds9=None, new=False, restore=None):
    '''Return a DS9 instance associated with a DS9 window.

    Parameters
    ----------
    ds9 : str, pyds9.DS9 instance, optional
      The ID of DS9 window.
    new : bool, optional
      If True, then a new window is openned unless `ds9` specifies
      an existing window.
    restore : str
      File name of the previously saved sessions to restore

    Returns
    -------
    A DS9 instance.

    If `ds9` is None, then either a new DS9 window is created and the
    associated DS9 instance is returned (`new`==True or no DS9 window
    is open), or the existing DS9 window that is openned the first is
    assicated with the returned DS9 instance (`new`=False).

    If `ds9` is specified, then the DS9 window with the same ID will
    be associated with the returned DS9 instance, or a new window with
    the specified ID will be opened.

    v1.0.1 : 5/8/2015, JYL @PSI
      Added keyword parameter `restore`
    '''
    if ds9 is not None:
        ds9_id = ds9
    else:
        ds9_id = None

    import pyds9
    if isinstance(ds9_id, pyds9.DS9):
        return ds9_id

    targs = pyds9.ds9_targets()
    if targs is not None:
        targs = [x.split()[0].split(':')[1] for x in targs]
    if ds9_id is None:
        if targs is None:
            return DS9(restore=restore)
        elif new:
            i = 1
            newid = 'ds9_'+str(i)
            while newid in targs:
                i += 1
                newid = 'ds9_'+str(i)
            return DS9(restore=restore, target=newid)
        else:
            return DS9(restore=restore, target=targs[0])
    else:
        return DS9(restore=restore, target=ds9_id)

