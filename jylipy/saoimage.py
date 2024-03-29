'''Extended DS9 class moduel'''


import numpy as np
__all__ = ['Region', 'CircularRegion', 'EllipseRegion', 'BoxRegion',
           'AnnulusRegion', 'VectorRegion', 'TextRegion', 'PointRegion',
           'XPointRegion', 'DS9', 'getds9']

class Region(object):
    '''Base class for DS9 regions'''

    parname = None
    _shape = None

    def __init__(self, *args, **kwargs):
        '''The initialization of Region class depends on its shape.
        The general form is:

        r = Region(*par, shape='shape name', parname=('x', 'y', ...))

        *par : numbers
            Parameters for shape, same definition and order as in
            DS9.get('regions -format saoimage')
        shape : str
            Shape of region, same as defined in DS9
        parname : tuple of str
            The name of parameters

        It can also take keyword arguments to specify the properties
        of region, such as color, width, text, etc.


        To subclass Region for specific shape:

        class BoxRegion(Region):
            parname = ('x', 'y', 'a', 'b', 'angle')
            _shape = 'box'

        The initialization of subclass is now simply:

        r = BoxRegion(x, y, a, b, angle)
        '''
        if self.parname is None:
            self.parname = kwargs.pop('parname', ('x', 'y'))
        if self._shape is None:
            self._shape = kwargs.pop('shape', None)
        if len(self.parname) != len(args):
            raise TypeError('{0}.__init__() takes {1} arguments ({2} given)'.
                format(type(self),len(self.parname)+1,len(args)+1))
        self.frame = kwargs.pop('frame', None)
        self.ds9 = kwargs.pop('ds9', None)
        self._zerobased = kwargs.pop('zerobased', True)
        for i in range(len(args)):
            self.__dict__[self.parname[i]] = args[i]
        if self._shape is None:
            raise Warning('region shape is not defined')
        self.specs = {}
        self.specs['color'] = kwargs.pop('color', 'green')
        self.specs['width'] = kwargs.pop('width', 1)
        for k, v in kwargs.items():
            self.specs[k] = v

    @property
    def zerobased(self):
        return self._zerobased

    @zerobased.setter
    def zerobased(self, v):
        if self.zerobased ^ v:
            self._zerobased = v
            offset = -1 if v else 1
            for k in self.parname:
                if k[0] in ['x', 'y']:
                    setattr(self, k, getattr(self, k) + offset)

    @property
    def shape(self):
        return self._shape

    @property
    def par(self):
        v = []
        for k in self.parname:
            v.append(self.__dict__[k])
        return v

    def __repr__(self):
        return '<'+super(Region, self).__repr__().split()[0].split('.')[-1]+ \
            '('+self.__str__().split('(')[1]+'>'

    def __str__(self):
        par = []
        for k in self.parname:
            par.append(self.__dict__[k])
        return str(self.shape)+str(tuple(par))

    def show(self, ds9=None, frame=None, print_cmd=False):
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
                if k[0] in ['x', 'y']:
                    par[-1] += 1
        propstr = ''
        for k, v in self.specs.items():
            vstr = '"'+str(v)+'"' if isinstance(v, (str, bytes)) else str(v)
            #vstr = str(v)
            propstr = propstr + ' {}={}'.format(k, vstr)
        propstr = '#'+propstr
        cmdstr = 'image; {} {} {}'.format(self.shape,
                ' '.join(str(par)[1:-1].split(',')), propstr)
        if print_cmd:
            print("ds9.set('regions', '{}')".format(cmdstr))
        else:
            ds9.set('regions', cmdstr)


class CircularRegion(Region):
    '''DS9 circular region class'''
    parname = ('x','y','r')
    _shape = 'circle'

    @property
    def extent(self):
        """(xmin, xmax, ymin, ymax)"""
        return np.array([self.par[0] - self.par[2], \
                         self.par[0] + self.par[2], \
                         self.par[1] - self.par[2], \
                         self.par[1] + self.par[2]])
    @property
    def area(self):
        return np.pi * self.r**2


class EllipseRegion(Region):
    '''DS9 ellipse region class'''
    parname = ('x','y','a','b','angle')
    _shape = 'ellipse'

    @property
    def area(self):
        return np.pi * self.a * self.b


class BoxRegion(Region):
    '''DS9 box region class'''
    parname = ('x', 'y', 'a', 'b', 'angle')
    _shape = 'box'

    def _extent(self):
        a = np.deg2rad(self.par[4])
        cosa = np.cos(a)
        sina = np.sin(a)
        dx = self.par[2] * cosa - self.par[3] * sina
        dy = self.par[2] * sina + self.par[3] * cosa
        return dx, dy

    @property
    def corners(self):
        """array of shape (4, 2), coordinates of four corners"""
        a = np.deg2rad(self.par[4])
        cosa = np.cos(a)
        sina = np.sin(a)
        dx1 = self.par[2] * cosa - self.par[3] * sina
        dy1 = self.par[2] * sina + self.par[3] * cosa
        dx2 = - self.par[2] * cosa - self.par[3] * sina
        dy2 = - self.par[2] * sina + self.par[3] * cosa
        c1 = [self.par[0] + dx1/2, self.par[1] + dy1/2]
        c2 = [self.par[0] + dx2/2, self.par[1] + dy2/2]
        c3 = [self.par[0] - dx1/2, self.par[1] - dy1/2]
        c4 = [self.par[0] - dx2/2, self.par[1] - dy2/2]
        return np.array([c1, c2, c3, c4])

    @property
    def extent(self):
        """(xmin, xmax, ymin, ymax)"""
        c = self.corners
        return np.array([c[:, 0].min(), c[:, 0].max(),
                         c[:, 1].min(), c[:, 1].max()])

    @property
    def area(self):
        return self.a * self.b


class AnnulusRegion(Region):
    '''DS9 annulus region class'''
    parname = ('x', 'y', 'r_in', 'r_out')
    _shape = 'annulus'

    @property
    def area(self):
        return np.pi * (self.r_out**2 - self.r_in**2)


class VectorRegion(Region):
    """Vecotor region"""
    parname = ('x', 'y', 'length', 'angle')
    _shape = 'vector'


class TextRegion(Region):
    """Text region"""
    parname = ('x', 'y')
    _shape = 'text'


class ProjectionRegion(Region):
    parname = ('x1', 'y1', 'x2', 'y2', 'width')
    _shape = 'projection'


class LineRegion(Region):
    parname = ('x1', 'y1', 'x2', 'y2')
    _shape = 'line'

    @property
    def angle(self):
        """
        Angle of line with respect to +x axis towards +y axis in degrees
        """
        return np.rad2deg(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))

    @property
    def length(self):
        """Length of the line segment
        """
        return np.sqrt((self.y2 - self.y1)**2 + (self.x2 - self.x1)**2)

    @property
    def parameters(self):
        """Parameters of line equation

        Line equation has the form:
            a * x + b * y + 1 = 0

        Parameters attribute is an array of [a, b]
        """
        dy = self.y2 - self.y1
        dx = self.x2 - self.x1
        c = dx * self.y1 - dy * self.x1
        return np.array([dy / c, -dx / c])

    def getx(self, y):
        """Given a `y`, return an `x` such that (x, y) is on the line
        """
        p = self.parameters
        return (-1 - p[1] * y) / p[0]

    def gety(self, x):
        """Given an `x`, return a `y` such that (x, y) is on the line
        """
        p = self.parameters
        return (-1 - p[0] * x) / p[1]

    def intersect(self, line):
        """Return the coordinate (x, y) of the intersection with another line

        Parameters
        ----------
        line : LineRegion
            The line to intersect.

        returns
        -------
        [x, y] : 2-element array
            The coordinate of intersection.  If the two lines are parallel,
            then return [np.nan, np.nan]
        """
        p1 = self.parameters
        p2 = line.parameters
        dom = p1[0] * p2[1] - p2[0] * p1[1]
        if dom == 0:
            return np.array([np.nan, np.nan])
        return np.array([(p1[1] - p2[1]) / dom, (p2[0] - p1[0]) / dom])

    def to_vector(self, start=0, **kwargs):
        """Convert to a VectorRegion

        Parameters
        ----------
        start : 0 or 1
            Start point of the vector.  0 represents (self.x1, self.y1),
            1 represents (self.x2, self.y2)
        kwargs : dict
            Keyword parameters to pass to VectorRegion object
        Returns
        -------
        VectorRegion
        """
        if start == 0:
            x = self.x1
            y = self.y1
            angle = self.angle
        elif start == 1:
            x = self.x2
            y = self.y2
            angle = (self.angle + 180) % 360
        specs = self.specs
        keys_to_remove = ['dashlist', 'select', 'line']
        for k in keys_to_remove:
            specs.pop(k, None)
        specs['vector'] = 1
        zerobased = kwargs.pop('zerobased', self.zerobased)
        for k, v in kwargs.items():
            specs[k] = v
        vect = VectorRegion(x, y, self.length, angle,
                            zerobased=self.zerobased, **specs)
        vect.zerobased = zerobased
        return vect


class SegmentRegion(Region):
    """Segment region
    """
    _shape = ''

    def __init__(self, points, **kwargs):
        """
        points : 2D array of shape (2, N)
            Points along the segment line.  the x- and y-coordinates are
            in points[0, :], points[1, :], respectively.  N is the number
            of points on the sement line.
        """
        super().__init__(0, 0, **kwargs)
        self.parname = None
        self.points = np.asarray(points)

    @property
    def zerobased(self):
        return self._zerobased

    @zerobased.setter
    def zerobased(self, v):
        if self.zerobased ^ v:
            self._zerobased = v
            offset = -1 if v else 1
            self.points += offset

    def show(self, ds9=None, frame=None, print_cmd=False):
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

        par = self.points.T.flatten()
        if self.zerobased:
            par += 1
        parstr = '# segment(' + ', '.join([str(x) for x in par]) + ') '
        propstr = ''
        for k, v in self.specs.items():
            vstr = '"'+str(v)+'"' if isinstance(v, (str, bytes)) else str(v)
            #vstr = str(v)
            propstr = propstr + ' {}={}'.format(k, vstr)
        cmdstr = 'image; ' + parstr + propstr
        if print_cmd:
            print("ds9.set('regions', '{}')".format(cmdstr))
        else:
            ds9.set('regions', cmdstr)


class PointRegion(Region):
    """DS9 point region group"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape = self._shape + ' point'


class XPointRegion(PointRegion):
    _shape = 'X'


class CrossPointRegion(PointRegion):
    _shape = 'Cross'


class CirclePointRegion(PointRegion):
    _shape = 'Circle'


class BoxPointRegion(PointRegion):
    _shape = 'Box'


class DiamondPointRegion(PointRegion):
    _shape = 'Diamond'


class ArrowPointRegion(PointRegion):
    _shape = 'Arrow'


class BoxCirclePointRegion(PointRegion):
    _shape = 'BoxCircle'


class RegionList(list):
    """Region list class"""

    _mapping = {'circle': CircularRegion,
                'ellipse': EllipseRegion,
                'box': BoxRegion,
                'annulus': AnnulusRegion,
                'vector': VectorRegion,
                'text': TextRegion,
                'projection': ProjectionRegion,
                'line': LineRegion,
                'x_point': XPointRegion,
                'circle_point': CirclePointRegion,
                'box_point': BoxPointRegion,
                'diamond_point': DiamondPointRegion,
                'cross_point': CrossPointRegion,
                'arrow_point': ArrowPointRegion,
                'boxcircle_point': BoxCirclePointRegion}

    @classmethod
    def from_ds9(cls, d, frame=None, system='image', zerobased=False):
        """Return a list of region objects from DS9 window

        d : `DS9`
            The DS9 window to collect region objects from
        """
        obj = cls()
        if frame is not None:
            fno0 = d.get('frame')
            d.set('frame '+str(frame))
        cf = d.get('frame')
        sys0 = d.get('region system')
        if sys0 != system:
            d.set('region system '+system)
        regstr = d.get('regions -format ds9').strip().split('\n')

        gs = {}
        global_specs = regstr[1].split(' ')
        if global_specs[0] == 'global':
            for i in range(1, len(global_specs)):
                if global_specs[i].find('=') != -1:
                    k, v = global_specs[i].split('=')
                    if i < len(global_specs):
                        while (i<len(global_specs)-1) and \
                                (global_specs[i+1].find('=') == -1):
                            i += 1
                            v = ' '.join([v, global_specs[i]])
                    try:
                        v = int(v)
                    except ValueError:
                        pass
                    gs[k] = v
        obj.global_specs = gs
        obj.global_specs['frame'] = cf
        obj.global_specs['zerobased'] = False
        if len(regstr) > 2:
            obj.global_specs['system'] = regstr[2]
        if len(regstr) > 3:
            for s in regstr[3:]:
                spec = obj.global_specs.copy()
                spec['ds9'] = d
                s = s.split('#')
                if s[0] == '':
                    s = s[1].strip().split(')')
                    s[0] = s[0] + ')'
                    if s[1] == '':
                        _ = s.pop()
                shape, par = s[0].strip().split('(')
                par = eval('(' + par)
                if len(s) > 1:
                    s = s[1].strip()
                    if s.find('font') != -1:
                        t = s.split('"')
                        t[0] = t[0].replace('font=', '')
                        s = t[0].strip().split(' ') + t[2].strip().split(' ') + ['font="'+t[1]+'"']
                    elif s.find('line') != -1:
                        w = s.find('line')
                        s = '_'.join([s[:w+6], s[w+7:]])
                        w1 = s.rfind('line')
                        if (w1 != -1) and (w1 != w):
                            s = '_'.join([s[:w1+6], s[w1+7:]])
                        s = s.split(' ')
                    else:
                        s = s.split(' ')
                    for sp in s:
                        #print(sp)
                        k, v = sp.split('=')
                        spec[k] = v
                    if 'line' in spec:
                        spec['line'] = ' '.join(spec['line'].split('_'))
                    #if 'text' in spec.keys():
                    #    spec['text'] = spec['text'].strip('{').strip('}')
                if shape == 'point':
                    shape = spec['point'] + '_' + 'point'
                    _ = spec.pop('point')
                if shape in obj._mapping:
                    obj.append(obj._mapping[shape](*par, **spec))
                else:
                    obj.append('{} {} {}'.format(shape, str(par), str(spec)))
                #print()
            if zerobased:
                for r in obj:
                    r.zerobased = True
        return obj

    def show(self, **kwargs):
        for r in self:
            r.show(**kwargs)


import pyds9
class DS9(pyds9.DS9):
    '''Extended pyds9.DS9 class.'''

    def __init__(self, restore=None, **kwargs):
        self.zerobased = kwargs.pop('zerobased', True)
        super(DS9, self).__init__(**kwargs)
        if restore is not None:
            from os.path import isfile
            if not isfile(restore):
                raise Warning('restoration file '+restore+' not found')
            else:
                self.restore(restore)

    @property
    def width(self):
        """DS9 window width"""
        return int(self.get('width'))

    @width.setter
    def width(self, value):
        self.set('width {}'.format(value))

    @property
    def height(self):
        """DS9 window height"""
        return int(self.get('height'))

    @height.setter
    def height(self, value):
        self.set('height {}'.format(value))

    @property
    def frames(self):
        return self.get('frame all').split()

    @property
    def actives(self):
        return self.get('frame active').split()

    @property
    def pan(self):
        ct = np.float32(self.get('pan').split())
        if self.zerobased:
            ct -= 1
        return ct

    @pan.setter
    def pan(self, v):
        v = np.array(v)
        if self.zerobased:
            v += 1
        self.set('pan to {} {}'.format(v[0], v[1]))

    @property
    def zoom(self):
        z = np.float32(self.get('zoom').split())
        if len(z) == 1:
            return z[0]
        else:
            return z

    @zoom.setter
    def zoom(self, v):
        if (not hasattr(v, '__iter__')):
            self.set('zoom to {}'.format(v))
        elif  len(v) == 1:
            self.set('zoom to {}'.format(v[0]))
        else:
            self.set('zoom to {} {}'.format(v[0], v[1]))

    @property
    def rotate(self):
        return float(self.get('rotate'))

    @rotate.setter
    def rotate(self, v):
        self.set('rotate to {}'.format(v))

    @property
    def cmap(self):
        return self.get('cmap')

    @cmap.setter
    def cmap(self, v):
        self.set('cmap {}'.format(v))

    @property
    def cmap_value(self):
        return np.float32(self.get('cmap value').split())

    @cmap_value.setter
    def cmap_value(self, v):
        self.set('cmap value {} {}'.format(v[0], v[1]))

    @property
    def scale(self):
        return self.get('scale')

    @scale.setter
    def scale(self, v):
        self.set('scale {}'.format(v))

    @property
    def scale_limits(self):
        return np.float32(self.get('scale limits').split())

    @scale_limits.setter
    def scale_limits(self, v):
        self.set('scale limits {} {}'.format(v[0], v[1]))

    def zoomin(self):
        self.set('zoom 2')

    def zoomout(self):
        self.set('zoom 0.5')

    def zoomfit(self):
        self.set('zoom to fit')

    def cursor(self, coord='image', value=False):
        '''Return cursor position (y, x) in 0-based indices

        x, y = cursor()'''
        x, y = self.get('imexam coordinate '+coord).split()
        if value:
            return float(x)-1, float(y)-1, \
                float(self.get(' '.join(['data', coord, x, y, '1 1 yes'])))
        else:
            return float(x)-1, float(y)-1

    #def get_arr2np(self):
        '''Replacement of the original pyds9.DS9.get_arr2np(), which seems
        to return a float32 array with bytes swapped, and the image size
        corrected.'''
        #im = super(DS9, self).get_arr2np().byteswap()
        #return im.reshape(*im.shape[::-1])

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
                self.zoomfit()
            elif option and k == 'h':
                self.set('header')
                option = False
            elif k == 'i':
                self.zoomin()
            elif k == 'm':
                self.set('match frame image')
                self.set('match scale')
                self.set('match colorbar')
            elif k == 'n':
                self.set('frame next')
            elif k == 'o':
                self.zoomout()
            elif not option and k == 'p':
                self.set('frame prev')
                option = False
            elif option and k == 'p':
                self.set('pan open')
                option = False
            elif k == 'q':
                break
            elif not shift and k == 'r':
                self.rotate += 1
                self.data[self.get('frame')]['rotate'] += 1
            elif shift and k == 'r':
                self.rotate -= 1
                self.data[self.get('frame')]['rotate'] -= 1
                shift = False
            elif option and k == 's':
                self.set('scale open')
                option = False
            elif shift and k == 'x':
                self.xpa()
                shift = False
            elif k == 'Right':
                self.set('pan -1, 0')
                self.data[self.get('frame')]['shift'][1] += 1
            elif k == 'Left':
                self.set('pan 1 0')
                self.data[self.get('frame')]['shift'][1] -= 1
            elif k == 'Up':
                self.set('pan 0 -1')
                self.data[self.get('frame')]['shift'][0] += 1
            elif k == 'Down':
                self.set('pan 0 1')
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
          4 - invalid PDS format
          5 - invalid FITS format
          6 - Unrecognized FITS extension
          7 - Unrecognized FITS extension error

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
            if isinstance(im, (str,bytes)):
                from os.path import isfile
                if not isfile(im):
                    if verbose:
                        print()
                        print('File does not exist: {0}'.format(im))
                    st.append(1)
                elif im.split('[')[0].lower().endswith(('.fits','.fit','fz')):
                    try:
                        if ext is None:
                            tmp = self.set('fits {0}'.format(im))
                        else:
                            tmp = self.set('fits {0}[{1}]'.format(im,ext))
                        st.append(0)
                    except ValueError:
                        if ext is None:
                            if verbose:
                                print()
                                print('Invalid FITS format')
                            st.append(5)
                        else:
                            from astropy.io import fits
                            info = fits.info(im,output=False)
                            if ext >= len(info):
                                if verbose:
                                    print()
                                    print('Error: Extension ' + repr(ext) \
                                        +' does not exist!')
                                st.append(2)
                            elif (info[ext][3] in ('ImageHDU','CompImageHDU'))\
                                    and (len(info[ext][5])>1):
                                if verbose:
                                    print()
                                    print('Error: Extension ' + repr(ext) \
                                            +' contains no image!')
                                    print()
                                st.append(3)
                            else:
                                if verbose:
                                    print()
                                    print('Unrecognized FITS extension error')
                                st.append(7)
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
                        st.append(4)
                else:
                    if verbose:
                        print()
                        print('Unrecognized extension')
                    st.append(6)
            else:
                self.set_np2arr(np.asarray(im).astype('f4'))
                st.append(0)

            # set DS9 parameters
            if st[-1] == 0:
                if par is not None:
                    self.set(par)
            else:
                self.set('frame delete')

        if len(st) == 1:
            st = st[0]
        return st

    def multiframe(self, fitsfile):
        '''Display multiframe FITS'''
        self.set('multiframe '+fitsfile)

    def region(self, **kwargs):
        '''Returns a list of regions already defined in the frame

        Note: the keyword `zerobased` controls the coordinate indexing
        convention.  DS9 convention is 1-based, but Python convention
        is 0-based!'''
        return RegionList.from_ds9(self, **kwargs)

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
                    y, x = jylipy.centroid(self.get_arr2np(), center=[y,x],
                            verbose=verbose)
                aperture.append(P.CircularAperture((x,y), radius[i%nr]))
                self.set('regions','image; circle('
                        + ','.join([str(x),str(y),str(radius[i%nr])])+')')
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
                        self.set('regions', 'image; circle(' \
                                + ','.join([str(x),str(y),str(aperture.r)]) \
                                +')')
                elif hasattr(aperture, 'r_in'):  # annulus aperture
                    for x, y in pos:
                        if zerobased:
                            x, y = x+1, y+1
                        self.set('regions', 'image; annulus(' \
                            + ','.join([str(x), str(y), str(aperture.r_in), \
                                        str(aperture.r_out)])+')')
                else:
                    pass
            else:
                l = len(aperture)
                if l == 3:  # circular aperture
                    x,y,r = aperture
                    self.set('regions', 'image; circle(' \
                            + ','.join([str(x),str(y),str(r)])+')')
                elif l == 4:  # annulus aperture
                    x,y,r1,r2 = aperture
                    self.set('regions', 'image; annulus(' \
                            + ','.join([str(x),str(y),str(r1),str(r2)])+')')
                else:
                    pass
        else:
            for apt in aperture:
                self.show_aperture(apt)

        if frame is not None:
            self.set('frame '+fno0)

    def set(self, par, buf=None, blen=-1):
        """XPA set that accepts a single command line or an array of lines
        """
        if isinstance(par, str):
            return super().set(par, buf=buf, blen=blen)
        else:
            st = []
            if not hasattr(buf, '__iter__'):
                buf = [buf]*len(par)
            if not hasattr(blen, '__iter__'):
                blen = [blen]*len(par)
            for p, b, l in zip(par, buf, blen):
                st.append(super().set(p, b, l))
            return st

    def get(self, par=None):
        """XPA get that accepts a single command line or an array of lines
        """
        if isinstance(par, str) or (par is None):
            return super().get(par)
        else:
            out = []
            for p in par:
                out.append(super().get(p))
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
            raise ValueError("The format of image file is not specified. "
                "Please include an extension in the file name.")

        from os.path import basename
        if all:
            frames = [int(x) for x in self.get('frame active').split()]
            current_frame = self.get('frame')
            from os.path import splitext
            root, ext = splitext(outfile)
            for i in frames:
                self.set('frame {}'.format(i))
                fmtstr = '_{:0' + \
                        '{}'.format(int(np.ceil(np.log10(max(frames))))) + '}'
                self.set('saveimage '+ root + fmtstr.format(i) + ext)
            self.set('frame {}'.format(current_frame))
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
            self.set(cmd)
        self.set('frame '+cf)


class DS9DisplayPar(dict):
    """DS9 display parameters"""

    @classmethod
    def from_ds9(cls, ds9, frame='current'):
        """Collect display parameters from DS9

        Parameters
        ----------
        ds9 : `pyds9.DS9`
            The DS9 window to collect parameters
        frame : 'current', 'all', int, str, int array, str array, optional
            The DS9 frames to be processed
        """
        par_names = ['scale', 'scale_limits', 'cmap', 'cmap_value',
                     'pan', 'zoom', 'rotate']
        obj = cls()
        obj.par_names = par_names
        if frame == 'all':
            obj['frame'] = np.array(ds9.frames)
            obj._len = len(obj['frame'])
        elif frame == 'current':
            obj['frame'] = ds9.get('frame')
            obj._len = 0
        else:
            if isinstance(frame, str) or (not hasattr(frame, '__iter__')):
                obj['frame'] = np.array([frame])
            else:
                obj['frame'] = np.array(frame)
            obj._len = len(obj['frame'])
        if obj._len == 0:
            for k in par_names:
                obj[k] = getattr(ds9, k)
        else:
            for k in par_names:
                obj[k] = []
            current_frame = ds9.get('frame')
            for f in obj['frame']:
                ds9.set('frame {}'.format(f))
                for k in par_names:
                    obj[k].append(getattr(ds9, k))
            for k in par_names:
                obj[k] = np.array(obj[k])
            ds9.set('frame {}'.format(current_frame))
        if (frame not in ['all', 'current']) \
                and (isinstance(frame, str) or \
                     (not hasattr(frame, '__iter__'))):
            for k, v in obj.items():
                obj[k] = v[0]
            obj._len = 0
        return obj

    @classmethod
    def from_table(cls, indata):
        """Collect display parameters from a table"""
        keys = []
        cols = []
        for k in indata.keys():
            try:
                _ = int(k.split('_')[-1])
                this_key ='_'.join(k.split('_')[:-1])
                if this_key in keys:
                    cols[-1].append(k)
                else:
                    keys.append(this_key)
                    cols.append([k])
            except ValueError:
                keys.append(k)
                cols.append([k])
        obj = cls()
        for k, c in zip(keys, cols):
            data = indata[c].as_array()
            obj[k] = np.squeeze(data.view(dtype=(data.dtype[0], np.shape(c))))
        obj._len = len(indata)
        if len(indata) == 1:
            for k in obj.keys():
                obj[k] = obj[k][0]
            obj._len = 0
        return obj

    @classmethod
    def from_csv(cls, file, **kwargs):
        """Collect display parameters from csv file

        Parameters
        ----------
        file : str
            Input csv file
        **kwargs : keyword parameters for `astropy.io.ascii.read`
        """
        from astropy.io import ascii
        from astropy import table
        indata = ascii.read(file, **kwargs)
        return cls.from_table(indata)

    def __len__(self):
        return self._len

    def as_table(self):
        """Convert parameters to an `astropy.table.Table`"""
        from astropy.table import Table, Column
        out = Table()
        if len(self) == 0:
            for k, v in self.items():
                if (not isinstance(v, str)) and hasattr(v, '__iter__'):
                    for i in range(len(v)):
                        c = Column([v[i]], name=k+'_{}'.format(i))
                        out.add_column(c)
                else:
                    c = Column([v], name=k)
                    out.add_column(c)
        else:
            for k, v in self.items():
                if v.ndim == 1:
                    c = Column(v, name=k)
                    out.add_column(c)
                else:
                    for i in range(v.shape[1]):
                        c = Column(v[:,i], name=k+'_{}'.format(i))
                        out.add_column(c)
        return out

    def write(self, file, **kwargs):
        """Write parameters to file"""
        self.as_table().write(file, **kwargs)


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

