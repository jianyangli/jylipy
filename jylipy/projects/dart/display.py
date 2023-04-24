import numpy as np
import astropy.units as u
from astropy.time import Time
from jylipy.image import ImageSet, Centroid
from jylipy.saoimage import TextRegion, VectorRegion, CircularRegion, RegionList
from jylipy import getds9



class ImageSequence(Centroid):
    """Display image sequence
    
    Optional parameters accepted:
        xc, yc : float
            Center pixel positions
        center : ... x 2 array
            Center pixel positions, override keywords `xc` and `yc`
        zoom : float
            DS9 zoom
        scale : 'linear', 'log'
            DS9 scale setting
        vmin, vmax : float
            DS9 scale limits
        rotate : float
            DS9 rotate angle
        cmap : str
            DS9 color map names
        cm1, cm2 : float
            DS9 color map parameters
    """

    def __init__(self, *args, **kwargs):
        anno = kwargs.pop('annotation', None)
        super().__init__(*args, **kwargs)
        if anno is not None:
            if isinstance(anno, Annotation):
                anno = np.full(self.shape, anno)
            elif not hasattr(anno, '__iter__'):
                raise ValueError('annotation needs to be either an `Annotation` object or an iterable of it')
            elif np.shape(anno) != self.shape:
                raise ValueError('Annotation needs to have the same shape as images')
            elif not all([isinstance(x, Annotation) for x in anno]):
                raise ValueError('All elements in `annotation` iterable needs to be `Annotation` object')
        self.annotation = anno        
    
    def display(self, d, offset=0):
        """Display image sequence
        
        offset : float or two element array
            The offset of pan center from image centroid
        """
        if self.image is None:
            for i in range(self.size):
                self._load_image(i)
        if not hasattr(offset, '__iter__'):
            offset = [offset] * 2
        for i in range(self.size):
            if self._1d['image'][i] is None:
                self._load_image(i)
            d.imdisp(self._1d['image'][i])
            ds9keys = set(self.attr).intersection(
                {'_xc', '_yc', '_zoom', '_scale', '_vmin', '_vmax', '_rotate', '_cmap', '_cm1', '_cm2'})
            #print(ds9keys)
            if '_xc' in ds9keys and '_yc' in ds9keys:
                if all(np.array(offset) == 0):
                    xc = self._1d['_xc'][i]
                    yc = self._1d['_yc'][i]
                else:
                    if '_rotate' in ds9keys:
                        cos = np.cos(np.deg2rad(self._1d['_rotate'][i]))
                        sin = np.sin(np.deg2rad(self._1d['_rotate'][i]))
                        xc = self._1d['_xc'][i] - offset[0] * cos - offset[1] * sin
                        yc = self._1d['_yc'][i] + offset[0] * sin - offset[1] * cos
                    else:
                        xc = self._1d['_xc'][i] - offset[0]
                        yc = self._1d['_yc'][i] - offset[1]
                d.set('pan to {} {}'.format(xc, yc))
            if '_zoom' in ds9keys:
                d.set('zoom to {}'.format(self._1d['_zoom'][i]))
            if '_scale' in ds9keys:
                d.set('scale {}'.format(self._1d['_scale'][i]))
            if '_vmin' in ds9keys and '_vmax' in ds9keys:
                d.set('scale limits {} {}'.format(self._1d['_vmin'][i], self._1d['_vmax'][i]))
            if '_rotate' in ds9keys:
                d.set('rotate to {}'.format(self._1d['_rotate'][i]))
            if '_cmap' in ds9keys:
                d.set('cmap {}'.format(self._1d['_cmap'][i]))
            if '_cm1' in ds9keys and '_cm2' in ds9keys:
                d.set('cmap {} {}'.format(self._1d['_cm1'][i], self._1d['_cm2'][i]))
            if self.annotation is not None:
                self.annotation[i].show(d)

def _rot(x, y, angle):
    angle = u.Quantity(angle, u.deg)
    cos = np.cos(angle).value
    sin = np.sin(angle).value
    x1 = x * cos + y * sin
    y1 = - x * sin + y * cos
    return x1, y1


class Text(TextRegion):
    
    def __init__(self, x, y, text, dx=0, dy=0, rotate=0, **kwargs):
        """
        x, y : float
            Pixel position of text center
        text : str
            Text to be displayed
        dx, dy : float
            Offset of text center in the display window in pixels
        rotate : float
            DS9 rotate angle
        **kwargs : other TextRegion keyword parameters
        """
        ddx, ddy = _rot(dx, dy, rotate)
        textrotate = kwargs.pop('textrotate', 0)
        kwargs['textrotate'] = 0
        super().__init__(ddx + x, ddy + y, text=text, **kwargs)
        

class Vector(VectorRegion):
    
    def __init__(self, x, y, length, angle, dx=0, dy=0, rotate=0, **kwargs):
        """
        x, y : float
            Pixel position of VectorRegion
        length : float
            Length of vector, pixel
        angle : float
            Angle of vector in display window, deg
        dx, dy : float
            Offset of vector in the display window, pixel
        rotate : float
            DS9 rotate angle
        **kwargs : other VectorRegion keyword parameters
        """
        ddx, ddy = _rot(dx, dy, rotate)
        super().__init__(ddx + x, ddy + y, length, 90 - rotate + angle, **kwargs)


class ImpactVector(VectorRegion):
    
    def __init__(self, x, y, length, angle, slide=0, rotate=0, **kwargs):
        """
        x, y : float
            Pixel position of VectorRegion
        length : float
            Length of vector, pixel
        angle : float
            Angle of vector in display window, deg
        dx, dy : float
            Offset of vector in the display window, pixel
        rotate : float
            DS9 rotate angle
        **kwargs : other VectorRegion keyword parameters
        """
        ddx = slide * np.cos(np.deg2rad(90 - rotate + angle))
        ddy = slide * np.sin(np.deg2rad(90 - rotate + angle))
        super().__init__(ddx + x, ddy + y, length, 90 - rotate + angle, **kwargs)


class ScaleBar(Vector):
    
    def __init__(self, x, y, length, **kwargs):
        vector = kwargs.pop('vector', 0)
        kwargs['vector'] = 0
        angle = kwargs.pop('angle', 90)
        super().__init__(x, y, length, angle, **kwargs)


class Annotation(dict):
    """Annotations to be added to DS9
    
    The object needs to be intialized with any keyword parameters.  The keys are the names
    of annotations, and the parameters are provided by the values of parameters.  The value
    of each dictionary item is array-like: [RegionClass, p1, p2, p3, ..., kwargs], where
    
        RegionClass : Class of a DS9 Region, must be the first element of the array
        p1, p2, p3, ... : region parameters to initialize RegionClass
        kwargs : dict, additional keyword parameters for RegionClass, must be the last element of the array
    
    Examples
    --------
    >>> xc = 350
    >>> yc = 240
    >>> text = 'text'
    >>> dart = 68 + 180
    >>> orientat = 135
    >>> sun = 118
    >>> vel = 48
    >>> 
    >>> anno = Annotation(text=[Text, xc, yc, text,
    ...                         {'dx': -35, 'dy': 75, 'rotate': orientat, 'color': 'white',
    ...                          'font': 'helvetica 14 normal roman'}],
    ...                   impact=[ImpactVector, xc, yc, 20, dart, 
    ...                           {'rotate': orientat, 'slide': -30, 'color': 'red', 'width': 2, 'text': 'DART'}],
    ...                   sun=[Vector, xc, yc, 20, sun,
    ...                        {'dx': -60, 'dy': 45, 'rotate': orientat, 'color': 'yellow', 'width': 2, 'text': 'Sun'}],
    ...                   vel=[Vector, xc, yc, 20, vel,
    ...                        {'dx': -60, 'dy': 45, 'rotate': orientat, 'color': 'cyan', 'width': 2, 'text': '+V'}],
    ...                   scalebar=[ScaleBar, xc, yc, 25,
    ...                             {'dx': -80, 'dy': 20, 'rotate': orientat, 'color': 'green', 'width': 4,
    ...                              'text': 'scale bar'}]
    ...                  )
    """
    def __init__(self, **kwargs):
        items = kwargs.copy()
        # filter out None values
        for name, val in kwargs.items():
            if val is None:
                items.pop(name)
        # initialize objects
        for name, val in items.items():
            self[name] = val[0](*tuple(val[1:-1]), **val[-1])
            
    def show(self, d):
        for k, v in self.items():
            v.show(ds9=d)


from jylipy.saoimage import CircularRegion, TextRegion, RegionList


class BullsEye(RegionList):

    def __init__(self, center, radii, labels=None, circle_kwargs={}, label_kwargs={}):
        circles = [CircularRegion(center[0], center[1], r, **circle_kwargs) for r in radii]
        super().__init__(circles)
        self.center = center
        self.radii = radii
        if labels is not None:
            for lbl, r in zip(labels, radii):
                self.append(TextRegion(center[0] + r * np.cos(np.deg2rad(45)), center[1] + r * np.sin(np.deg2rad(45)), text=lbl, **label_kwargs))


def generate_hst_dart_annotations(info, title=None, sun=None, vel=None,
    dart=None, scalebar=None, xc=1000, yc=1000, display_text=True,
    fontsize=12, time_unit='auto'):
    """Generate annotation array from info table for HST DART images
    
    Keyword parameters are used to pass position parameters for all annotation
    items.
        title : [dx, dy]
            Image title
        sun : [dx, dy, length]
            Sun vector
        vel : [dx, dy, length]
            Orbital velocity vectory
        dart : [slide, length], or [dx, dy, length]
            DART velocity vector.  `slide` is the distance to slide the vector from the image center
        scalebar : [dx, dy, length]
            Scale bar.  Length in km.
        display_text : bool
            Display text if `True`
        fontsize : int
            Font size for title
        time_unit : str, can be ['auto', 'sec', 'min', 'hour', 'day',
                                 'week', 'month', 'year']
            Unit of time displayed in the title.  If 'auto', then use the
            largest unit possible with value > 1.  For example, use hours
            for time < 24 hours, and days for 24 hours < time < 1 week,
            and so on.
        
    Default `None` is not to incude the corresponding annotation.
    """
    anno = []
    units = [u.second, u.minute, u.hour, u.day, u.week, u.year]
    time_intervals =  u.Quantity([1 * x for x in units])
    for r in info:
        if title is not None:
            dt = (Time(r['utc-mid']) - Time('2022-09-26T23:15'))
            if time_unit == 'auto':
                unit = units[np.where(abs(dt) > time_intervals)[0][-1]]
                #print(np.where(abs(dt) > time_intervals)[0][-1])
            else:
                unit = time_unit
            #print(unit)
            dt = dt.to_value(unit)
            title_text = '{} | T{:+.1f} {}'.format(r['utc-mid'][5:16], dt,
                                                   unit)
            text = [Text, xc, yc, title_text,
                             {'dx': title[0], 'dy': title[1], 'rotate': 0,
                              'color': 'white', 'font': 'helvetica {} normal roman'.format(fontsize)}]
        else:
            text = None
        if dart is not None:
            if len(dart) == 2:
                dartvec = [ImpactVector, xc, yc, dart[1], r['dartpa'],
                      {'rotate': 0, 'slide': dart[0], 'color': 'red', 'width': 2, 'text': 'DART' if display_text else ''}]
            elif len(dart) == 3:
                dartvec = [Vector, xc, yc, dart[2], r['dartpa'],
                      {'dx': dart[0], 'dy': dart[1], 'rotate': 0,
                       'color': 'red', 'width': 2,
                       'text': 'DART' if display_text else ''}]
            else:
                dartvec = None
        else:
            dartvec = None
        if sun is not None:
            sunvec = [Vector, xc, yc, sun[2], r['sunpa'],
                  {'dx': sun[0], 'dy': sun[1], 'rotate': 0, 'color': 'yellow', 'width': 2, 'text': 'Sun' if display_text else ''}]
        else:
            sunvec = None
        if vel is not None:
            velvec = [Vector, xc, yc, vel[2], r['velpa'],
                  {'dx': vel[0], 'dy': vel[1], 'rotate': 0, 'color': 'cyan', 'width': 2, 'text': '+V' if display_text else ''}]
        else:
            velvec=None
        if scalebar is not None:
            pxlscl = (0.04 * u.arcsec * r['range'] * u.au).to_value('km', u.dimensionless_angles())
            len_pix = scalebar[2] / pxlscl
            sclbar = [ScaleBar, xc, yc, len_pix,
                  {'dx': scalebar[0], 'dy': scalebar[1], 'rotate': 0, 'color': 'green', 'width': 4,
                    'text': '{:.0f}'.format(scalebar[2]) if display_text else ''}]
        else:
            sclbar = None
        anno.append(Annotation(title=text, dart=dartvec, sun=sunvec, vel=velvec, scalebar=sclbar))
    return anno


@u.quantity_input(dt=u.hour, delta=u.au)
def add_speedrings(dt, delta=0.074 * u.au, ds9=None,
    rr=np.linspace(20, 200, 10), center=[1001, 1001]):
    lbl = (rr * 0.04 * u.arcsec * delta / dt).to('m/s', u.dimensionless_angles())
    lbl = ['{:.2f}'.format(x) for x in lbl]
    spdrng = BullsEye(center, rr, labels=lbl, circle_kwargs={'dash': 1, 'zerobased': False})
    if ds9 is None:
        ds9 = getds9()
    spdrng.show(ds9=ds9)
