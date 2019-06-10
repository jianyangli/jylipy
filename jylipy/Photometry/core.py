'''
Units of all angles are in degrees!!!

'''

from collections import OrderedDict
import numpy as np, numbers
from ..core import ulen, condition
from ..plotting import density, pplot
from ..apext import units, Table, table, MPFitter, Column, fits
import astropy.units as u
from astropy.modeling import FittableModel, Fittable1DModel, Fittable2DModel, Parameter


recipi = 1/np.pi  # reciprocal pi


def _2rad(*args):
    return tuple(map(np.deg2rad, args))


def _2deg(*args):
    return tuple(map(np.rad2deg, args))


class GeometryError(Exception):
    '''Exception for non-geometry error.'''

    def __init__(self, msg, **kwargs):
        self.msg = msg
        self.geometry = list(kwargs.keys())
        for k in self.geometry:
            self.__dict__[k] = kwargs[k]

    def __str__(self):
        return repr(self.msg)


class ScatteringGeometry(object):
    '''Scattering geometry class.

    Initialize the class with one of the following four ways:

    geom = ScatteringGeometry()
      Initialize an empty class.  In this case, only `read` and `append`
      methods should be called

    geom = ScatteringGeometry(sca)
      `sca` is another ScatteringGeometry class instance

    geom = ScatteringGeometry(table)
      `table` is a Table containing all the needed fields

    geom = ScatteringGeometry(dictionary)
      `dictionary` is a dictionary containing all the needed fields

    Or initialize with keywords in one of the three angle combinations:

    geom = ScatteringGeometry(inc=value, emi=value, pha=value, **kwargs)
    geom = ScatteringGeometry(inc=value, emi=value, psi=value, **kwargs)
    geom = ScatteringGeometry(pha=value, lat=value, lon=value, **kwargs)

    Optional parameters **kwargs:

    cos : bool
      If `True`, then the all the scattering geometry parameters will
      be in the cosines of the corresponding angles.  In this case,
      keyword `unit` is not effective.  Default is `False`.
    unit : astropy.units
      The unit of all geometry parameters.  Default is `deg`.
      If the unit of initialization parameters are different from
      `unit`, then they will be converted to `unit`, unless `cos` is
      set to `True`.

 v1.0.1 : 11/1/2015, JYL @PSI
   Improve some class functions.
   Remove all setters for six geometry properties
   remove `__setitem__`
   Change property `names` to `angles`
   Disable methods `_iscos`
 v1.0.1 : 1/11/2016, JYL @PSI
   Removed keyword `fromfile`, use the first argument as the filename
     for initialization from a file
   Added initialization with no argument to generate an empty class instance
    '''

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            if len(kwargs) == 0:
                # Initialize an empty class.  In this case only `append` and `read` method can be called
                self._cos = None
                self._unit = None
                self._data = None
                return
            # Initialize from keywords
            self._check_keywords(kwargs)
            self._cos = kwargs.pop('cos')
            self._unit = kwargs.pop('unit')
            par = set(kwargs.keys())
            if set('inc emi pha'.split()) == par:
                data = [kwargs['inc'], kwargs['emi'], kwargs['pha']]
                names = 'inc emi pha'.split()
            elif set('inc emi psi'.split()) == par:
                data = [kwargs['inc'], kwargs['emi'], kwargs['psi']]
                names = 'inc emi psi'.split()
            elif set('pha lat lon'.split()) == par:
                data = [kwargs['pha'], kwargs['lat'], kwargs['lon']]
                names = 'pha lat lon'.split()
            else:
                raise GeometryError('Scattering geometry combination not recognized.')
            for i in range(len(data)):
                if not hasattr(data[i],'__iter__'):
                    data[i] = [data[i]]
            self._data = Table(data=data, names=names)
        elif len(args) == 1:
            if isinstance(args[0], ScatteringGeometry):
                # initialize with another ScatteringGeometry
                self._unit = args[0].unit
                self._cos = args[0].cos
                self._data = args[0]._data.copy()
            elif isinstance(args[0], Table) or isinstance(args[0], table.Row):
                # initialize from a table
                angle_keys = 'inc emi pha psi lat lon'.split()
                k = [c for c in args[0].colnames if c in angle_keys]
                if (not {'inc','emi','pha'}.issubset(set(k))) and (not {'inc','emi','psi'}.issubset(set(k))) and (not {'pha','lat','lon'}.issubset(set(k))):
                    raise ValueError('Initializing table is invalid')
                v = args[0][k]
                if v[k[0]].unit is None:
                    self._unit = units.deg
                    for x in k:
                        v[x].unit = units.deg
                else:
                    self._unit = v[k[0]].unit
                self._data = Table(v)
                self._cos = kwargs.pop('cos', False)
            elif isinstance(args[0], dict):
                # Initialize from a dictionary
                kwargs.update(args[0])
                self.__init__(**kwargs)
            elif isinstance(args[0], str):
                # Initialize from a file
                self.read(args[0], **kwargs)
                self._cos = kwargs.pop('cos', False)
        else:
            raise ValueError('At most 1 argument expected, {0} received'.format(len(args)))

    def _check_keywords(self, kwargs):
        # 1. check whether the input parameters are correct set
        # 2. check whether the length of all parameters are the same
        # 3. change all scaler input to vector
        # 4. change all numbers to quantity
        # save changes in the input dictionary
        if not set(kwargs.keys()).issubset(set('cos unit inc emi pha psi lat lon'.split())):
            raise TypeError('unexpected keyword received.')
        cos = kwargs.pop('cos', False)
        unit = units.Unit(kwargs.pop('unit', units.deg))
        if len(kwargs) != 3:
            raise TypeError('%s requires exactly three geometric parameters to initialize, got %d.' % (type(self), len(kwargs)))
        if set(kwargs.keys()) not in [set('inc emi pha'.split()), set('inc emi psi'.split()), set('pha lat lon'.split())]:
            raise TypeError('%s can only be initiated with (inc, emi, pha), or (inc, emi, psi), or (pha, lat, lon).' % type(self))
        if not unit.is_equivalent(units.deg):
            raise ValueError('Unit "rad" or equivalent is expected, but "%s" received in `unit` keyword.' % str(unit))
        keys = list(kwargs.keys())
        length = ulen(kwargs[keys[0]])
        for k in keys:
            val = kwargs[k]
            if isinstance(val, units.Quantity):
                val = val.to(condition(cos, units.dimensionless_unscaled, unit))
                if val.isscalar:
                    val = [val.value]*val.unit
                    l = 1
                else:
                    l = len(val)
            else:
                if hasattr(val, '__iter__'):
                    l = len(val)
                    val = val*condition(cos, units.dimensionless_unscaled, unit)
                else:
                    l = 1
                    val = [val]*condition(cos, units.dimensionless_unscaled, unit)
            if l != length:
                raise TypeError('All input geometry parameters must have the same length')
            kwargs[k] = val
        kwargs['cos'] = cos
        kwargs['unit'] = unit

    @property
    def angles(self):
        return list(self._data.keys())

    @property
    def unit(self):
        '''Unit of data, either deg or rad in astropy.units'''
        return self._unit
    @unit.setter
    def unit(self, value):
        value = units.Unit(value)
        if not value.is_equivalent(units.deg):
            raise ValueError('Unit "rad" or equivalent is expected, but "%s" is received.' % str(value))
        if not self.cos:
            for k in list(self._data.keys()):
                self._data[k] = self._data.getcolumn(k).to(value)
        self._unit = value

    @property
    def cos(self):
        '''If `True`, then the data saved are all the cosines of the
        corresponding angles.  Otherwise as angles with unit specified
        in .unit'''
        return self._cos
    @cos.setter
    def cos(self, value):
        if value != self.cos:
            if value == True:
                for k in list(self._data.keys()):
                    self._data[k] = np.cos(self._data.getcolumn(k))
            else:
                for k in list(self._data.keys()):
                    self._data[k] = np.arccos(self._data.getcolumn(k)).to(self.unit)
            self._cos = value

    @property
    def inc(self):
        '''Incidence angles'''
        if 'inc' not in self.angles:
            self._add_angle('inc')
        v = self._data.getcolumn('inc')
        #return condition(len(self) == 1, np.squeeze(v)*self.unit, v)
        return v

    def _add_angle(self, key):
        if key == 'inc':
            if ('pha' not in self.angles) or ('lat' not in self.angles) or ('lon' not in self.angles):
                raise GeometryError('Insufficient information available to calculate incidence angles')
            v = self.calcinc(self.pha, self.lat, self.lon, cos=self.cos).to(self.unit)
        elif key == 'emi':
            if ('pha' not in self.angles) or ('lat' not in self.angles) or ('lon' not in self.angles):
                raise GeometryError('Insufficient information available to calculate emission angles')
            v = self.calcemi(self.pha, self.lat, self.lon, cos=self.cos).to(self.unit)
        elif key == 'pha':
            if ('inc' not in self.angles) or ('emi' not in self.angles) or ('psi' not in self.angles):
                raise GeometryError('Insufficient information available to calculate phase angles')
            v = self.calcpha(self.inc, self.emi, self.psi, cos=self.cos).to(self.unit)
        elif key == 'psi':
            if (('inc' not in self.angles) or ('emi' not in self.angles) or ('pha' not in self.angles)) and (('pha' not in self.angles) or ('lat' not in self.angles) or ('lon' not in self.angles)):
                raise GeometryError('Insufficient information available to calculate plane angles')
            v = self.calcpsi(self.inc, self.emi, self.pha, cos=self.cos).to(self.unit)
        elif key == 'lat':
            if ('inc' not in self.angles) or ('emi' not in self.angles) or (('psi' not in self.angles) and ('pha' not in self.angles)):
                raise GeometryError('Insufficient information available to calculate latitudes')
            v = self.calclat(self.inc, self.emi, self.pha, cos=self.cos).to(self.unit)
        elif key == 'lon':
            if ('inc' not in self.angles) or ('emi' not in self.angles) or (('psi' not in self.angles) and ('pha' not in self.angles)):
                raise GeometryError('Insufficient information available to calculate longitudes')
            v = self.calclon(self.inc, self.emi, self.pha, cos=self.cos).to(self.unit)
        else:
            pass
        if not hasattr(v.value,'__iter__'):
            v = [v.value]*v.unit
        col = Column(v, name=key)
        self._data.add_column(col)
        self.angles.append(key)

    @property
    def emi(self):
        '''Emission angles'''
        if 'emi' not in self.angles:
            self._add_angle('emi')
        v = self._data.getcolumn('emi')
        #return condition(len(self) == 1, np.squeeze(v)*self.unit, v)
        return v

    @property
    def pha(self):
        '''Phase angles'''
        if 'pha' not in self.angles:
            self._add_angle('pha')
        v = self._data.getcolumn('pha')
        #return condition(len(self) == 1, np.squeeze(v)*self.unit, v)
        return v

    @property
    def psi(self):
        '''Plane angles (angle between incidence plane and emission plane)'''
        if 'psi' not in self.angles:
            self._add_angle('psi')
        v = self._data.getcolumn('psi')
        #return condition(len(self) == 1, np.squeeze(v)*self.unit, v)
        return v

    @property
    def lat(self):
        '''Photometric latitudes'''
        if 'lat' not in self.angles:
            self._add_angle('lat')
        v = self._data.getcolumn('lat')
        #return condition(len(self) == 1, np.squeeze(v)*self.unit, v)
        return v

    @property
    def lon(self):
        '''Photometric longitudes'''
        if 'lon' not in self.angles:
            self._add_angle('lon')
        v = self._data.getcolumn('lon')
        #return condition(len(self) == 1, np.squeeze(v)*self.unit, v)
        return v

    @property
    def mu0(self):
        if self.cos:
            return self.inc
        else:
            return np.cos(self.inc)

    @property
    def mu(self):
        if self.cos:
            return self.emi
        else:
            return np.cos(self.emi)

    @property
    def formatter(self):
        '''See astropy.table.Table.formatter'''
        return self._data.formatter
    @formatter.setter
    def formatter(self, value):
        self._data.formatter = value

    def __getitem__(self, k):
        return ScatteringGeometry(Table(self._data[k]), cos=self.cos)

    def __setitem__(self, k, v):
        if (not isinstance(v, ScatteringGeometry)) and (not isinstance(v, dict)) and (not isinstance(v, Table)):
            raise TypeError('can only assign dictionary type with another ScatteringGeometry, Table, or dictionary')
        v = ScatteringGeometry(v)
        v.unit = self.unit
        v.cos = self.cos
        for c in self.angles:
            if c not in v.angles:
                v._add_angle(c)
        self._data[k] = v._data

    def __array__(self):
        return self._data.__array__()

    def __len__(self):
        return self._data.__len__()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__().replace('Table', 'ScatteringGeometry')

    def __copy__(self):
        return ScatteringGeometry(self._data, cos=self.cos)

    def __eq__(self, other):
        if not isinstance(other, ScatteringGeometry):
            raise TypeError('comparison of %s with %s not defined.' % (type(self), type(other)))
        if ((self.inc-other.inc).value > 1e-10).any() or \
            ((self.emi-other.emi).value > 1e-10).any() or \
            ((self.pha-other.pha).value > 1e-10).any():
            return False
        else:
            return True

    @staticmethod
    def calcpsi(inc, emi, pha, cos=False, rejected={}, good={}):
        '''Calculate scattering plane angle `psi` from incidence, emission,
        and phase angles

        All angles are in radiance or in astropy Quantity.

        See Shkuratov et al. (2011), Eq. 1.
        '''
        cosinc, cosemi, cospha = ScatteringGeometry._setcos(cos, inc, emi, pha)
        sininc = np.sqrt(1-cosinc*cosinc)
        sinemi = np.sqrt(1-cosemi*cosemi)
        cospsi = (cospha - cosinc*cosemi) / (sininc * sinemi)
        return condition(cos, cospsi, np.arccos(cospsi))

    @staticmethod
    def calclat(inc, emi, pha, cos=False, rejected={}, good={}):
        '''Calculate photometric latitude from incidence, emission, and
        scattering plane angles

        All angles are in radiance or in astropy Quantity.

        Use Eqs. (10) and (11) in Kreslavsky et al. (2000) JGR 105,
        20,281-20,295
        '''
        cosemi = ScatteringGeometry._setcos(cos, emi)
        coslon = ScatteringGeometry.calclon(inc, emi, pha, cos=cos)
        if not cos:
            coslon = np.cos(coslon)
        coslat = cosemi/coslon
        return condition(cos, coslat, np.arccos(coslat))

    @staticmethod
    def calclon(inc, emi, pha, cos=False, rejected={}, good={}):
        '''Calculate photometric longitude from incidence, emission, and
        scattering plane angles

        All angles are in radiance or in astropy Quantity.

        Use Eqs. (10) and (11) in Kreslavsky et al. (2000) JGR 105,
        20,281-20,295
        '''
        cosinc, cosemi, cospha = ScatteringGeometry._setcos(cos, inc, emi, pha)
        sinpha = np.sqrt(1.-cospha*cospha)
        lon = np.arctan2(cosinc/cosemi-cospha, sinpha)
        return condition(cos, np.cos(lon), lon)

    @staticmethod
    def calcpha(inc, emi, psi, cos=False, rejected={}, good={}):
        '''Calculate phase angle from incidence, emission, and scattering
        plane angles

        All angles are in radiance or in astropy Quantity.

        See Shkuratov et al. (2011), Eq. 2.
        '''
        cosinc, cosemi, cospsi = ScatteringGeometry._setcos(cos, inc, emi, psi)
        sininc = np.sqrt(1-cosinc*cosinc)
        sinemi = np.sqrt(1-cosemi*cosemi)
        cospha = cosinc*cosemi + sininc*sinemi*cospsi
        return condition(cos, cospha, np.arccos(cospha))

    @staticmethod
    def calcinc(pha, lat, lon, cos=False, rejected={}, good={}):
        '''Calculate incidence angle from phase angle and photometric
        longitude and latitude.

        All angles are in radiance or in astropy Quantity.

        See Shkuratov et al. (2011), Eq. 1.
        '''
        cospha, coslat, coslon = ScatteringGeometry._setcos(cos, pha, lat, lon)
        sinpha = np.sqrt(1-cospha*cospha)
        sinlon = np.sqrt(1-coslon*coslon)
        cosinc = coslat * (cospha*coslon + sinpha*sinlon)
        return condition(cos, cosinc, np.arccos(cosinc))

    @staticmethod
    def calcemi(pha, lat, lon, cos=False, rejected={}, good={}):
        '''Calculate emission angle from phase angle and photometric
        longitude and latitude

        All angles are in radiance or in astropy Quantity.

        See Shkuratov et al. (2011), Eq. 1.
        '''
        coslat, coslon = ScatteringGeometry._setcos(cos, lat, lon)
        cosemi = coslat*coslon
        return condition(cos, cosemi, np.arccos(cosemi))

    def astable(self):
        '''Return data as an astropy.table.Table instance'''
        return self._data

    def argsort(self, keys=None, kind=None, **kwargs):
        '''Return the indices which would sort the geometry data
        according to one or more keys.

        Available keys include ['Incidence', 'emi', 'pha',
        'psi', 'lat', 'lon']

        See astropy.table.Table.argsort'''
        if keys is not None:
            if isinstance(keys, (str,bytes)):
                keys = [keys]
            if not set(keys).issubset(set(self._data.keys())):
                raise ValueError("keys must be in "+str(self.names()))
        return self._data.argsort(keys=keys, kind=kind, **kwargs)

    def copy(self, *args, **kwarg):
        '''Return a copy of the scattering geometry instance'''
        return self.__copy__()

    def pprint(self, *args, **kwargs):
        '''Print a formatted string representation of the data

        See astropy.table.Table.pprint'''
        return self._data.pprint(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        '''Reverse the order of data.

        See astropy.table.Table.reverse'''
        return self._data.reverse(*args, **kwargs)

    def show_in_browser(self, *args, **kwargs):
        '''Render the data in HTML and show it in a web browser.

        See astropy.table.Table.show_in_browser'''
        return self._data.show_in_browser(*args, **kwargs)

    def sort(self, keys):
        '''Sort geometry data according to one or more keys.

        Available keys include ['Incidence', 'emi', 'pha',
        'psi', 'lat', 'lon']

        See table.Table.sort'''
        if isinstance(keys, (str,bytes)):
            keys = [keys]
        if not set(keys).issubset(set(self._data.keys())):
            raise ValueError("keys must be in "+str(self.names()))
        return self._data.sort(keys)

    def write(self, *args, **kwargs):
        '''Write data out in the specified format

        *** The `cos` property of this class is not saved this way ***
        See astropy.table.Table.write'''
        return self._data.write(*args, **kwargs)

    def read(self, *args, **kwargs):
        '''Read data from a file'''
        data = Table.read(*args, **kwargs)
        self._data = data
        self._unit = self._data[self.angles[0]].unit
        if 'cos' in data.meta:
            self._cos = data.meta['cos']
        else:
            self._cos = False

    @staticmethod
    def _setcos(cos, *var):
        if cos:
            return condition(len(var) == 1, var[0], var)
        else:
            var = _2rad(*var)
            vout = [np.cos(v) for v in var]
            return condition(len(var)==1, vout[0], tuple(vout))

    def merge(self, sca):
        '''Merge two instances of ScatteringGeometry'''
        keymap = {'inc': 'Incidence', 'emi': 'emi', 'pha': 'pha', 'psi': 'psi', 'lat': 'lat', 'lon': 'lon'}
        keys = set(self.angles)
        [keys.add(k) for k in sca.angles]
        for k in keys:
            if k not in self.angles:
                self._add_angle(keymap[k])
            if k not in sca.angles:
                sca._add_angle(keymap[k])
        unit0 = sca.unit
        cos0 = sca.cos
        if unit0 != self.unit:
            sca.unit = self.unit
        if cos0 != self.cos:
            sca.cos = self.cos
        self._data = table.vstack((self._data, sca._data))
        if unit0 != self.unit:
            sca.unit = unit0
        if cos0 != self.cos:
            sca.cos = cos0

    def append(self, v):
        '''Append data to the end'''
        v = ScatteringGeometry(v)
        if self._data is None:
            self.__init__(v)
        else:
            v.unit = self.unit
            v.cos = self.cos
            for c in self.angles:
                if c not in v.angles:
                    v._add_angle(c)
            for c in v.angles:
                if c not in self.angles:
                    self._add_angle(c)
            self._data = table.vstack((self._data, v._data))

    def remove_rows(self, *args, **kwargs):
        self._data.remove_rows(*args, **kwargs)

    def remove_row(self, *args, **kwargs):
        self._data.remove_row(*args, **kwargs)

    def is_valid(self):
        '''Check the validity of geometry, returns a bool array'''
        return np.isfinite(self.inc) & np.isfinite(self.emi) & \
            np.isfinite(self.pha) & np.isfinite(self.lon) & \
            np.isfinite(self.lat) & np.isfinite(self.psi) & \
            (self.inc+self.emi >= self.pha) & \
            (abs(self.inc-self.emi) <= self.pha)

    def validate(self):
        good = self.is_valid()
        if good.all():
            return []
        w = np.where(~good)[0]
        self.remove_rows(w)
        return list(w)


class LatLon(object):
    '''Latitude-Longitude coordinate class'''
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            self._data = None
        elif len(args) == 1:
            if isinstance(args[0], Table):
                self._data = args[0].copy()
            elif isinstance(args[0], dict):
                self._data = Table(args[0])
            elif isinstance(args[0], LatLon):
                self._data = args[0]._data.copy()
            else:
                raise ValueError('Unrecognized data type')
        elif len(args) == 2:
            self._data = Table(args, names=['lon','lat'])
        if self._data is not None:
            if self._data['lon'].unit is None:
                u = kwargs.pop('unit', None)
                if u is None:
                    self._data['lon'].unit = units.deg
                    self._data['lat'].unit = units.deg
                else:
                    self._data['lon'].unit = u
                    self._data['lat'].unit = u

    @property
    def lon(self):
        return self._data.getcolumn('lon')

    @property
    def lat(self):
        return self._data.getcolumn('lat')

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, k):
        return self._data[k]

    def __copy__(self):
        return LatLon(self)

    def astable(self):
        return self._data.copy()

    def copy(self):
        return self.__copy__()

    def append(self, v):
        if v is not None:
            v = LatLon(v)
            self._data = table.vstack((self._data, v._data))

    def remove_row(self, *args, **kwargs):
        self._data.remove_row(*args, **kwargs)

    def remove_rows(self, *args, **kwargs):
        self._data.remove_rows(*args, **kwargs)

    def merge(self, geo):
        if geo is not None:
            geo = LatLon(geo)
            self._data = table.vstack((self._data, geo._data))


class PhotometricData(object):
    '''Photometric data class.

    Class can be initialized in four ways:

    PhotometricData()
      Will initiate an empty class.  In this case, only `read`, `append`,
      `extract`, and `populate` methods should be called.

    PhotometricData(data)
      data : PhotometricData instance, Table, or dict
     If `Table` or `dict`, it has to have keys for angles ('inc',
     'emi', 'pha', 'psi', 'lat', 'lon') that makes a complete set
     of scattering angles, and at least one reflectance quantity
     ('BDR', 'RADF', 'BRDF', 'REFF').

    PhotometricData(file)
      file : str
     The name of PhotometricData data file.

    PhotometricData(keywords=values)
      inc, emi, pha, psi, lat, lon : array-like
     Scattering angles
      bdr, r, iof, i/f, radf, brdf, reff : array-like
     Reflectance quantity

    Other keywords:

      cos : bool, optional
     If `True`, then angles are cosines.  Default is `False`
      unit : str, optional
     Unit of angles ('dge' or 'rad'), default is 'deg'.
      type : str, optional
     Set data flag `measured`, can be 'binned' or 'measured'.  If
     'binned', then the binning parameters will be copied or set.

    Properties:
    -----------
    .sca: ScatteringGeometry class instance, scattering geometries
    .inc, .emi, .pha, .psi, .pholat, .pholon, .mu0, .mu: array-like, scattering
      geometry data
    .ref: Table instance, reflectance data table
    .BDR, .BRDF, .RADF, .REFF: numpy array, various reflectance quantities,
    .type: str, type of data, 'measured' or 'binned'
    .refkey: str list, the name of reflectance quantities in `ref` table
    .lonlim, .latlim: 2-element array-like, the limit of longitude and
      latitude covered by data.
    .binparms: Binning parameters.

    Methods:
    --------
    .append: Append photometric data to the end
    .merge: Merge two PhotometricData instances
    .astable: Convert the data to an astropy Table
    .plot: Plot the data
    .read: Read data from a data file
    .write: Write the data to a data file

    TBD:
      The class only supports a simgle layered photometric data, i.e., not
      spectral data that many bands share the same scattering geometry

    v1.0.0 : 11/1/2015, JYL @PSI
    v1.0.1 : 1/11/2016, JYL @PSI
      Removed keyword `fromfile`, use the first argument as the filename
     for initialization from a file
      Added initialization with no arguments to generate an empty class instance
    '''

    def __init__(self, *args, **kwargs):
        if (len(args) == 0) or ((len(args) == 1) and (args[0] is None)):
            if len(kwargs) == 0:
                # Initialize an empty class.  In this case, only `append` and `read` methods can be called
                self._data = None
                self.sca = None
                self.geo = None
                self._type = None
                return

            # Initialize from data passed by keywords
            self._type = kwargs.pop('type', 'measured')
            if self._type == 'binned':
                if 'binparms' not in list(kwargs.keys()):
                    raise ValueError('`binparms` keyword is not found while `type` is set to `binned`.')
                self.binparms = kwargs.pop('binparms')
            else:
                self.binparms = None

            #self.lonlim = kwargs.pop('lonlim', None)
            #self.latlim = kwargs.pop('latlim', None)

            # collect scattering geometry data
            scakey = {}
            for k in 'inc emi pha psi pholat pholon'.split():
                if k in list(kwargs.keys()):
                    scakey[k] = kwargs.pop(k)
            if 'pholat' in scakey.keys():
                scakey['lat'] = scakey['pholat']
                scakey.pop('pholat')
            if 'pholon' in scakey.keys():
                scakey['lon'] = scakey['pholon']
                scakey.pop('pholon')
            scakey['cos'] = kwargs.pop('cos', False)
            scakey['unit'] = kwargs.pop('unit', units.deg)
            self.sca = ScatteringGeometry(**scakey)

            # collect reflectance data
            keys = list(kwargs.keys())
            if 'bdr' in keys:
                self._data = Table([kwargs['bdr']], names=['BDR'])
            elif 'r' in keys:
                self._data = Table([kwargs['r']], names=['BDR'])
            elif 'iof' in keys:
                self._data = Table([kwargs['iof']], names=['RADF'])
            elif 'i/f' in keys:
                self._data = Table([kwargs['i/f']], names=['RADF'])
            elif 'radf' in keys:
                self._data = Table([kwargs['radf']], names=['RADF'])
            elif 'brdf' in keys:
                self._data = Table([kwargs['brdf']], names=['BRDF'])
            elif 'reff' in keys:
                self._data = Table([kwargs['reff']], names=['REFF'])
            else:
                raise ValueError('No reflectance data found.')

            # collect lat-lon data
            if ('geolat' in keys) & ('geolon' in keys):
                self.geo = LatLon(kwargs['geolon'],kwargs['geolat'])
            else:
                self.geo = None

        elif len(args) == 1:
            if isinstance(args[0], PhotometricData):
                # Initialize from another PhotometricData instance
                data = args[0]
                self.sca = data.sca.copy()
                self._data = data._data.copy()
                if data.geo is None:
                    self.geo = None
                else:
                    self.geo = data.geo.copy()
                self._type = data.type
                self.binparms = args[0].binparms
            elif isinstance(args[0], Table):
                # Initialize from an astropy Table
                cos = kwargs.pop('cos', False)
                angle_keys = ['inc', 'emi', 'pha', 'psi']
                ref_keys = ['BDR', 'RADF', 'BRDF', 'REFF']
                geo_keys = ['lat', 'lon']
                ak = [x for x in args[0].colnames if x in angle_keys]
                rk = [x for x in args[0].colnames if x in ref_keys]
                gk = [x for x in args[0].colnames if x in geo_keys]
                self.sca = ScatteringGeometry(args[0][ak], cos=cos)
                if len(gk)>0:
                    self.geo = LatLon(args[0][gk])
                else:
                    self.geo = None
                self._data = args[0][rk].copy()
                meta = args[0].meta
                if 'type' in meta:
                    self._type = meta['type']
                else:
                    self._type = kwargs.pop('type', 'measured')
                if 'binparms' in meta:
                    self.binparms = meta['binparms']
                else:
                    self.binparms = None
            elif isinstance(args[0], dict):
                # Initialize from a dictionary
                kwargs.update(args[0])
                self.__init__(**kwargs)
            elif isinstance(args[0], str):
                # Initialize from a file
                self.read(args[0], **kwargs)
        else:
            raise ValueError('At most 1 argument expected, {0} received'.format(len(args)))
        self._set_properties()

    def _set_properties(self):
        if len(self)>0:
            if self.geo is None:
                self._lonlim = None
                self._latlim = None
            else:
                self._lonlim = [self.geolon.min(), self.geolon.max()]
                self._latlim = [self.geolat.min(), self.geolat.max()]
            sca = self.sca
            self._inclim = [sca.inc.min(), sca.inc.max()]
            self._emilim = [sca.emi.min(), sca.emi.max()]
            self._phalim = [sca.pha.min(), sca.pha.max()]
        else:
            self._lonlim = None
            self._latlim = None
            self._inclim = None
            self._emilim = None
            self._phalim = None

    @property
    def type(self):
        return self._type

    @property
    def refkey(self):
        return list(self._data.keys())

    @property
    def BDR(self):
        if 'BDR' not in self.refkey:
            self._add_refkey('BDR')
        return self._data.getcolumn('BDR')

    @property
    def RADF(self):
        if 'RADF' not in self.refkey:
            self._add_refkey('RADF')
        return self._data.getcolumn('RADF')

    @property
    def BRDF(self):
        if 'BRDF' not in self.refkey:
            self._add_refkey('BRDF')
        return self._data.getcolumn('BRDF')

    @property
    def REFF(self):
        if 'REFF' not in self.refkey:
            self._add_refkey('REFF')
        return self._data.getcolumn('REFF')

    @property
    def ref(self):
        return self._data

    @property
    def geolon(self):
        if self.geo is not None:
            return self.geo.lon

    @property
    def geolat(self):
        if self.geo is not None:
            return self.geo.lat

    @property
    def inc(self):
        return self.sca.inc

    @property
    def emi(self):
        return self.sca.emi

    @property
    def pha(self):
        return self.sca.pha

    @property
    def psi(self):
        return self.sca.psi

    @property
    def pholat(self):
        return self.sca.lat

    @property
    def pholon(self):
        return self.sca.lon

    @property
    def mu0(self):
        return self.sca.mu0

    @property
    def mu(self):
        return self.sca.mu

    @property
    def lonlim(self):
        return self._lonlim

    @property
    def latlim(self):
        return self._latlim

    @property
    def inclim(self):
        return self._inclim

    @property
    def emilim(self):
        return self._emilim

    @property
    def phalim(self):
        return self._phalim

    def _add_refkey(self, key):
        if key == 'BDR':
            if 'RADF' in self.refkey:
                self._data.add_column(Column(self.RADF/np.pi, name=key))
            elif 'BRDF' in self.refkey:
                self._data.add_column(Column(self.BRDF*self.mu0, name=key))
            else:
                self._data.add_column(Column(self.REFF*self.mu0/np.pi, name=key))
        elif key == 'RADF':
            if 'BDR' in self.refkey:
                self._data.add_column(Column(self.BDR*np.pi, name=key))
            elif 'BRDF' in self.refkey:
                self._data.add_column(Column(self.BRDF*self.mu0*np.pi, name=key))
            else:
                self._data.add_column(Column(self.REFF*self.mu0, name=key))
        elif key == 'BRDF':
            if 'BDR' in self.refkey:
                self._data.add_column(Column(self.BDR/self.mu0, name=key))
            elif 'RADF' in self.refkey:
                self._data.add_column(Column(self.RADF/(np.pi*self.mu0), name=key))
            else:
                self._data.add_column(Column(self.REFF/np.pi, name=key))
        elif key == 'REFF':
            if 'BDR' in self.refkey:
                self._data.add_column(Column(self.BDR*np.pi/self.mu0, name=key))
            elif 'RADF' in self.refkey:
                self._data.add_column(Column(self.RADF/self.mu0, name=key))
            else:
                self._data.add_column(Column(self.BRDF*np.pi, name=key))
        else:
            pass

    def __getitem__(self, k):
        s = self.sca[k].astable()
        r = self._data[k]
        if self.geo is not None:
            g = self.geo[k]
            out = PhotometricData(table.hstack((s,r,g)))
        else:
            out = PhotometricData(table.hstack((s,r)))
        return out

    #def __setitem__(self, k, v):
    #   v = PhotometricData(v)
    #   for c in self.refkey:
    #       if c not in v.refkey:
    #           v._add_refkey(c)
    #   for c in v.refkey:
    #       if c not in self.refkey:
    #           self._add_refkey(c)
    #   self._data[k] = v._data
    #   self.sca[k] = v.sca

    def __array__(self):
        return table.hstack((self.sca.astable(), self._data)).__array__()

    def __len__(self):
        if self._data is None:
            return 0
        else:
            return len(self._data)

    def __iter__(self):
        for i in range(len(self)):
            yield self.astable()[i]

    def __str__(self):
        return self.astable().__str__()

    def __repr__(self):
        return self.astable().__repr__().replace('Table', 'PhotometricData')

    def __copy__(self):
        return PhotometricData(self)

    def copy(self):
        return self.__copy__()

    def astable(self):
        if self._data is None:
            return None
        scatbl = self.sca.astable()
        tblkeys = scatbl.keys()
        if 'lon' in tblkeys:
            scatbl.rename_column('lon','pholon')
        if 'lat' in tblkeys:
            scatbl.rename_column('lat','pholat')
        out = table.hstack((scatbl, self._data))
        if self.geo is not None:
            geotbl = self.geo.astable()
            tblkeys = geotbl.keys()
            if 'lon' in tblkeys:
                geotbl.rename_column('lon','geolon')
            if 'lat' in tblkeys:
                geotbl.rename_column('lat','geolat')
            out = table.hstack((out, geotbl))
        out.meta['dtype'] = self.type
        if self.lonlim is not None:
            out.meta['maxlon'] = self.lonlim[1]
            out.meta['minlon'] = self.lonlim[0]
        if self.latlim is not None:
            out.meta['maxlat'] = self.latlim[1]
            out.meta['minlat'] = self.latlim[0]
        if self.binparms is not None:
            out.meta['binparms'] = self.binparms
        return out

    def plot(self, x=None, y='RADF', correction=None, unit='deg', type='auto', **kwargs):
        '''Plot photometric data.

        x : str, optional
          Quantity for x-axis, can be 'inc', 'emi', 'pha', 'psi', 'lat', 'lon'.
        y : str, optional
          Quantity for y-axis, can be 'BDR', 'RADF', 'BRDF', 'REFF'.
        unit : str or astropy.units, optional
          The unit of x-axis.  Can be 'deg' or 'rad'.
        type : str, optional
          'auto': The type is automatically determined
          'scatter' : Scatter point plot
          'density' : Density plot
        *kwargs* : Keywords accepted by jylipy.plotting.pplot

        v1.0.0 : 11/1/2015, JYL @PSI
        '''

        # set plot type
        if type == 'auto':
            if len(self) > 100000:
                type = 'density'
            else:
                type = 'scatter'

        # prepare plotting quantities
        xlbl = {'inc': 'Incidence Angle', 'emi': 'Emission Angle', 'pha': 'Phase Angle', 'psi': 'Plane Angle', 'lat': 'Photometric Latitude', 'lon': 'Photometric Longitude'}
        yy = getattr(self, y)
        ylabel = kwargs.pop('ylabel', y)
        if correction != None:
            if correction.lower() in ['ls', 'lommel-seeliger']:
                mu0 = np.cos(self.inc)
                mu = np.cos(self.emi)
                corr = 2*mu0/(mu0+mu)
                yy = yy/corr
                ylabel = ylabel+'$/[2\mu_0/(\mu_0+\mu)]$'
            elif correction.lower() in ['lambert']:
                corr = np.cos(self.inc)
                yy = yy/corr
                ylabel = ylabel+'$/\mu_0$'
            else:
                raise ValueError('correction type must be in [''LS'', ''Lommel-Seeliger'', ''Lambert'', {0} received'.format(correction))

        # make plots
        if type == 'density':
            if x is None:
                x = 'pha'
            xx = getattr(self, x).to(unit).value
            xlabel = kwargs.pop('xlabel', '{0} ({1})'.format(xlbl[x], str(unit)))
            density(xx, yy, xlabel=xlabel, ylabel=ylabel, **kwargs)
        elif type == 'scatter':
            from matplotlib import pyplot as plt
            if x is not None:
                xx = getattr(self, x).to(unit).value
                xlabel = kwargs.pop('xlabel', '{0} ({1})'.format(xlbl[x], str(unit)))
                plt.plot(xx, yy, 'o')
                pplot(xlabel=xlabel, ylabel=ylabel, **kwargs)
            else:
                f, ax = plt.subplots(3,1,num=plt.gcf().number)
                xs = ['pha','inc','emi']
                for x,i in zip(xs,list(range(3))):
                    xx = getattr(self, x).to(unit).value
                    xlabel = '{0} ({1})'.format(xlbl[x], str(unit))
                    ax[i].plot(xx, yy, 'o')
                    pplot(ax[i], xlabel=xlabel, ylabel=ylabel, **kwargs)
                plt.draw()
        else:
            raise ValueError("`type` of plot can only be 'auto', 'scatter', or 'density'")

    def write(self, *args, **kwargs):
        '''Save photometric data to a FITS file'''
        data = self.astable()
        return data.write(*args, **kwargs)

    def read(self, filename):
        '''Read photometric data from file'''
        infits = fits.open(filename)[1]
        indata = Table(infits.data)
        ang_keys = ['inc', 'emi', 'pha', 'psi', 'pholat', 'pholon']
        ref_keys = ['BDR', 'RADF', 'BRDF', 'REFF']
        geo_keys = ['lat', 'lon', 'geolat', 'geolon']

        ak = [x for x in indata.colnames if x in ang_keys]
        at = indata[ak]
        rk = [x for x in indata.colnames if x in ref_keys]
        rt = indata[rk]
        gk = [x for x in indata.colnames if x in geo_keys]
        gt = indata[gk]
        if 'geolat' in gt.keys():
            gt.rename_column('geolat', 'lat')
        if 'geolon' in gt.keys():
            gt.rename_column('geolon', 'lon')
        if 'TUNIT1' in infits.header:
            unit = infits.header['TUNIT1'].strip()
        else:
            unit = 'deg'
        for c in ak:
            at[c].unit = unit
        self.sca = ScatteringGeometry(at)
        self._data = rt.copy()
        self._type = infits.header.pop('DTYPE', 'measured')
        self.binparms = infits.header.pop('binparms',None)
        if len(gk)>0:
            self.geo = LatLon(gt)
        else:
            self.geo = None
        self._set_properties()

    def append(self, v):
        '''Append data to the end'''
        if self._data is None:
            self.__init__(v)
        else:
            v = PhotometricData(v)
            for c in self.refkey:
                if c not in v.refkey:
                    v._add_refkey(c)
            for c in v.refkey:
                if c not in self.refkey:
                    self._add_refkey(c)
            self.sca.append(v.sca)
            self._data = table.vstack((self._data, v._data))
            if self.geo is not None:
                self.geo.append(v.geo)
            self._set_properties()

    def merge(self, pho, type='auto'):
        '''Merge two PhotometricData instances'''
        import warnings
        assert isinstance(pho, PhotometricData)
        if len(pho) == 0:
            return

        if type == 'auto':
            if (self.type == 'measured') and (pho.type == 'measured'):
                type = 'simple'
            #self._simple_merge(pho)
            elif (self.type == 'measured') and (pho.type == 'binned'):
            # raise a warning, no merge
                warnings.warn('Merge a "binned" type into a "measured" type, simple mode will be used, a "measured" type will be set to output')
                type = 'simple'
            elif (self.type == 'binned') and (pho.type == 'measured'):
                boundary = self.binparms['boundary']
                print('    binning data')
                binner = Binner(boundary=boundary)
                pho_binned = binner(pho)
                type = 'binned'
            else:
                type = 'binned'
        if type == 'binned':
            # bin pho first, then merge
            if 'pho_binned' in locals():
                self._binned_merge(pho_binned)
            else:
                self._binned_merge(pho)
        else:
            # check boundaries of both binned data, then merge
            self._simple_merge(pho)
        self._set_properties()

    def _simple_merge(self, pho):
        '''Merge two photometric datasets in a simple case, i.e., both
        `measured` type.'''

        self.sca.merge(pho.sca)
        if self.geo is not None and pho.geo is not None:
            self.geo.merge(pho.geo)

        keys = set(self.refkey)
        [keys.add(k) for k in pho.refkey]
        for k in keys:
            if k not in self.refkey:
                self._add_refkey(k)
            if k not in pho.refkey:
                pho._add_refkey(k)
        self._data = table.vstack((self._data, pho._data))
        self._set_properties()
        #if self.lonlim is not None and pho.lonlim is not None:
        #   self.lonlim = [min([self.lonlim[0],pho.lonlim[0]]),max([self.lonlim[1],pho.lonlim[1]])]
        #else:
        #   self.lonlim = None
        #if self.latlim is not None and pho.latlim is not None:
        #   self.latlim = [min([self.latlim[0],pho.latlim[0]]),max([self.latlim[1],pho.latlim[1]])]
        #else:
        #   self.latlim = None

    def _binned_merge(self, pho):
        # check binning boundaries
        if set(self.binparms['dims']) != set(pho.binparms['dims']):
            warnings.warn('Cannot merge two `binned` type PhotometricData with different binning dimensions')
            return
        bds = []
        indx = []
        for d in self.binparms['dims']:
            i = self.binparms['dims'].index(d)
            bd1 = np.asarray(self.binparms['boundary'][i])
            bd1len = len(bd1)
            j = pho.binparms['dims'].index(d)
            bd2 = np.asarray(self.binparms['boundary'][j])
            bd2len = len(bd2)
            indx.append(j)
            min1 = bd1.min()
            min2 = bd2.min()
            max1 = bd1.max()
            max2 = bd2.max()
            bds.append(None)
            if (min1<min2) & (max1>=max2):
                # case 1
                #print 'case 1'
                v1 = bd1[np.where((bd1>=min2) & (bd1<=max2))]
                v2 = bd2
                if len(v1) != 0:
                    if (len(v1) == len(v2)):
                        if abs(v1-v2).max() < 1e-7:
                            bds[-1] = bd1
            elif (min1>=min2) & (max1<max2):
                # case 2
                #print 'case 2'
                v1 = bd2[np.where((bd2>=min1) & (bd2<=max1))]
                v2 = bd1
                if len(v1) != 0:
                    if (len(v1) == len(v2)):
                        if abs(v1-v2).max() < 1e-7:
                            bds[-1] = bd2
            elif (min1<min2) & (max1<max2):
                # cases 3, 5
                if min2<max1:
                    # case 3
                    #print 'case 3'
                    v1 = bd1[np.where(bd1>=min2)]
                    v2 = bd2[np.where(bd2<=max1)]
                    if len(v1) == len(v2):
                        if len(v1) == 0:
                            bds[-1] = np.concatenate((bd1, bd2[np.where(bd2>max1)]))
                        else:
                            if abs(v1-v2).max() < 1e-7:
                                bds[-1] = np.concatenate((bd1, bd2[np.where(bd2>max1)]))
                else:
                    # case 5
                    #print 'case 5'
                    bds[-1] = np.concatenate((bd1, bd2))
            else:  # (min1>=min2) & (max1>=max2):
                # cases 4, 6
                if min1<max2:
                    # case 4
                    #print 'case 4'
                    v1 = bd1[np.where(bd1<=max2)]
                    v2 = bd2[np.where(bd2>=min1)]
                    if len(v1) == len(v2):
                        if len(v1) == 0:
                            bds[-1] = np.concatenate((bd2, bd1[np.where(bd1>max2)]))
                        else:
                            if abs(v1-v2).max() < 1e-7:
                                bds[-1] = np.concatenate((bd2, bd1[np.where(bd1>max2)]))
                else:
                    # case 6
                    #print 'case 6'
                    bds[-1] = np.concatenate((bd2, bd1))
            if bds[-1] is None:
                raise ValueError('unmatched binning boundaries')

        unit0 = pho.sca.unit
        cos0 = pho.sca.cos
        if unit0 != self.sca.unit:
            pho.sca.unit = self.sca.unit
        if cos0 != self.sca.cos:
            pho.sca.cos = self.sca.cos

        self_ang = [getattr(self.sca, self.binparms['dims'][i]).value for i in range(3)]
        pho_ang = [getattr(pho.sca, pho.binparms['dims'][indx[i]]).value for i in range(3)]
        for i1,i2 in zip(bds[0][:-1],bds[0][1:]):
            self_indx0 = (self_ang[0] >= i1) & (self_ang[0] < i2)
            pho_indx0 = (pho_ang[0] >= i1) & (pho_ang[0] < i2)
            for j1,j2 in zip(bds[1][:-1],bds[1][1:]):
                self_indx1 = self_indx0 & (self_ang[1] >= j1) & (self_ang[1] < j2)
                pho_indx1 = pho_indx0 & (pho_ang[1] >= j1) & (pho_ang[1] < j2)
                for k1, k2 in zip(bds[2][:-1],bds[2][1:]):
                    self_indx2 = self_indx1 & (self_ang[2] >= k1) & (self_ang[2] < k2)
                    pho_indx2 = pho_indx1 & (pho_ang[2] >= k1) & (pho_ang[2] < k2)
                    if self_indx2.any():
                        if pho_indx2.any():
                            w1 = self.binparms['count'][self_indx2]
                            w2 = pho.binparms['count'][pho_indx2]
                            ww = w1+w2
                            ang = {}
                            for p,i in zip(self.binparms['dims'],list(range(3))):
                                ang[p] = (self_ang[i][self_indx2]*w1+pho_ang[i][pho_indx2]*w2)/ww
                            col = {}
                            for k in pho.refkey:
                                if k not in self.refkey:
                                    self._add_refkey(k)
                                col[k.lower()] = (getattr(self, k)[self_indx2]*w1+getattr(pho, k)[pho_indx2]*w2)/ww
                            ang.update(col)
                            self[self_indx2] = ang
                            self.binparms['count'][self_indx2] = ww
                    else:
                        if pho_indx2.any():
                            self.append(pho[pho_indx2])
                            self.binparms['count'] = np.append(self.binparms['count'],pho.binparms['count'][pho_indx2])

        self.binparms['boundary'] = bds

        if unit0 != self.sca.unit:
            pho.sca.unit = unit0
        if cos0 != self.sca.cos:
            pho.sca.cos = cos0

    def remove_rows(self, *args, **kwargs):
        self._data.remove_rows(*args, **kwargs)
        self.sca.remove_rows(*args, **kwargs)
        if self.geo is not None:
            self.geo.remove_rows(*args, **kwargs)
        self._set_properties()

    def remove_row(self, *args, **kwargs):
        self._data.remove_row(*args, **kwargs)
        self.sca.remove_row(*args, **kwargs)
        if self.geo is not None:
            self.geo.remove_row(*args, **kwargs)
        self._set_properties()

    def trim(self, ilim=None, elim=None, alim=None, rlim=None,
        latlim=None, lonlim=None):
        '''Trim photometric data based on the limits in (i, e, a).

        v1.0.0 : 1/11/2016, JYL @PSI
        '''
        rm = np.zeros(len(self), dtype=bool)
        for data,lim in zip([self.inc, self.emi, self.pha, self.geolat,
                self.geolon],[ilim, elim, alim, latlim, lonlim]):
            if lim is not None:
                if hasattr(lim[0],'unit'):
                    l1 = lim[0].to('deg').value
                else:
                    l1 = lim[0]
                if hasattr(lim[1],'unit'):
                    l2 = lim[1].to('deg').value
                else:
                    l2 = lim[1]
                d = data.to('deg').value
                rm |= (d<l1) | (d>l2)
        if rlim is not None:
            rm |= (self.BDR>rlim[1]) | (self.BDR<rlim[0])
        rmidx = np.where(rm)[0]
        self.remove_rows(rmidx)

    def fit(self, m0, fitter=None, **kwargs):
        '''Fit photometric data

        m0 : PhotometricModel type
          The initial model
        fitter, PhotometricModelFitter type
          The fitter used to fit the model.  Default is a MPFitter
        '''
        if fitter is None:
            fitter = PhotometricMPFitter()
        fitter(m0, self, **kwargs)
        return fitter

    def extract(self, *args, **kwargs):
        '''Extract I/F data from one geometric backplane cube and
        corresponding image cube

        See `extract_phodata` for calling interface

        return_data : bool, optional
          If `True`, then the extracted data will be returned.  Otherwise
          the extracted data will just be appened to itself.

        v1.0.0 : 1/12/2016, JYL @PSI
        '''
        return_data = kwargs.pop('return_data',None)
        data = extract_phodata(*args, **kwargs)
        self.append(data)
        if return_data:
            return data

    def populate(self, illfile, iofdata=0, **kwargs):
        '''Collect photometric data

        illfile : list of str
          Names of illumination backplane files
        outfile : str
          Format string of output files.  It is expected to contain `{0}`
          for the filter number to be inserted.
        iofdata : list of str, optional
          Names of corresponding I/F data files
        Other keywords are the same as extract_phodata()

        v1.0.0 : 1/11/2016, JYL @PSI
        '''
        from os.path import basename
        from numpy import where, concatenate, array, asarray, repeat

        illfile = np.asarray(illfile)
        if is_iterable(iofdata):
            iofdata = np.asarray(iofdata)
        else:
            iofdata = np.repeat(0, len(illdata))

        for illf, ioff in zip(illfile, iofdata):
            print('  Extracting from ', basename(illf))
            self.extract(illf, ioff, **kwargs)

    def bin(self, **kwargs):
        '''Bin photometric data'''
        return Binner(**kwargs)(self)

    def validate(self):
        v = self.sca.is_valid()
        if v.all():
            return []
        w = np.where(~v)[0]
        self.remove_rows(w)
        return list(w)


class PhotometricDataGroup(OrderedDict):
    '''Class for a group of photometric data

    Initialize from a directory or a list of files, or just initialized empty

    1. Initialize an empty instance:

      pho = PhotometricDataGroup()

    2. Initialize with a list of data, each of whose elements can initialize
    a PhotometricData instance:

      pho = PhotometricDataGroup(list_of_PhotometriData, key=key)

    In this case, a key can be provided as a keyword as the unique ID of
    input photometric data.  If not provided, the keys will be automatically
    assigned as integers starting from 0.

    3. Initialize from a data file:
      pho = PhotometricDataGroup('filename')

    Properties:
    -----------
    data : an ordered dictionary of data that can be used to initialize
    keyname : string, the name of data key

    v1.0.0 : JYL @PSI,
    '''
    def __init__(self, *args, **kwargs):

        if len(args) == 0:
            self.keyname = None
        elif len(args) == 1:
            if hasattr(args[0], '__iter__'):
                # initialize with a list of PhotometricData
                data = args[0]
                ndata = len(data)
                if len(kwargs) == 0:
                    self.keyname = 'index'
                    keys = list(range(ndata))
                else:
                    self.keyname = list(kwargs.keys())[0]
                    keys = kwargs.pop(self.keyname)
                if ndata != len(keys):
                    raise ValueError('length of data must be the same as length of keys, {0} {1} received'.format(ndata, len(keys)))
                for k, d in zip(keys,data):
                    self[k] = PhotometricData(d)
            elif isinstance(args[0], str):
                self.read(args[0])
            else:
                raise ValueError('a list of PhotometricData instance or a string is expected, {0} received'.format(type(args[0])))
        else:
            raise ValueError('1 or 2 arguments are expected, {0} received'.format(len(args)))

    def read(self, filename):
        '''load data from a file'''
        pass

    def write(self, filename, overwrite=False):
        '''write data to a file'''
        pass


_memory_size = lambda x: x*4.470348358154297e-08


class PhotometricDataGrid(object):
    '''Class for photometric data on a regular lat-lon grid

    Properties:

    _lon, _lat, _data, _count, _list
    '''

    _version = '1.0.0'

    def __init__(self, lon=None, lat=None, datafile=None, maxmem=1.5):
        '''PhotometricDataGrid class initialization

        lon, lat : array-like numbers, optional
          The longitude and latitude boundaries of grid
        datafile : str, optional
          The file to save the grid data
        maxmem : number
          Maximum memory allowed in GB.  The higher this number, the more the
          program will potentially stress the memory.  The lower this number,
          the more frequent the program will save data to disk and clean up
          the memory, therefore slower.

        v1.0.0 : 1/12/2016, JYL @PSI
        '''
        self.file = datafile
        self.max_mem = maxmem
        self.reset_grid(lon, lat)
        if self.file is not None:
            self.read()

    def reset_grid(self, lon, lat):
        '''Reset the longitude-latitude grid, therefore the whole class'''
        from collections import OrderedDict
        self._lat = lat
        self._lon = lon
        if self._lat is not None:
            if not hasattr(self._lat, 'unit'):
                self._lat = self._lat*units.deg
        if self._lon is not None:
            if not hasattr(self._lon, 'unit'):
                self._lon = self._lon*units.deg

        self._info = OrderedDict()
        self._info1d = OrderedDict()
        info_fields = np.array('file lonmin lonmax latmin latmax count incmin incmax emimin emimax phamin phamax masked loaded'.split())
        for k in info_fields:
            self._info[k] = None
            self._info1d[k] = None

        if (self._lon is not None) & (self._lat is not None):
            nlon = len(lon)-1
            nlat = len(lat)-1
            self._data = np.empty((nlat, nlon), dtype=PhotometricData)
            self._data1d = self._data.reshape(-1)
            for i in range(self.size):
                self._data1d[i] = PhotometricData()
            n = '%i' % (np.floor(np.log10(self.size))+1)
            fmt = '%'+n+'.'+n+'i'
            fnames = np.array(['phgrd_'+fmt % i+'.fits' for i in range(self.size)])
            self._info['file'] = fnames.reshape((nlat,nlon))
            for i in [1,2,3,4,6,7,8,9,10,11]:
                self._info[info_fields[i]] = np.zeros((nlat,nlon))*units.deg
            self._info['count'] = np.zeros((nlat,nlon))
            self._info['masked'] = np.ones((nlat,nlon), dtype=bool)
            self._info['loaded'] = np.zeros((nlat,nlon), dtype=bool)
            for k in list(self._info.keys()):
                self._info1d[k] = self._info[k].reshape(-1)
            self._flushed = np.zeros((nlat,nlon), dtype=bool)
            self._flushed1d = self._flushed.reshape(-1)
        else:
            self._data = None
            self._data1d = None
            self._flushed = None
            self._flushed1d = None

    def _clean_memory(self, forced=False, verbose=False):
        '''Check the size of object, free memory by deleting'''
        if not forced:
            sz = _memory_size((self._info1d['count']*self._info1d['loaded'].astype('i')).sum())
            if sz<self.max_mem:
                return False
        # free memory by deleting all PhotometricData instances
        if verbose:
            print('Cleaning memory...')
        for i in range(self.size):
            if (self._info1d['loaded'][i]) and \
                    (not self._info1d['masked'][i]) and \
                    (not self._flushed1d[i]):
                self._save_data(i)
                self._data1d[i] = None
                self._info1d['loaded'][i] = False
                self._flushed1d[i] = True
        return True

    @property
    def lon(self):
        return self._lon

    @property
    def lat(self):
        return self._lat

    @property
    def count(self):
        return self._info['count']

    @property
    def loaded(self):
        return self._info['loaded']

    @property
    def version(self):
        return self._version

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return np.size(self._data)

    def __len__(self):
        return len(self._data)

    def _process_key(self, key):
        """Process index key and return lists of indices"""
        if hasattr(key,'__iter__'):
            if len(key)!=2:
                raise KeyError('invalid index')
            i, j = key
            if isinstance(i, slice):
                ii = i.indices(len(self.lat)-1)
                i = list(range(ii[0],ii[1],ii[2]))
            if isinstance(j, slice):
                jj = j.indices(len(self.lon)-1)
                j = list(range(jj[0],jj[1],jj[2]))
        else:
            i = key
            j = list(range(len(self.lon)-1))
        y, x = np.meshgrid(i, j)
        return y, x

    def __getitem__(self, key):
        """Return value ``self[key]``"""
        y, x = self._process_key(key)
        # check whether the data to be loaded are too large
        if _memory_size(self._info['count'][y,x].sum())>self.max_mem:
            raise MemoryError('insufficient memory to load all data requested')
        # check whether memory clean is needed for this load
        loaded = self._info['loaded'].copy()
        loaded[y,x] = True
        if _memory_size((self._info['count']*loaded.astype('i')).sum())>self.max_mem:
            self._clean_memory(forced=True)
        for i, j in zip(y.flatten(),x.flatten()):
            if not self._info['loaded'][i, j]:
                self._load_data(i, j)
        out = self._data[key]
        return out

    def __setitem__(self, key, value):
        """Assign value ``value`` to ``self[key]``"""
        # process input
        valid_v = False
        if isinstance(value, (PhotometricData, int)):
            valid_v = True
            value_shape = (1,1)
        elif isinstance(value, (list, tuple, np.ndarray)):
            if np.array([isinstance(v, (PhotometricData, int)) for v in np.asanyarray(value).flatten()]).all():
                valid_v = True
                value_shape = np.shape(value)
        if not valid_v:
            raise ValueError('Only ``PhotometricData`` or array of it can be assigned.')
        y, x = self._process_key(key)
        if x.shape != value_shape:
            raise ValueError('Values to be assigned must have the same shape.')
        # assign values
        self._data[key] = value
        self._info['loaded'][key] = True
        self._info['masked'][key] = (value == 0)
        self._flushed[key] = self._info['masked'][key] | False
        for i,j in zip(y.flatten(), x.flatten()):
            self._update_property(i,j)
        # free memory if needed
        loaded = self._info['loaded']
        if _memory_size((self._info['count']*loaded.astype('i')).sum())>self.max_mem:
            self._clean_memory(forced=True)

    def _load_data(self, *args):
        '''Load data for position [i,j] or position [i] for flattened case'''
        import os
        if self.file is None:
            raise ValueError('data file not specified')
        infofile, path = self._path_name(self.file)
        cleaned = self._clean_memory()
        if len(args) == 2:
            i, j = args
            if not self._info['masked'][i,j]:
                f = path+'/'+self._info['file'][i,j]
                if os.path.isfile(f):
                    self._data[i,j] = PhotometricData(f)
                else:
                    raise IOError('Data record not found for position ({}, {}'
                        ') from file {}'.format(i,j,f))
            self._info['loaded'][i,j] = True
            self._flushed[i,j] = True
        elif len(args) == 1:
            i = args[0]
            if not self._info1d['masked'][i]:
                f = path+'/'+self._info1d['file'][i]
                if os.path.isfile(f):
                    self._data1d[i] = PhotometricData(f)
                else:
                    raise IOError('Data record not found for flattened '
                        'position ({}) from file {}'.format(i,f))
            self._info1d['loaded'][i] = True
            self._flushed1d[i] = True
        else:
            raise ValueError('2 or 3 arguments expected, {0} received'.format(len(args)+1))
        return cleaned

    def _save_data(self, *args, outfile=None, update_flush_flag=True, **kwargs):
        """Save data at specified position to output file

        If ``outfile`` is set, then data will be directly saved to the
        specified file.

        If ``outfile`` is not set, then the output file will be inferred from
        ``self.file``.

        If ``update_flush_flag`` is True, then the corresponding
        ``self._flushed`` flag will be updated to True after saving data.
        Otherwise this flag is not updated.
        """
        import os
        if outfile is None:
            if self.file is None:
                raise ValueError('data file not specified')
            infofile, path = self._path_name(self.file)
            if not os.path.isdir(path):
                os.mkdir(path)
        else:
            path = None
        if len(args) == 2:
            i, j = args
            if not self._info['masked'][i,j]:
                if path is None:
                    f = outfile
                else:
                    f = path+'/'+self._info['file'][i,j]
                if not self._info['loaded'][i,j]:
                    self._load_data(i,j)
                self._data[i,j].write(f, overwrite=True)
                if update_flush_flag:
                    self._flushed[i,j] = True
        elif len(args) == 1:
            i = args[0]
            if not self._info1d['masked'][i]:
                if path is None:
                    f = outfile
                else:
                    f = path+'/'+self._info1d['file'][i]
                if not self._info1d['loaded'][i]:
                    self._load_data(i)
                self._data1d[i].write(f, overwrite=True)
                if update_flush_flag:
                    self._flushed1d[i] = True
        else:
            raise ValueError('2 or 3 arguments expected, {0} received'.format(len(args)+1))

    def _path_name(self, outfile):
        if outfile.endswith('.fits'):
            outdir = '.'.join(outfile.split('.')[:-1])+'_dir'
        else:
            outdir = outfile+'_dir'
            outfile += '.fits'
        return outfile, outdir

    def write(self, outfile=None, overwrite=False):
        '''Write data to disk.

        If ``outfile is None``, then this method flushes all data to disk
        based on the location specified by ``self.file``.  If
        ``self.file is None``, then it will throw an exception.

        If a file name is provided via ``outfile``, then this method saves all
        data to the specified output file.  If ``self.file == None`` before
        calling this smethod, then the provided file name will be associated
        to class object for operations.  Otherwise ``self.file`` will not be
        changed, and the location specified by ``outfile`` is just an extra
        copy of the data.

        outfile : str, optional
          Output file name.  If omitted, then the program does a memory flush,
          just updating the PhotometricData at grid points that have changed
          since read into memory.
        overwrite : bool, optional
          Overwrite the existing data.  If `outfile` is `None`, then `overwrite`
          will be ignored.

        v1.0.0 : 1/12/2016, JYL @PSI
        '''
        import os

        if outfile is None:
            if self.file is None:
                raise ValueError('output file not specified')
            outfile = self.file
            flush = True
        else:
            if self.file is None:
                self.file = outfile
                flush = True
            else:
                flush = False

        outfile, outdir = self._path_name(outfile)
        if os.path.isfile(outfile):
            if overwrite:
                os.remove(outfile)
                if os.path.isdir(outdir):
                    os.system('rm -rf '+outdir)
            elif not flush:
                raise IOError('output file {0} already exists'.format(outfile))

        # save envolope information
        hdr0 = fits.Header()
        hdr0['version'] = self.version
        primary_hdu = fits.PrimaryHDU(header=hdr0)
        hdr1 = fits.Header()
        hdr1['extname'] = 'INFO'
        table_hdu = fits.BinTableHDU(Table(self._info1d), header=hdr1)
        lon_hdu = fits.ImageHDU(self.lon.value, name='lon')
        lon_hdu.header['bunit'] = str(self.lon.unit)
        lat_hdu = fits.ImageHDU(self.lat.value, name='lat')
        lat_hdu.header['bunit'] = str(self.lat.unit)
        hdu_list = fits.HDUList([primary_hdu, table_hdu, lon_hdu, lat_hdu])
        hdu_list.writeto(outfile, overwrite=True)

        # save data
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        for i in range(self.size):
            f = outdir+'/'+self._info1d['file'][i]
            if self._info1d['masked'][i]:
                if os.path.isfile(f):
                    os.remove(f)
            else:
                if flush:
                    if not self._flushed1d[i]:
                        self._save_data(i, outfile=f)
                else:
                    self._save_data(i, outfile=f, update_flush_flag=False)

    def read(self, infile=None, verbose=False, load=False):
        '''Read data from a directory or a list of files

        infile : str
          The summary file of data storage

        v1.0.0 : 1/12/2016, JYL @PSI
        '''
        import os
        if infile is None:
            if self.file is None:
                raise ValueError('input file not specified')
            else:
                infile = self.file
        infile, indir = self._path_name(infile)

        if not os.path.isfile(infile):
            raise ValueError('input file {0} not found'.format(infile))
        if not os.path.isdir(indir):
            raise ValueError('input directory {0} not found'.format(indir))

        # set up data structure and info array
        info = self.info(infile)
        self.file = infile
        self._version = info['version']
        self._lon = info['lon']
        self._lat = info['lat']
        self._info1d = info['info']
        self._info1d['loaded'][:] = False
        nlat = len(self.lat)-1
        nlon = len(self.lon)-1
        for k in list(self._info1d.keys()):
            self._info[k] = self._info1d[k].reshape((nlat,nlon))
        self._data = np.zeros((nlat,nlon), dtype=PhotometricData)
        self._data1d = self._data.reshape(-1)
        self._flushed = np.ones((nlat,nlon), dtype=bool)
        self._flushed1d = self._flushed.reshape(-1)

        # load photometric data
        if load:
            nf = nlat*nlon
            tag = -0.1
            for i in range(nf):
                if verbose:
                    prog = float(i)/nf*100
                    if prog>tag:
                        print('%3.1f%% completed: %i files read' % (prog, i))
                        tag = prog+1
                cleaned = self._load_data(i)
                if cleaned:
                    raise MemoryError('no sufficient memory available to load data')

    def port(self, indata, verbose=True):
        '''Port in data from PhotometricData instance'''
        if (self.lon is None) or (self.lat is None):
            raise ValueError('the grid parameters (lon, lat) not specified')
        if verbose:
            print('porting data from PhotometricData instance')
        nlat, nlon = self.shape
        nf = nlat*nlon
        lon = indata.geolon
        for lon1, lon2, i in zip(self.lon[:-1],self.lon[1:],list(range(nlon))):
            ww = np.where((lon>lon1)&(lon<=lon2))
            if len(ww[0]) == 0:
                continue
            d1 = indata[ww]
            lat = d1.geolat
            tag = -0.1
            for lat1, lat2, j in zip(self.lat[:-1],self.lat[1:],list(range(nlat))):
                if verbose:
                    prog = (float(i)*nlat+j)/nf*100
                    if prog>tag:
                        print('%5.1f%% completed:  lon = (%5.1f, %5.1f), lat = (%5.1f, %5.1f)' % (prog,lon1.value,lon2.value,lat1.value,lat2.value))
                        tag = prog+0.999
                ww = np.where((lat>lat1)&(lat<=lat2))
                if len(ww[0]) == 0:
                    continue
                d2 = d1[ww]
                if not self._info['loaded'][j,i]:
                    self._load_data(j,i)
                self._data[j,i].append(d2)
                self._update_property(j,i,masked=False,loaded=True)
                self._flushed[j,i] = False
                self._clean_memory(verbose=verbose)

    def _update_property(self,j,i,masked=None,loaded=None):
        if masked is not None:
            self._info['masked'][j,i] = masked
        if loaded is not None:
            self._info['loaded'][j,i] = loaded
        if self._info['masked'][j,i]:
            data = 0
        else:
            data = self[j,i]
        if isinstance(data, int) or len(data) <= 0:
            self._info['count'][j,i] = 0.
            self._info['latmin'][j,i] = 0.
            self._info['latmax'][j,i] = 0.
            self._info['lonmin'][j,i] = 0.
            self._info['lonmax'][j,i] = 0.
            self._info['incmin'][j,i] = 0.
            self._info['incmax'][j,i] = 0.
            self._info['emimin'][j,i] = 0.
            self._info['emimax'][j,i] = 0.
            self._info['phamin'][j,i] = 0.
            self._info['phamax'][j,i] = 0.
            if masked is None:
                self._info['masked'][j,i] = True
        else:
            self._info['count'][j,i] = len(data)
            if data.latlim is None:
                self._info['latmin'][j,i] = 0 * u.deg
                self._info['latmax'][j,i] = 0 * u.deg
            else:
                self._info['latmin'][j,i] = data.latlim[0]
                self._info['latmax'][j,i] = data.latlim[1]
            if data.lonlim is None:
                self._info['lonmin'][j,i] = 0 * u.deg
                self._info['lonmax'][j,i] = 0 * u.deg
            else:
                self._info['lonmin'][j,i] = data.lonlim[0]
                self._info['lonmax'][j,i] = data.lonlim[1]
            self._info['incmin'][j,i] = data.inclim[0]
            self._info['incmax'][j,i] = data.inclim[1]
            self._info['emimin'][j,i] = data.emilim[0]
            self._info['emimax'][j,i] = data.emilim[1]
            self._info['phamin'][j,i] = data.phalim[0]
            self._info['phamax'][j,i] = data.phalim[1]

    def _update_property_1d(self,i,masked=None,loaded=None):
        if masked is not None:
            self._info1d['masked'][i] = masked
        if loaded is not None:
            self._info1d['loaded'][i] = loaded
        if self._info1d['masked'][i]:
            data = 0
        else:
            if not self._info1d['loaded'][i]:
                self._load_data(i)
            data = self._data1d[i]
        if isinstance(data, int) or len(data) <= 0:
            self._info1d['count'][i] = 0.
            self._info1d['latmin'][i] = 0.
            self._info1d['latmax'][i] = 0.
            self._info1d['lonmin'][i] = 0.
            self._info1d['lonmax'][i] = 0.
            self._info1d['incmin'][i] = 0.
            self._info1d['incmax'][i] = 0.
            self._info1d['emimin'][i] = 0.
            self._info1d['emimax'][i] = 0.
            self._info1d['phamin'][i] = 0.
            self._info1d['phamax'][i] = 0.
            if masked is None:
                self._info1d['masked'][i] = True
        else:
            self._info1d['count'][i] = len(data)
            if data.latlim is None:
                self._info1d['latmin'][i] = 0 * u.deg
                self._info1d['latmax'][i] = 0 * u.deg
            else:
                self._info1d['latmin'][i] = data.latlim[0]
                self._info1d['latmax'][i] = data.latlim[1]
            if data.lonlim is None:
                self._info1d['lonmin'][i] = 0 * u.deg
                self._info1d['lonmax'][i] = 0 * u.deg
            else:
                self._info1d['lonmin'][i] = data.lonlim[0]
                self._info1d['lonmax'][i] = data.lonlim[1]
            self._info1d['incmin'][i] = data.inclim[0]
            self._info1d['incmax'][i] = data.inclim[1]
            self._info1d['emimin'][i] = data.emilim[0]
            self._info1d['emimax'][i] = data.emilim[1]
            self._info1d['phamin'][i] = data.phalim[0]
            self._info1d['phamax'][i] = data.phalim[1]

    def populate(self, *args, **kwargs):
        '''Populate photometric data grid.

        The data can be either imported from another PhotometricData, or can
        be directly extracted from backplanes and images.

        When import from another PhotometricData variable, then it has to
        contain geographic longitude and latitude information.

        If import from data files, the calling sequence is the same as
        `PhotometricData.populate`

        Keywords:
        ---------
        verbose : bool, optional
          Default is `True`

        v1.0.0 : 1/12/2016, JYL @PSI
        '''

        if (self.lon is None) or (self.lat is None):
            raise ValueError('grid parameters (lon, lat) not specified')

        if len(args) != 1:
            raise ValueError('exactly 2 arguments are required, received {0}'.format(len(args)+1))

        verbose = kwargs.pop('verbose', True)
        indata = args[0]
        if isinstance(indata, PhotometricData):
            self.port(indata, verbose=verbose)
            return

        elif (not isinstance(indata, (str,bytes))) and hasattr(indata, '__iter__'):
            if isinstance(indata[0], str):
                if verbose:
                    print('importing data from backplanes and image files')
                illfile = np.asarray(indata)
                ioffile = kwargs.pop('ioffile',None)
                if ioffile is None:
                    ioffile = np.repeat(None, len(illfile))
                else:
                    ioffile = np.asarray(ioffile)

                d = PhotometricData()
                for illf, ioff in zip(illfile, ioffile):
                    if verbose:
                        from os.path import basename
                        print('Extracting from ', basename(illf))
                    d.extract(illf, ioff, verbose=verbose, **kwargs)
                    sz = _memory_size(len(d))
                    if _memory_size(len(d)) > self.max_mem:
                        self.port(d, verbose=verbose)
                        d = PhotometricData()
                if len(d)>0:
                    self.port(d, verbose=verbose)
            else:
                raise ValueError('input parameter error')
        else:
            raise ValueError('input parameter error')
        self.write()

    def info(self, infile=None):
        '''Print out the data information'''
        if infile is None:
            ver = self.version
            lon = self.lon
            lat = self.lat
            info = dict(self._info1d)
        else:
            infile, indir = self._path_name(infile)
            from astropy.io import fits
            inf = fits.open(infile)
            ver = inf[0].header['version']
            lon = inf['lon'].data*units.Unit(inf['lon'].header['bunit'])
            lat = inf['lat'].data*units.Unit(inf['lat'].header['bunit'])
            info = Table(inf['info'].data).asdict()
            for k in 'latmin latmax lonmin lonmax incmin incmax emimin emimax phamin phamax'.split():
                info[k] = info[k]*units.deg

        return {'version': ver, 'lon': lon, 'lat': lat, 'info': info}

    def merge(self, pho, verbose=True):
        '''Merge with another PhotometricDataGroup instance'''
        if (self.lon != pho.lon).any() or (self.lat != pho.lat).any():
            raise ValueError('can''t merge with different longitude-latitude grid')
        import sys
        nlat = len(self.lat)-1
        tag = -0.1
        for i in range(len(self.lon)-1):
            for j in range(len(self.lat)-1):
                prog = (float(i)*nlat+j)/self.size*100
                if verbose:
                    if prog>tag:
                        sys.stdout.write('%5.1f%% completed\r' % prog)
                        sys.stdout.flush()
                        tag = prog+0.999
                if self._info['masked'][j,i]:
                    if not pho._info['masked'][j,i]:
                        self._data[j,i] = pho[j,i].copy()
                else:
                    if not pho._info['masked'][j,i]:
                        if not self._info['loaded'][j,i]:
                            self._load_data(j,i)
                        self._data[j,i].merge(pho[j,i])
                self._update_property(j,i,masked=False,loaded=True)
                self._flushed[j,i] = False
                self._clean_memory(verbose=verbose)
        self.write()

    def trim(self, ilim=None, elim=None, alim=None, rlim=None, verbose=True):
        '''Trim photometric data based on the limits in (i, e, a).

        v1.0.0 : 1/19/2016, JYL @PSI
        '''
        tag = -0.1
        import sys
        nlat,nlon = self.shape
        for i in range(self.size):
            prog = float(i)/self.size*100
            if verbose:
                #print prog, tag
                if prog > tag:
                    sys.stdout.write('%5.1f%% completed\r' % prog)
                    sys.stdout.flush()
                    tag = np.ceil(prog+0.1)
            if self._info1d['masked'][i]:
                continue
            if not self._info1d['loaded'][i]:
                self._load_data(i)
            d = self._data1d[i]
            rm = np.zeros(len(d), dtype=bool)
            for data,lim in zip([d.inc,d.emi,d.pha],[ilim, elim, alim]):
                if lim is not None:
                    if hasattr(lim[0],'unit'):
                        l1 = lim[0].to('deg').value
                    else:
                        l1 = lim[0]
                    if hasattr(lim[1],'unit'):
                        l2 = lim[1].to('deg').value
                    else:
                        l2 = lim[1]
                    dd = data.to('deg').value
                    rm |= (dd<l1) | (dd>l2)
            if rlim is not None:
                rm |= (d.BDR>rlim[1]) | (d.BDR<rlim[0])
            rmidx = np.where(rm)[0]
            d.remove_rows(rmidx)
            self._update_property(i/nlon,i%nlon)
            self._flushed1d[i] = False

    def fit(self, model, fitter=None, **kwargs):
        '''Fit data to model grid

        model : PhotometricModel class instance
          Initial model
        fitter : PhotometricModelFitter class instance, optional
          Fitter to be used
        **kwargs : optional parameters accepted by fitter

        v1.0.0 : 1/18/2016, JYL @PSI
        '''
        if fitter is None:
            fitter = PhotometricGridMPFitter()
        fitter(model, self, **kwargs)
        return fitter

    def convert(self, maxsize=1., verbose=True):
        '''Convert to PhotometricData type

        v1.0.0 : 1/19/2016 JYL @PSI
        '''
        import sys
        sz = _memory_size((self._info['count']*(~self._info['masked']).astype('i')).sum())
        if sz>maxsize:
            raise MemoryError('data size exceeding maximum memory size allowed')
        out = PhotometricData()
        tag = -0.1
        for i in range(self.size):
            if verbose:
                prog = float(i)/self.size*100
                if prog>tag:
                    sys.stdout.write('%5.1f%% completed\r' % prog)
                    sys.stdout.flush()
                    tag = np.ceil(prog+0.95)
            if not self._info1d['masked'][i]:
                self._load_data(i)
                out.append(self._data1d[i])
        return out

    def bin(self, outfile, **kwargs):
        """Bin photometric data in all grid.

        See `Binner` for parameters.

        Returns a `PhotometricDataGrid` object associated with output file
        ``outfile``.
        """
        import os
        out = PhotometricDataGrid(lon=self.lon.copy(), lat=self.lat.copy())
        if os.path.isfile(outfile):
            raise IOError('Output file already exists.')
        out.file = outfile
        out._info1d['masked'] = self._info1d['masked']
        for i in range(len(out._data1d)):
            if self._info1d['masked'][i]:
                out._data1d[i] = 0
            else:
                if not self._info1d['loaded'][i]:
                    self._load_data(i)
                out._data1d[i] = self._data1d[i].bin(**kwargs)
            out._update_property_1d(i, loaded=True)
        return out


class PhotometricModelFitter(object):
    '''Base class for fitting photometric data to model

    v1.0.0 : 2015, JYL @PSI
    v1.0.1 : 1/11/2016, JYL @PSI
      Removed the default fitter definition and leave it for inherited class
      Add `fitter` keyword to `__call__`.
    '''

    def __init__(self):
        self.fitted = False

    def __call__(self, model, pho, ilim=None, elim=None, alim=None, rlim=None, **kwargs):
        '''
        Parameters:
        -----------
        model : PhotometricModel instance
      The initial model to fit to data
        pho : PhotometricData instance
      The data to be fitted
        fitter : Any Fitter-like class
      The fitter to be used.  If this keyword is present, then the
      fitter class defined in this class is overrided.  If not specified,
      and no fitter class is defined in this class or its inherited class,
      an error will be thrown.
        **kwargs: Other keywords accepted by the fitter.

        v1.0.0 : 2015, JYL @PSI
        v1.0.1 : 1/11/2016, JYL @PSI
          Added fitter keywords
        '''
        if 'fitter' in kwargs:
            f = kwargs.pop('fitter')()
        else:
            if not hasattr(self, 'fitter'):
                raise ValueError('fitter not defined')
            else:
                f = self.fitter()

        self.data = pho.copy()
        self.data.validate()
        self.data.trim(ilim=ilim, elim=elim, alim=alim, rlim=rlim)
        inputs = []
        for k in model.inputs:
            inputs.append(getattr(self.data.sca,k).to('deg').value)
        bdr = self.data.BDR
        if bdr.ndim == 1:
            self.model = f(model, inputs[0], inputs[1], inputs[2], bdr, **kwargs)
            self.fit_info = f.fit_info
            self.fit = self.model(*inputs)
            self.RMS = np.sqrt(((self.fit-self.data.BDR)**2).mean())
            self.RRMS = self.RMS/self.data.BDR.mean()
            self.fitted = True
            return self.model
        else:
            self.model = []
            self.fit_info = []
            self.fit = []
            self.RMS = []
            self.RRMS = []
            self.fitted = []
            for r in bdr.T:
                self.model.append(f(model, inputs[0], inputs[1], inputs[2], r, **kwargs))
                self.fit_info.append(f.fit_info)
                self.fit.append(self.model[-1](*inputs))
                self.RMS.append(np.sqrt(((self.fit[-1]-r)**2).mean()))
                self.RRMS.append(self.RMS[-1]/r.mean())
                self.fitted.append(True)
            return self.model

    def plot(self, index=None):
        if hasattr(self.model, '__iter__'):
            if index is None:
                raise ValueError('Index is not specified.')
            fitted = self.fitted[index]
            data = self.data.BDR[:,index]
            fit = self.fit[index]
        else:
            fitted = self.fitted
            data = self.data.BDR
            fit = self.fit
        if fitted == False:
            print('No model has been fitted.')
            return
        from matplotlib import pyplot as plt
        ratio = data/fit
        figs = []
        figs.append(plt.figure(100))
        plt.clf()
        f, ax = plt.subplots(3, 1, num=100)
        for i, v, xlbl in zip(list(range(3)), [self.data.inc.value, self.data.emi.value, self.data.pha.value], ['Incidence', 'Emission', 'Phase']):
            ax[i].plot(v, ratio, 'o')
            ax[i].hlines(1, v.min(), v.max())
            pplot(ax[i], xlabel=xlbl+' ('+str(self.data.inc.unit)+')', ylabel='Measured/Modeled')
        figs.append(plt.figure(101))
        plt.clf()
        plt.plot(data, fit, 'o')
        tmp1 = data
        if isinstance(data, units.Quantity):
            tmp1 = data.value
        tmp2 = fit
        if isinstance(fit, units.Quantity):
            tmp2 = fit.value
        tmp = np.concatenate((tmp1, tmp2))
        lim = [tmp.min(),tmp.max()]
        plt.plot(lim, lim)
        pplot(xlabel='Measured BDR',ylabel='Modeled BDR')
        return figs


class PhotometricMPFitter(PhotometricModelFitter):
    '''Photometric model fitter using MPFit'''
    fitter = MPFitter


class PhotometricGridFitter(object):
    def __init__(self):
        self.fitted = False

    def __call__(self, model, data, fitter=None, ilim=None, elim=None,
        alim=None, rlim=None, latlim=None, lonlim=None, multi=None, **kwargs):
        """Fit PhotometricDataGrid to model

        model : `~astropy.modeling.Model` instance
            Model to be fitted
        data : `PhotometricDataGrid`
            Data to be fitted
        fitter : astropy fitter class instance
            Fitter used to fit the data
        ilim : 2-element array like number or `astropy.units.Quantity`
            Limit of incidence angle
        elim : 2-element array like number or `astropy.units.Quantity`
            Limit of emission angle
        alim : 2-element array like number or `astropy.units.Quantity`
            Limit of phase angle
        rlim : 2-element array like number or `astropy.units.Quantity`
            Limit of bidirectional reflectance
        latlim : 2-element array like number of `astropy.units.Quantity`
            Latitude range to be fitted
        lonlim : 2-element array like number of `astropy.units.Quantity`
            Longitude range to be fitted
        multi : number
            Number of multiple processes to run
        **kwargs : dict
            Keyword arguments accepted by `PhotometricData.fit()`

        Return : `ModelGrid` instance
        """
        verbose = kwargs.pop('verbose', True)
        if latlim is None:
            latlim = [-90, 90]
        if lonlim is None:
            lonlim = [0, 360]
        if fitter is not None:
            self.fitter = fitter
        if not hasattr(self, 'fitter'):
            raise ValueError('Fitter not defined.')
        nlat, nlon = data.shape
        self.model = ModelGrid(type(model), nlon, nlat)
        self.fit_info = np.zeros((nlat,nlon),dtype=object)
        self.fit = np.zeros((nlat,nlon),dtype=np.ndarray)
        self.RMS = np.zeros((nlat,nlon),dtype=object)
        self.RRMS = np.zeros((nlat,nlon),dtype=object)
        self.mask = np.ones((nlat,nlon),dtype=bool)
        index_boundary = (np.asarray(latlim)+90)/180*nlat
        i1 = int(np.floor(index_boundary[0]))
        i2 = int(np.ceil(index_boundary[1]))
        ii = range(i1, i2, 1)
        nii = len(ii)
        index_boundary = np.asarray(lonlim)/360*nlon
        j1 = int(np.floor(index_boundary[0]))
        j2 = int(np.ceil(index_boundary[1]))
        jj = range(j1, j2, 1)
        njj = len(jj)

        def fit_ij(i, j):
            if (not data._info['masked'][i,j]) and isinstance(data[i,j], PhotometricData):
                d = data[i,j].copy()
                d.validate()
                d.trim(ilim=ilim, elim=elim, alim=alim, rlim=rlim)
                if len(d) > 10:
                    fitter = d.fit(model, fitter=self.fitter(), verbose=False,
                        **kwargs)
                    return fitter, i, j
            return None, i, j

        def process_fit(fitter, i, j, verbose=False):
            if fitter is not None:
                # assemble to a model set
                if hasattr(fitter.model, '__iter__'):
                    params = np.array([m.parameters for m in fitter.model])
                    model_set = type(fitter.model[0])(*params.T,
                        n_models=params.shape[0])
                else:
                    model_set = fitter.model
                self.model[i,j] = model_set
                self.fit_info[i,j] = fitter.fit_info
                self.fit[i,j] = fitter.fit
                self.RMS[i,j] = fitter.RMS
                self.RRMS[i,j] = fitter.RRMS
                self.mask[i,j] = False
            if verbose:
                print('Grid ({0}, {1}) of ({2}-{3}, {4}-{5})'.format(i,j,i1,i2,
                    j1,j2), end=': ')
                if not self.mask[i,j]:
                    if len(model_set) == 1:
                        print(model_set.__repr__())
                    else:
                        print(model_set)
                else:
                    print('not fitted.')

        def worker(ii, jj, out_q):
            results = []
            for i, j in zip(ii, jj):
                results.append(fit_ij(i, j))
            out_q.put(results)

        if multi is not None:
            if verbose:
                print(f'Multiprocessing with {multi} workers')
            import multiprocessing
            from time import sleep
            iis, jjs = np.meshgrid(ii, jj, indexing='ij')
            iis = iis.flatten()
            jjs = jjs.flatten()
            niis = len(iis)
            boundaries = [int(x) for x in np.round(np.linspace(0, niis+1,
                multi+1))]
            procs = []
            out_q = multiprocessing.Queue()
            for b1,b2 in zip(boundaries[:-1], boundaries[1:]):
                p = multiprocessing.Process(target=worker,
                    args=(iis[b1:b2], jjs[b1:b2], out_q))
                procs.append(p)
                p.start()
            for i in range(multi):
                results = out_q.get()
                for r in results:
                    if r is not None:
                        process_fit(r[0], r[1], r[2], verbose=verbose)
            for p in procs:
                p.join()

        else:
            for i in ii:
                for j in jj:
                    fitter, m, n = fit_ij(i, j)
                    process_fit(fitter, m, n, verbose=verbose)

        self.fitted = True
        self.model.extra['RMS'] = self.RMS.astype(float)
        self.model.extra['RRMS'] = self.RRMS.astype(float)
        return self.model


class PhotometricGridMPFitter(PhotometricGridFitter):
    fitter = PhotometricMPFitter


class ModelGrid(object):
    """The longitude-latitude model grid class

    This class is to support the modeling of `PhotometricDataGrid` class.  It
    contains a grid of the same photometric model.
    """

    _version = '1.0.0'

    def __init__(self, m0=None, nlon=None, nlat=None, datafile=None):
        """Initialization

        m0 : Model class
            The class name of model.
        nlon : number
            The number of longitude grid points.  Non-integer will be rounded
            to integer.
        nlat : number
            The number of latitude grid points.  Non-integer will be rounded
            to integer.
        datafile : str
            Name of data file to initialize class.
       """
        self.extra = {}
        if datafile is not None:
            self.read(datafile)
            return

        self._model_class = None
        self._model_grid = None
        self._param_names = None
        self._nlon = nlon
        self._nlat = nlat
        self._mask = None
        self.model_class = m0
        self._init_model_params()

    def _init_model_params(self):
        """Initialize model class using default parameters of self.model_class
        """
        if self.model_class is not None:
            m = self.model_class()
            self._param_names = m.param_names
            if (self.nlat is not None) and (self.nlon is not None):
                self._model_grid = np.repeat(m, self.nlat*self.nlon).reshape(self.nlat,self.nlon)
                self._mask = np.ones((self.nlat,self.nlon),dtype=bool)
                for k in m.param_names:
                    self.__dict__[k] = np.repeat(getattr(m, k), self.nlat*self.nlon).reshape(self.nlat,self.nlon).tolist()

    @property
    def param_names(self):
        """Parameter names"""
        return self._param_names

    @property
    def model_grid(self):
        """A numpy array contains the model grid"""
        return self._model_grid

    @property
    def nlon(self):
        """Number of longitude grid points"""
        return self._nlon

    @property
    def nlat(self):
        """Number of latitude grid points"""
        return self._nlat

    @property
    def model_class(self):
        """Model class"""
        return self._model_class

    @model_class.setter
    def model_class(self, m0):
        """Set model class"""
        if m0 is None:
            self._model_class = None
            self._model_grid = None
            self._param_names = None
        else:
            self._model_class = m0
        self._init_model_params()

    @property
    def mask(self):
        """Model grid mask, where invalide models are masked (True)"""
        return self._mask

    @property
    def shape(self):
        """Shape of model grid"""
        if self._model_grid is None:
            return ()
        else:
            return self._model_grid.shape

    def __len__(self):
        return len(self._model_grid)

    def __getitem__(self, k):
        """Return model at specified index"""
        if self._model_grid is not None:
            return self._model_grid[k]

    def __setitem__(self, k, v):
        """Set model at specified index"""
        if self._model_grid is None:
            raise ValueError('model grid not defined yet')
        if isinstance(v, self.model_class):
            self._model_grid[k] = v
        else:
            if (len(self._model_grid[0,0]) == 1) and (not hasattr(v, '__iter__')):
                    self._model_grid[k] = self.model_class(v)
            else:
                self._model_grid[k] = self.model_class(v[0:len(self.model_grid[0,0])])
        self.mask[k] = False
        self.update_params(*k)

    def update_params(self, lat=None, lon=None, key=None, grid=True):
        """Update model parameter attribute from model grid

        lat : array-like int
            Indices of latitudes to be updated.  Default is all latitude grid
            points
        lon : array-like int
            Indices of longitudes to be updated.  Default is all longitude
            grid points
        key : array-like str
            Names of parameters to be updated.  Default is all parameters
        grid : bool
            If `True` (defult), then `lat`, `lon` will be used to generate a
            grid for the update.  Otherwise the update will be performed at
            the coordinates of each pair of elements in `lat`, `lon`.
        """
        if lat is None:
            lat = list(range(self.nlat))
        elif isinstance(lat,slice):
            n1, n2, n3 = lat.indices(self.nlat)
            lat = list(range(n1,n2,n3))
        else:
            if ulen(lat) == 1:
                lat = [lat]
        if lon is None:
            lon = list(range(self.nlon))
        elif isinstance(lon, slice):
            n1, n2, n3 = lon.indices(self.nlon)
            lon = list(range(n1,n2,n3))
        else:
            if ulen(lon) == 1:
                lon = [lon]
        if key is None:
            key = self.param_names
        else:
            if not hasattr(key,'__iter__'):
                key = np.asarray(key)
        if grid:
            for i in lat:
                for j in lon:
                    if not self.mask[i,j]:
                        for k in key:
                            self.__dict__[k][i][j] = getattr(self.model_grid[i,j],k).value
        else:
            for i, j in zip(lat, lon):
                if not self.mask[i,j]:
                    for k in key:
                        self.__dict__[k][i][j] = getattr(self.model_grid[i,j],k).value

    def write(self, filename, overwrite=False):
        """Write model grid to a FITS file

        filename : str
            The output file name
        overwrite : bool
            Overwrite existing file

        The output file is a multi-extension FITS file.

        Primary extension:
            `.header['model']` = str : model name
            `.header['parnames'] = str : tuple of model parameters
            No data
        Secondary extension has a name 'MASK'
            2D int array of shape (nlat, nlon) : model mask
        Other extensions stores the model parameters, one parameter in each
            extension, corresponding to the parameter names stored in the
            primary header `.header['parnames']`.  The extension names are
            parameter names.  The shape of data is (nlat, nlon) if the model
            at all grid points are single model, or (nlat, nlon, n_models) if
            model set.
        """
        out = fits.HDUList()
        hdu = fits.PrimaryHDU()
        hdu.header['model'] = self.model_class.name
        hdu.header['parnames'] = str(self.param_names)
        out.append(hdu)
        hdu = fits.ImageHDU(self.mask.astype('i'), name='mask')
        out.append(hdu)
        indx = np.where(~self.mask.flatten())[0][0]
        n_models = len(self._model_grid.flatten()[indx])
        if n_models == 1:
            for k in self.param_names:
                hdu = fits.ImageHDU(getattr(self, k), name=k)
                out.append(hdu)
        else:
            for k in self.param_names:
                v = getattr(self, k)
                par = np.zeros((self.nlat, self.nlon, n_models))
                for i in range(self.nlat):
                    for j in range(self.nlon):
                        if self.mask[i,j]:
                            par[i,j] = np.repeat(v[i][j], n_models)
                        else:
                            par[i,j] = v[i][j]
                hdu = fits.ImageHDU(par, name=k)
                out.append(hdu)
        ex_keys = self.extra.keys()
        if len(ex_keys) > 0:
            out[0].header['extra'] = str(tuple(ex_keys))
            for k in ex_keys:
                hdu = fits.ImageHDU(self.extra[k], name=k)
                out.append(hdu)
        out.writeto(filename, overwrite=overwrite)

    def read(self, filename):
        """Read model grid from input FITS file

        filename : str
            The name of input FITS file
        """
        hdus = fits.open(filename)
        self._model_class = eval(hdus['primary'].header['model'])
        self._param_names = eval(hdus['primary'].header['parnames'])
        self._mask = hdus['mask'].data.astype(bool)
        self._nlat, self._nlon = self.mask.shape
        if hdus[self._param_names[0]].data.ndim == 2:
            for k in self.param_names:
                self.__dict__[k] = hdus[k].data.copy()
        elif hdus[self._param_names[0]].data.ndim == 3:
            for k in self.param_names:
                self.__dict__[k] = [[hdus[k].data[i,j] for j in
                    range(self.nlon)] for i in range(self.nlat)]
            ii, jj = np.where(self.mask)
            for i,j in zip(ii,jj):
                for k in self.param_names:
                    self.__dict__[k][i][j] = self.__dict__[k][i][j][0]
        self._model_grid = np.zeros((self.nlat, self.nlon), dtype=self.model_class)
        for i in range(self.nlat):
            for j in range(self.nlon):
                if not self.mask[i,j]:
                    parms = {}
                    for k in self.param_names:
                        parms[k] = getattr(self,k)[i][j]
                    if hasattr(parms[self.param_names[0]], '__iter__'):
                        parms['n_models'] = len(parms[self.param_names[0]])
                    else:
                        parms['n_models'] = 1
                    self._model_grid[i,j] = self.model_class(**parms)
                else:
                    self._model_grid[i,j] = self.model_class()
        if 'extra' in hdus['primary'].header:
            keys = eval(hdus['primary'].header['extra'])
            for k in keys:
                self.extra[k] = hdus[k].data

class PhaseFunction(FittableModel):

    inputs = ('a',)
    outputs = ('p',)

    @staticmethod
    def check_phase_angle(alpha):
        alpha = np.squeeze(alpha)
        if hasattr(alpha, 'unit'):
            alpha = alpha.to('deg').value
        return alpha


class HG1(PhaseFunction):
    '''
 1-term Henyey-Greenstein function

 The single-term HG function has the form as in Hapke (2012), Eq. 6.5:
                             (1-g**2)
     HG_1(alpha) = -----------------------------
                   (1+2*g*cos(alpha)+g**2) **1.5
 where -1<g<1.  g=0 means isotripic scattering, g>0 forward scattering,
 and g<0 backward scattering.

 v1.0.0 : JYL @PSI, October 30, 2014
    '''

    g = Parameter(min=-1., max=1.)

    @staticmethod
    def evaluate(pha, g):
        return (1-g*g)/(1+2*g*np.cos(pha)+g*g)**1.5

    @staticmethod
    def fit_deriv(pha, g):
        cosa = np.cos(pha)
        g2 = g*g
        return (g*(g2-5)-(g2+3)*cosa)/(1+2*cosa*g+g2)**2.5


class HG2(PhaseFunction):
    '''
 2-term Henyey-Greenstein function

 The two-parameter HG function has the form as in Hapke (2012), Eq.
 6.7a:
                                (1-b**2)
     HG_b(alpha; b) = -----------------------------
                      (1-2*g*cos(alpha)+g**2) **1.5
                                (1-g**2)
     HG_f(alpha; b) = -----------------------------
                      (1+2*g*cos(alpha)+g**2) **1.5
     HG_2(alpha) = (1+c)/2 * HG_b(alpha; b) + (1-c)/2 * HG_f(alpha; b)

 The HG_b describes backward lobe and the HG_f describes forward lobe.

 v1.0.0 : JYL @PSI, October 30, 2014
    '''

    b = Parameter(min=0., max=1.)
    c = Parameter(min=-1., max=1.)

    @staticmethod
    def evaluate(pha, b, c):
        cosa = np.cos(pha)
        b2 = b*b
        num = (1-b2)
        d1 = (1+b2)
        d2 = 2*b*cosa
        hgb = num/(d1-d2)**1.5
        hgf = num/(d1+d2)**1.5
        return 0.5*((1+c)*hgb + (1-c)*hgf)

    @staticmethod
    def fit_deriv(pha, b, c):
        cosa = np.cos(pha)
        b2 = b*b
        num = (1-b2)
        n1 = (b2-5)*b
        n2 = (b2+3)*cosa
        d1 = 1+b2
        d2 = 2*cosa*b
        hgb = num/(d1-d2)**1.5
        hgf = num/(d1+d2)**1.5
        dhgb = (n1+n2)/(d1-d2)**2.5
        dhgf = (n1-n2)/(d1+d2)**2.5
        d_b = 0.5*((1+c)*dhgb + (1-c)*dhgf)
        d_c = 0.5*hgb-0.5*hgf
        return [d_b, d_c]


class HG3(PhaseFunction):
    '''
 3-term Henyey-Greenstein function

 The three-parameter HG function has the form as in Hapke (2012), Eq.
 6.7b:
     HG_3(alpha) = (1+c)/2 * HG_b(alpha; b1) + (1-c)/2 * HG_f(alpha; b2)
 The range of values of parameters for two-parameter and three-parameter
 HG functions are: 0<=b, b1, b2<=1, and no constraints for c except that
 phase function has to be non-negative everywhere.

 v1.0.0 : JYL @PSI, October 30, 2014
    '''

    b1 = Parameter(min=0., max=1.)
    b2 = Parameter(min=0., max=1.)
    c = Parameter(min=-1., max=1.)

    @staticmethod
    def evaluate(pha, b1, b2, c):
        return 0.5*(1+c)*HG1.evaluate(pha,-b1) + 0.5*(1-c)*HG1.evaluate(pha,b2)

    @staticmethod
    def fit_deriv(pha, b1, b2, c):
        cosa = np.cos(pha)
        b1sq = b1*b1
        b2sq = b2*b2
        hgb = (1-b1sq)/(1-2*cosa*b1+b1sq)**1.5
        hgf = (1-b2sq)/(1+2*cosa*b2+b2sq)**1.5
        d_b1 = ((b1sq-5)*b1+(b1sq+3)*cosa)/(1-2*cosa*b1+b1sq)**2.5
        d_b2 = ((b2sq-5)*b2-(b2sq+3)*cosa)/(1+2*cosa*b2+b2sq)**2.5
        d_c = 0.5*hgb - 0.5*hgf
        return [d_b1, d_b2, d_c]


class Exponential(PhaseFunction):
    '''
 Exponential phase function model in magnitude (Takir et al. 2014)
    f(alpha) = exp(beta * alpha + gamma * alpha**2 + delta * alpha**3)

 v1.0.0 : JYL @PSI, October 17, 2014
    '''

    beta = Parameter(default=-.1, max=0.)
    gamma = Parameter(default=1e-4)
    delta = Parameter(default=1e-8)

    @staticmethod
    def evaluate(alpha, beta, gamma, delta):
        #alpha = self._check_unit(alpha)
        alpha = np.squeeze(alpha)
        if isinstance(alpha, units.Quantity):
            alpha = alpha.to('deg').value
        alpha2 = alpha*alpha
        return np.exp(beta*alpha+gamma*alpha2+delta*alpha*alpha2)

    @staticmethod
    def fit_deriv(alpha, beta, gamma, delta):
        #alpha = self._check_unit(alpha)
        alpha = np.squeeze(alpha)
        if isinstance(alpha, units.Quantity):
            alpha = alpha.to('deg').value
        exp = Exp.evaluate(alpha, beta, gamma, delta)
        alpha2 = alpha*alpha
        return [exp*alpha, exp*alpha2, exp*alpha*alpha2]


class LinMagnitude(PhaseFunction):
    '''
 Linear magnitude phase function model (Li et al., 2009; 2013)
    f(alpha) = 10**(-0.4*beta*alpha)
    alpha is in degrees

 v1.0.0: JYL @PSI, October 17, 2014
    '''

    beta = Parameter(default=0.04, min=0.)

    @staticmethod
    def evaluate(alpha, beta):
        alpha = PhaseFunction.check_phase_angle(alpha)
        return 10**(-.4*beta*alpha)

    @staticmethod
    def fit_deriv(alpha, beta):
        alpha = PhaseFunction.check_phase_angle(alpha)
        d_beta = -0.4*alpha*LinMagnitude.evaluate(alpha,beta)*np.log(10)
        return d_beta


class PolyMagnitude(PhaseFunction):
    '''
 3rd order polynomial magnitude phase function (Takir et al., 2014)
    f(alpha) = 10**(-0.4*(beta*alpha + gamma*alpha**2 + delta*alpha**3))

 v1.0.0 : JYL @PSI, October 17, 2014
 '''

    beta = Parameter(default=0.02,min=0.)
    gamma = Parameter(default=0.)
    delta = Parameter(default=0.)

    @staticmethod
    def evaluate(alpha, beta, gamma, delta):
        alpha = PhaseFunction.check_phase_angle(alpha)
        alpha2 = alpha*alpha
        return 10**(-.4*(beta*alpha+gamma*alpha2+delta*alpha*alpha2))

    @staticmethod
    def fit_deriv(alpha, beta, gamma, delta):
        alpha = PhaseFunction.check_phase_angle(alpha)
        c = -0.4*np.log(10)*Poly3Mag.evaluate(alpha, beta, gamma, delta)
        alpha2 = alpha*alpha
        return [c*alpha, c*alpha2, c*alpha*alpha2]


class ROLOPhase(PhaseFunction):
    '''ROLO polynomial phase function model (Hillier, Buratti, & Hill,
 1999, Icarus 141, 205-225; Buratti et al., 2011).

   f(pha) = C0 * exp(-C1 * pha) + A0 + A1*pha + A1*pha**2 +
            A3 * pha**3 + A4 * pha**4

 v1.0.0 : JYL @PSI, December 23, 2013
    '''

    C0 = Parameter(default=0.1, min=0.)
    C1 = Parameter(default=0.1, min=0.)
    A0 = Parameter(default=0.2, min=0.)
    A1 = Parameter(default=1e-6)
    A2 = Parameter(default=1e-8)
    A3 = Parameter(default=1e-10)
    A4 = Parameter(default=1e-12)

    @staticmethod
    def evaluate(pha, c0, c1, a0, a1, a2, a3, a4):
        pha = PhaseFunction.check_phase_angle(pha)
        pha2 = pha*pha
        return c0*np.exp(-c1*pha)+a0+a1*pha+a2*pha2+a3*pha*pha2+a4*pha2*pha2

    @staticmethod
    def fit_deriv(pha, c0, c1, a0, a1, a2, a3, a4):
        pha = PhaseFunction.check_phase_angle(pha)
        pha2 = pha*pha
        dc0 = np.exp(-c1*pha)
        if hasattr(pha, '__iter__'):
            dda = np.ones(len(pha))
        else:
            dda = 1.
        return [dc0, -c0*c1*dc0, dda, pha, pha2, pha*pha2, pha2*pha2]


class Akimov(PhaseFunction):
    '''Akimov 2-parameter phase function model (Akimov, 1998, Kinem.
 Phys. Celest. Bodies 4, 3-10).

    f(alpha) = [exp(-mu1 * alpha) + m * exp(-mu2 * alpha)] / (1+m)

 where mu1, mu2, and m are two parameters

 v1.0.0 : JYL @PSI, October 27, 2014
    '''

    A = Parameter(default=0.1, min=0.)
    mu1 = Parameter(default=0.2, min=0.)
    mu2 = Parameter(default=0.05, min=0.)
    m = Parameter(default=0.2, min=0.)

    @staticmethod
    def evaluate(alpha, A, mu1, mu2, m):
        alpha = PhaseFunction.check_phase_angle(alpha)
        return A*(np.exp(-mu1*alpha) + m*np.exp(-mu2*alpha))/(1+m)

    @staticmethod
    def fit_deriv(alpha, A, mu1, mu2, m):
        alpha = PhaseFunction.check_phase_angle(alpha)
        a, b = np.exp(-mu1*alpha), np.exp(-mu2*alpha)
        return [(np.exp(-mu1*alpha) + m*np.exp(-mu2*alpha))/(1+m), -alpha*a/(1+m), -alpha*m*b/(1+m), (b-a)/(1+m)**2]


class LinExponential(PhaseFunction):
    '''Linear-Exponential phase function model (Piironen 1994, Kaasalainen et al. 2001, 2003).

    f(alpha) = a * exp(-alpha/d) + b + k * alpha

 where a, d, b, k are parameters.  The unit of alpah is degree.

 v1.0.0 : JYL @PSI, January 10, 2016
    '''

    a = Parameter(default=1.0, min=0.)
    d = Parameter(default=5.0, min=0.)
    b = Parameter(default=0., min=0.)
    k = Parameter(default=-0.1, max=0.)

    @staticmethod
    def evaluate(alpha, a, d, b, k):
        alpha = PhaseFunction.check_phase_angle(alpha)
        return a*np.exp(-alpha/d)+b+k*alpha

    #@staticmethod
    #def fit_deriv(alpha, a, d, b, k):
    #   da = np.exp(-(alpha/d))
    #   dd = a*alpha/(d*d)*da
    #   db = 1.
    #   dk = alpha
    #   return [da, dd, db, dk]


class DiskFunction(FittableModel):

    inputs = ('i', 'e')
    outputs = ('d',)


class Lambert(DiskFunction):
    '''
 Lambert photometric model
   r = A * cos(i)

 v1.0.0 : JYL @PSI, October 27, 2014
    '''

    A = Parameter(default=1., min=0.0, max=1.0)

    @staticmethod
    def evaluate(i, e, A):
        i = _2rad(i)
        return A*np.cos(i)*recipi

    @staticmethod
    def fit_deriv(i, e, A):
        i = _2rad(i)
        return np.cos(i)*recipi


class LS(DiskFunction):
    '''
 Lommel-Seeliger model

 angles are in radiance or quantity

 v1.0.0 : JYL @PSI, October 27, 2014
    '''

    A = Parameter(default=1., min=0.)

    @staticmethod
    def evaluate(i, e, A):
        i, e = _2rad(i, e)
        mu0 = np.cos(i)
        mu = np.cos(e)
        return A*mu0/(mu0+mu)*recipi

    @staticmethod
    def fit_deriv(i, e, A):
        i, e = _2rad(i, e)
        mu0 = np.cos(i)
        mu = np.cos(e)
        return mu0/(mu0+mu)*recipi


class LunarLambert(DiskFunction):
    '''
 Lunar-Lambert model

 v1.0.0 : 1/19/2016, JYL @PSI
    '''
    A = Parameter(default=1., min=0.)
    L = Parameter(default=0.5, min=0., max=1.)

    @staticmethod
    def evaluate(i, e, A, L):
        i, e = _2rad(i, e)
        mu0 = np.cos(i)
        mu = np.cos(e)
        return A*(L*2*mu0/(mu0+mu)+(1-L)*mu0)

    @staticmethod
    def fit_deriv(i, e, A, L):
        i, e = _2rad(i, e)
        mu0 = np.cos(i)
        mu = np.cos(e)
        lunar = 2*mu0/(mu0+mu)
        dda = L*lunar+(1-L)*mu0
        ddl = A*(lunar-mu0)
        return [dda, ddl]


class Minnaert(DiskFunction):
    '''Minnaert disk function'''

    A = Parameter(default=0.5, min=0)
    k = Parameter(default=0.5, min=0., max=1.)

    @staticmethod
    def evaluate(i, e, A, k):
        i, e = _2rad(i, e)
        mu0 = np.cos(i)
        mu = np.cos(e)
        return A*mu0**k*mu**(k-1)


class AkimovDisk(FittableModel):
    '''Akimov parameterless disk function
    Parameters:
        `alpha`: phase angle
        `beta`: photometric latitude
        `gamma`: photometric longitude

    Model follows Eq. (19) in Shkuratov et al. 2011, PSS 59, 1326-1371

 v1.0.0 : 1/19/2016, JYL @PSI
    '''

    inputs = ('pha','lat','lon')
    outputs = ('d',)

    A = Parameter(default=1., min=0.)
    eta = Parameter(default=1., min=0., max=1.)

    @staticmethod
    def D(pha, lat, lon, eta):
        pha, lat, lon = _2rad(pha, lat, lon)
        return np.cos(pha/2)*np.cos(np.pi/(np.pi-pha)*(lon-pha/2))*np.cos(lat)**(eta*pha/(np.pi-pha))/np.cos(lon)*recipi

    @staticmethod
    def evaluate(pha, lat, lon, A, eta):
        return A*AkimovDisk.D(pha, lat, lon, eta)


class PhotometricModel(FittableModel):
    '''Base class for photometric models'''

    inputs = ('inc', 'emi', 'pha')
    outputs = ('r',)

    def BDR(self, inc, emi, pha):
        '''Bidirectional reflectance'''
        return self(inc, emi, pha)

    def RADF(self, inc, emi, pha):
        '''Radiance factor'''
        return self(inc, emi, pha)*np.pi

    IoF = RADF

    def BRDF(self, inc, emi, pha):
        '''Bidirectional reflectance distribution function'''
        return self(inc, emi, pha)/np.cos(_2rad(inc))

    def REFF(self, inc, emi, pha):
        '''Reflectance factor (reflectance coefficient)'''
        return self(inc, emi, pha)*np.pi/np.cos(_2rad(inc))

    def normref(self, emi):
        '''Normal reflectance'''
        return self(emi, emi, 0)

    def normalb(self, emi):
        '''Normal albedo'''
        return self.RADF(emi, emi, 0)


class MinnaertPoly3(PhotometricModel):
    '''Minnaert model + 3rd order polynomial phase function

    Following the definition in Bennu DRA:

        r(i,e,a) = A * f * mu**(k-1) * mu0**(k)

    where
        f = 10**(-0.4 * (beta*a + gamma*a**2 + delta*a**3))
        k = k0 + b*a

    i, e, a are all in degrees.

    '''

    A = Parameter(default=0.5, min=0)
    beta = Parameter(default=1.)
    gamma = Parameter(default=0.)
    delta = Parameter(default=0.)
    k = Parameter(default=0.5, min=0., max=1.)
    b = Parameter(default=0., min=0.)

    @staticmethod
    def evaluate(inc, emi, pha, A, beta, gamma, delta, k, b):
        AA = A*10**(-0.4*(beta*pha+gamma*pha*pha+delta*pha*pha*pha))
        kk = k+b*pha
        return Minnaert(AA, kk)(inc,emi)


class ROLOModel(PhotometricModel):
    '''
 ROLO model:

    BDR = mu0/(mu0+mu) * f(pha)
    f(pha) = C0 * exp(-C1 * pha) + A0 + A1*pha + A1*pha**2 +
                                   A3 * pha**3 + A4 * pha**4
    '''

    C0 = Parameter(default=0.1, min=0.)
    C1 = Parameter(default=0.1, min=0.)
    A0 = Parameter(default=0.2, min=0.)
    A1 = Parameter(default=1e-6)
    A2 = Parameter(default=1e-8)
    A3 = Parameter(default=1e-10)
    A4 = Parameter(default=1e-12)

    @staticmethod
    def evaluate(inc, emi, pha, C0, C1, A0, A1, A2, A3, A4):
        i, e = _2rad(inc, emi)
        mu0 = np.cos(inc)
        mu = np.cos(emi)
        f = ROLOPhase(C0, C1, A0, A1, A2, A3, A4)
        return mu0/(mu0+mu)*f(pha)

    def geoalb(self):
        return self.normalb(0.)


class LS_Akimov(PhotometricModel):
    '''
 Lommel-Seeliger disk function and Akimov phase function
    '''
    A = Parameter(default=0.1, min=0.)
    mu1 = Parameter(default=0.2)
    mu2 = Parameter(default=0.05)
    m = Parameter(default=0.2)

    @staticmethod
    def evaluate(inc, emi, pha, A, mu1, mu2, m):
        d = LS(A)
        f = Akimov(mu1, mu2, m)
        return d(inc,emi)*f(pha)


class LS_LinMag(PhotometricModel):
    '''
 Lommel-Seeliger disk function and linear magnitude phase function
    '''
    A0 = Parameter(default=0.1, min=0.)
    beta = Parameter(default=0.04, min=0.)

    @staticmethod
    def evaluate(inc,emi,pha,A0,beta):
        d = LS(A0)
        f = LinMagnitude(beta)
        return d(inc,emi)*f(pha)


class Akimov_LinMag(PhotometricModel):
    '''Akimov disk function and linear magnitude phase function
    '''

    inputs = ('pha', 'lat', 'lon')
    A0 = Parameter(default=0.1, min=0.)
    beta = Parameter(default=0.04, min=0.)

    @staticmethod
    def evaluate(pha,lat,lon,A0,beta):
        d = AkimovDisk(A0)
        f = LinMagnitude(beta)
        return d(pha,lat,lon)*f(pha)


class LambertPolyMag(PhotometricModel):
    '''Lambert disk-function + Polynomial magnitude model'''

    A = Parameter(default=1., min=0., max=1.)
    beta = Parameter(default=0.02, min=0.)
    gamma = Parameter(default=0.)
    delta = Parameter(default=0.)

    @staticmethod
    def evaluate(inc, emi, pha, A, beta, delta, gamma):
        return Lambert(A)(inc, emi)*PolyMagnitude(beta, gamma, delta)(pha)


class CompositePhotometricModel(PhotometricModel):
    '''Photometric model composed of a disk-function model and a
    phasef function model

    v1.0.0 : JYL @PSI, 2/25/2015'''

    #diskfunc = None
    #phasefunc = None

    def __init__(self, diskfunc, phasefunc, **kwargs):
        #if not isinstance(diskfunc, DiskFunction):
        #   raise TypeError('a DiskFunction type is expected')
        #if not isinstance(phasefunc, PhaseFunction):
        #   raise TypeError('a PhaseFunction type is expected')
        self.diskfunc = diskfunc
        self.phasefunc = phasefunc
        for p in diskfunc.param_names:
            setattr(self, p, getattr(diskfunc, p))
        for p in phasefunc.param_names:
            setattr(self, p, getattr(phasefunc, p))
        super(CompositePhotometricModel, self).__init__(**kwargs)

    def evaluate(self, inc, emi, pha, *par):
        ndp = len(diskfunc.param_names)
        d = self.diskfunc.evaluate(inc, emi, pha, *par[:ndp])
        f = self.phasefunc.evaluate(inc, emi, pha, *par[ndp:])
        return d*f


def PhotEqGeom(pha=None, inc=None, emi=None, step=1.):
    '''Return the (i, e) along the photometric equator for a given
    phase angle, or incidence angle, or emission angle'''
    if pha is not None:
        if pha>=0:
            emi = np.linspace(pha-90., 90., np.ceil((180-pha)/step)+1)
        else:
            emi = np.linspace(-90., 90+pha, np.ceil((180+pha)/step)+1)
        inc = emi-pha
        return inc, emi, np.ones_like(inc)*pha
    elif inc is not None:
        emi = np.linspace(-90, 90, np.ceil(180/step)+1)
        pha = inc-emi
        return np.ones_like(emi)*inc, emi, pha
    elif emi is not None:
        inc = np.linspace(-90., 90, np.ceil(180/step)+1)
        pha = inc-emi
        return inc, np.ones_like(inc)*emi, pha
    else:
        raise ValueError('one of the three parameters `pha`, `inc`, or `emi` must be specified')


def ref2mag(ref, radius, msun=-26.74):
    '''Convert average bidirectional reflectance to reduced magnitude'''

    Q = False
    if isinstance(ref, units.Quantity):
        ref = ref.value
        Q = True
    if isinstance(radius, units.Quantity):
        radius = radius.to('km').value
        Q = True
    if isinstance(msun, units.Quantity):
        msun = msun.to('mag').value
        Q = True

    mag = msun-2.5*np.log10(ref*np.pi*radius*radius*units.km.to('au')**2)
    if Q:
        return mag*units.mag
    else:
        return mag


def mag2ref(mag, radius, msun=-26.74):
    '''Convert reduced magnitude to average bidirectional reflectance'''

    Q = False
    if isinstance(mag, units.Quantity):
        mag = mag.value
        Q = True
    if isinstance(radius, units.Quantity):
        radius = radius.to('km').value
        Q = True
    if isinstance(msun, units.Quantity):
        msun = msun.to('mag').value
        Q = True

    ref = 10**((msun-mag)*0.4)/(np.pi*radius*radius*units.km.to('au')**2)
    if Q:
        return ref/units.sr
    else:
        return ref


def biniof(inc, emi, pha, iof, di, de, da, binned=None, verbose=False):
    '''Bin I/F data in parameter space

    Return
    ------
    incb, emib, phab, iofb, incerr, emierr, phaerr, ioferr

    v1.0.1 : 11/1/2015, JYL @PSI
      Improved the procedure
    '''

    inc = np.asanyarray(inc).flatten()
    emi = np.asanyarray(emi).flatten()
    pha = np.asanyarray(pha).flatten()
    iof = np.asanyarray(iof).flatten()

    if verbose:
        print('Bin photometric data to grid:')
        print('  i: from {0} to {1} with bin size {2}'.format(inc.min(), inc.max(), di))
        print('  e: from {0} to {1} with bin size {2}'.format(emi.min(), emi.max(), de))
        print('  a: from {0} to {1} with bin size {2}'.format(pha.min(), pha.max(), da))

    incbin = []
    emibin = []
    phabin = []
    iofbin = []
    incerr = []
    emierr = []
    phaerr = []
    ioferr = []
    count = []

    for a in np.arange(pha.min(), pha.max(), da):
        a_idx = (pha >= a) & (pha < a+da)
        if a_idx.any():
            for i in np.arange(inc.min(), inc.max(), di):
                i_idx = a_idx & (inc >= i) & (inc < i+di)
                if i_idx.any():
                    for e in np.arange(emi.min(), emi.max(), de):
                        e_idx = i_idx & (emi >= e) & (emi < e+de)
                        if e_idx.any():
                            inc_in = inc[e_idx]
                            emi_in = emi[e_idx]
                            pha_in = pha[e_idx]
                            iof_in = iof[e_idx]
                            incbin.append(inc_in.mean())
                            emibin.append(emi_in.mean())
                            phabin.append(pha_in.mean())
                            iofbin.append(iof_in.mean())
                            count.append(len(inc_in))
                            if count[-1] > 1:
                                incerr.append(inc_in.std())
                                emierr.append(emi_in.std())
                                phaerr.append(pha_in.std())
                                ioferr.append(iof_in.std())
                            else:
                                incerr.append(0.)
                                emierr.append(0.)
                                phaerr.append(0.)
                                ioferr.append(0.)

    return np.asarray(incbin), np.asarray(emibin), np.asarray(phabin), np.asarray(iofbin), np.asarray(incerr), np.asarray(emierr), np.asarray(phaerr), np.asarray(ioferr), np.asarray(count)


class Binner(object):
    '''Binner class to bin photometric data

 v1.0.0 : 11/1/2015, JYL @PSI
    '''

    def __init__(self, dims=['inc','emi','pha'], bins=(5,5,5), boundary=None):
        #if not isinstance(sca, ScatteringGeometry):
        #   raise TypeError('ScatteringGeometry class instance is expected.')
        import warnings
        self.dims = dims
        if boundary is not None:
            if len(boundary) != 3:
                warnings.warn('Invalid `boundary` is ignored.')
                boundary = None
        self.boundary = boundary
        # if `boundary` is specified, then `bins` will be ignored
        if self.boundary == None:
            self.bins = bins
        else:
            if bins != None:
                warnings.warn('`bins` is ignored when `boundary` is set')
                self.bins = None


    def bin(self, pho, verbose=False):
        if verbose:
            print()

        data = []
        for p in self.dims:
            data.append(getattr(pho.sca, p).value)
        data.append(getattr(pho,pho.refkey[0]))

        method = 'boundary'
        if self.boundary == None:
            self.boundary = []
            method = 'bin'
            for b, d in zip(self.bins, data):
                self.boundary.append(np.arange(d.min(), d.max()+b, b))

        if verbose:
            print('Bin {0} photometric data points to grid:'.format(len(data[0])))
            for p, d, bn, bd in zip(self.dims, data[:3], self.bins, self.boundary):
                print('  {0} from {1} to {2} with {3}: {4}'.format(p, d.min(), d.max(), method, condition(method=='bin', bn, bd)))

        binned = [[], [], [], []]
        error = [[], [], [], []]
        count = []

        for a1,a2 in zip(self.boundary[0][:-1],self.boundary[0][1:]):
            a_idx = (data[0] >= a1) & (data[0] < a2)
            if a_idx.any():
                for i1, i2 in zip(self.boundary[1][:-1],self.boundary[1][1:]):
                    i_idx = a_idx & (data[1] >= i1) & (data[1] < i2)
                    if i_idx.any():
                        for e1, e2 in zip(self.boundary[2][:-1],self.boundary[2][1:]):
                            e_idx = i_idx & (data[2] >= e1) & (data[2] < e2)
                            if e_idx.any():
                                data_in = [data[i][e_idx] for i in range(4)]
                                [binned[i].append(data_in[i].mean(axis=0)) for i in range(4)]
                                count.append(len(data_in[0]))
                                if count[-1] > 1:
                                    [error[i].append(data_in[i].std(axis=0)) for i in range(4)]
                                else:
                                    [error[i].append(0.) for i in range(3)]
                                    if data[3].ndim == 1:
                                        error[3].append(0.)
                                    else:
                                        error[3].append(np.zeros_like(data_in[3][0]))

        parms = {'dims': self.dims, 'bins': self.bins, 'boundary': self.boundary, 'count': np.array(count)}
        keys = {}
        for i in range(3):
            keys[self.dims[i]] = binned[i]
        keys[pho.refkey[0].lower()] = binned[3]
        keys['type'] = 'binned'
        keys['unit'] = pho.sca.unit
        keys['binparms'] = parms

        return PhotometricData(**keys)


    def __call__(self, pho, verbose=False):
        return self.bin(pho, verbose=verbose)


class ROLOModelFitter(PhotometricModelFitter):
    '''ROLO model fitter class'''

    def __call__(self, pho, model=None, **kwargs):
        if model is None:
            model = ROLOModel()
        return super(ROLOModelFitter, self).__call__(model, pho, **kwargs)

    def plot(self):
        figs = super(ROLOModelFitter, self).plot()
        from matplotlib import pyplot as plt
        figs.append(plt.figure(102))
        plt.clf()
        mu0 = np.cos(self.data.inc)
        mu = np.cos(self.data.emi)
        k = (mu0+mu)/mu0
        pha = self.data.pha.value
        plt.plot(pha, self.data.BDR*k, 'o')
        ph = np.linspace(pha.min(),pha.max(),300)
        pars = {}
        for i in range(len(self.model.parameters)):
            pars[self.model.param_names[i]] = self.model.parameters[i]
        model = ROLOPhase(**pars)
        plt.plot(ph, model(ph))
        pplot(xlabel='Phase ('+str(self.data.pha.unit)+')',ylabel='Phase Function')
        return figs


def fitROLO(pho, m0=None, fit_info=False, **kwargs):
    '''Fit photometric data to ROLO model.

 Parameters:
 -----------
 pho : PhotometricData instance
   The photometric data to be fitted
 m0 : ROLO instance, optional
   The intial guess of ROLO model
 fit_info : bool, optional
   If `True`, then a tuple will be returned (m, fit_info), where
   `fit_info` contains the fit_info structure of the fitter.

 Returns:
 --------
 m : ROLO model instance with the best-fit model parameters
 or (m, fit_info)

 v1.0.0 : Dec 7, 2015, JYL @PSI
    '''

    if m0 is None:
        m0 = ROLOModel()
    f = PhotometricModelFitter()
    m = f(m0, pho, **kwargs)
    if fit_info:
        return m, f.fitter.fit_info
    else:
        return m


import astropy
class Mul(astropy.modeling.mappings.Mapping):
    def __init__(self, *args, **kwarg):
        super(Mul,self).__init__(*args,**kwarg)
        self._outputs = 'y',
    def __call__(self,*args,**kwarg):
        ys = super(Mul,self).__call__(*args,**kwarg)
        print(ys)
        y = 1.
        for x in list(ys):
            y *= x
        return y


def extract_phodata(illfile, iofdata=0, maskdata=None, backplanes=None, binsize=None, verbose=True):
    '''Extract I/F data from one geometric backplane cube and
 corresponding image cube

 illdata : str
   The ISIS cube file that contains the illumination backplanes
 iofdata : str, optional
   The I/F image file, could be ISIS cube file or FITS file.  If
   specified by `iofdata` but not found, an error will be generated.
   If not specified, then the I/F data will be searched in the
   `illfile` before all geometry backplanes.  In this case, if no I/F
   data found, no error will be generated and the I/F data will simply
   not collected.
 maskfile : str
   ISIS cube or FITS file storing a mask.  If set, then override the mask
   stored in `illfile` cube file
 backplanes : list of str, optional
   The names of backplanes to be extracted.  Default is ['Phase Angle',
   'Local Emission Angle', 'Local Incidence Angle', 'Latitude', 'Longitude']
 binsize : int, optional
   The binning size for images before extraction

 Returns:
 --------
 A PhotometricData class instance containing the data extracted.

 v1.0.0 : 1/12/2016, JYL @PSI
   Adopted from `extact_phodata` from Dawn.py package
    '''

    from jylipy.pysis_ext import CubeFile
    from jylipy.core import rebin, readfits
    from os.path import basename, isfile, splitext
    from numpy import zeros, squeeze, where, empty, repeat, newaxis, array, rollaxis, concatenate, isfinite

    # List of all possible geometric backplanes generated by isis.phocube
    geo_backplanes = {'Phase Angle':'pha', 'Local Emission Angle':'emi', 'Local Incidence Angle':'inc', 'Latitude':'lat', 'Longitude':'lon', 'Incidence Angle':'inc0', 'Emission Angle':'emi0', 'Pixel Resolution':'res', 'Line Resolution':'lres', 'Sample Resolution':'sres', 'Detector Resolution':'dres', 'North Azimuth':'noraz', 'Sun Azimuth':'sunaz', 'Spacecraft Azimuth':'scaz', 'OffNadir Angle':'offang', 'Sub Spacecraft Ground Azimuth':'subscaz', 'Sub Solar Ground Azimuth':'subsaz', 'Morphology':'mor', 'Albedo':'alb', 'Mask': 'mask'}

    if backplanes is None:
        backplanes = ['Phase Angle', 'Local Emission Angle', 'Local Incidence Angle', 'Latitude', 'Longitude']

    # Read in and select illumination cube
    illcub = CubeFile(illfile)
    ill0 = illcub.apply_numpy_specials()
    illbackplanes = [x.strip('"') for x in illcub.label['IsisCube']['BandBin']['Name']]
    for b in backplanes:
        if b not in illbackplanes:
            if verbose:
                print('Warning: backplane {0} not found in input cube, dropped'.format(b))
            backplanes.pop(backplanes.index(b))
    ill = np.array([ill0[illbackplanes.index(backplanes[i])] for i in range(len(backplanes))])

    # Read in image data
    if isinstance(iofdata, (str, bytes)):
        if not isfile(iofdata):
            raise IOError('I/F data not found.')
        if iofdata.lower().endswith(('.fits','.fit')):
            im = readfits(iofdata, verbose=verbose)
        elif iofdata.lower().endswith('.img'):
            im = PDS.readpds(iofdata)
        else:
            im = CubeFile(iofdata).apply_numpy_specials()
        dim = im.shape
        if len(dim) == 2:
            imnames = ['Data0']
        else:
            imnames = ['Data{0}'.format(i) for i in range(dim[0])]
    else:
        if hasattr(iofdata, '__iter__'):
            im = array([ill0[i] for i in iofdata])
            imnames = [illbackplanes[i] for i in iofdata]
        else:
            im = ill0[iofdata]
            imnames = illbackplanes[iofdata]

    # Read mask
    if maskdata is not None:
        if isinstance(maskdata, (str, bytes)):
            if maskdata.lower().endswith(('.fits','.fit')):
                mask = readfits(maskdata,verbose=False).astype(bool)
            else:
                mask = CubeFile(maskdata)
                mask = mask.apply_numpy_specials().astype(bool)
        elif isinstance(maskdata, int):
            mask = ill0[maskdata].astype(bool)
        else:
            print('Warning: `maskdata` not recognized, ignored')
            maskdata = None
    if maskdata is None:
        if 'Mask' in illbackplanes:
            mask = ill0[illbackplanes.index('Mask')].astype(bool)
        else:
            mask = zeros(ill.shape[1:],dtype=bool)
    for k in ill:
        mask |= ~isfinite(k)
    ndim = im.shape
    if ndim == 2:
        mask |= ~isfinite(im)
    else:
        for i in im:
            mask |= ~isfinite(i)
    if hasattr(im, 'mask'):
        mask |= im.mask.astype(bool)

    # Bin data if needed
    if binsize is not None:
        ndim = len(im.shape)
        if ndim == 2:
            im = rebin(im, [binsize,binsize], mean=True)
        else:
            im = rebin(im, [1,binsize,binsize], mean=True)
        ill = rebin(ill, [1,binsize,binsize], mean=True)
        mask = rebin(mask, [binsize, binsize])

    # organize data
    ww = where(~mask)
    if len(ww[0])>0:
        data = np.concatenate((im[np.newaxis,...].astype('f4'), ill.astype('f4')))
        data = data[:,~mask]
        names = imnames + backplanes
        out = Table(list(data), names=names)
        for n in backplanes:
            out.rename_column(n, geo_backplanes[n])
        out.rename_column('Data0','RADF')
        return PhotometricData(out)
    else:
        if verbose:
            print('No valid data extracted.')
        return PhotometricData()
