
'''
Package contains routines for vector operations and coordinate
transformation.

Package dependences
-------------------
numpy, utilities

History
-------
8/10/2013, started by JYL @PSI
'''

import numpy as np
from jylipy.core import *

class Vector(np.ndarray):
	'''Vector object class

	Initialization
	--------------

		v = Vector(array, axis=0, type=0, coordinate=None)
		v = Vector(c1, c2, c3, axis=-1, type=0, coordinate=None)
		v = array.view(Vector)

	If initialized with one array, then the axis specified by keyword
	`axis` defines the three coordinate components of the array.  Default
	is the last axis.

	If initialized with three arrays, then they must have the same shape.

	Keyword `type` defines the coordinate type of input components (if
	string, then not type sensitive):

	Cartesian: defined by x, y, z; type in {0, 'cartesian', 'car'}
	Spherical: defined by r, phi, theta; type in {1, 'spherical', 'sph'}
	Cylindrical: defined by rho, phi, z; type in {2, 'cylindrical', 'cyl'}
	Geographic: defined by r, lon, lat; type in {3, 'geographic', 'geo'}

	Keywords
	--------
	type : str or int
	  The type of coordinate definition.  Default is Cartesian.
	coordinate : CoordinateSystem class instance
	  The coordinate system in which the vectors are defined.
	deg : bool
	  If `True`, then angles are in degrees.  Default is `False`, and
	  all angles are in radiance.

	Vector arithmatic operations
	----------------------------

	Addition (+)
	Subtraction (-)
	Scaling (*)
	Negative (-)
	Equal (==)
	Not equal (!=)
	Dot multiplication (.dot)
	Cross multiplication (.cross)
	Normal (.norm)

	Scaling: A vector can only be scaled by an array(-like).  Each
	  element in the array is a scaling factor for the corresponding
	  vector element.
	Normal: Support any orders.  E.g., order 2 norm is the length of
	  a vector in 3-D space; order 1 norm is the sum of abs(x_i) where
	  x_i is the coordinate in axis i.

	All operations support numpy broadcast.


	v1.1.0 : JYL @PSI, 2/20/2016
	  Changed the internal storage of vectors from structured array to
	  general numpy array.
	v1.1.0 : JYL @PSI, 12/14/2016
	  Bug fix in .__str__() and .reshape()
	'''

	_types = ({'code': ('cartesian', 'car'), 'colnames': 'x y z'.split()}, \
			  {'code': ('spherical', 'sph'), 'colnames': 'r theta phi'.split()}, \
			  {'code': ('cylindrical', 'cyl'), 'colnames': 'rho phi z'.split()}, \
			  {'code': ('geographic', 'geo'), 'colnames': 'r lat lon'.split()})

	def __new__(cls, *var, **kwargs):

		if len(var) not in [1,3]:
			raise TypeError('{0} takes either 1 argument or 3 arguments ({1} given)'.format(cls, len(var)))

		axis = kwargs.pop('axis', -1)
		ctype = kwargs.pop('type', 0)
		deg = kwargs.pop('deg', False)

		if len(var) == 1:
			# if already Vector, return a copy
			if isinstance(var[0], Vector):
				return var[0].copy()

			base = np.asarray(var[0])

			# intialize with an array
			if base.ndim == 1:
				base = np.asarray(base)
			elif base.shape[axis] != 3:
				raise ValueError('the length of input array along axis {0} must be 3, length {1} received'.format(axis, base.shape[axis]))
			else:
				base = np.rollaxis(base, axis)
			b1, b2, b3 = base

		# initialized with three coordinate components
		elif len(var) == 3:
			b1, b2, b3 = var
			b1 = np.asarray(b1)
			b2 = np.asarray(b2)
			b3 = np.asarray(b3)
			l1 = np.shape(b1)
			l2 = np.shape(b2)
			l3 = np.shape(b3)
			if (l1 != l2) | (l1 != l3):
				raise ValueError('three arrays must have the same shape')

		# convert to (x,y,z) if needed
		typecode = cls._choose_type(ctype)
		if typecode == 1:
			if deg:
				b2 = np.deg2rad(b2)
				b3 = np.deg2rad(b3)
			b1, b2, b3 = cls.sph2xyz(b1, b2, b3)
		elif typecode == 2:
			if deg:
				b2 = np.deg2rad(b2)
			b1, b2, b3 = cls.cyl2xyz(b1, b2, b3)
		elif typecode == 3:
			if deg:
				b2 = np.deg2rad(b2)
				b3 = np.deg2rad(b3)
			b1, b2, b3 = cls.geo2xyz(b1, b2, b3)

		# generate object
		data = np.asarray([b1,b2,b3])
		data = np.rollaxis(data, 0, data.ndim)
		obj = data.view(Vector)
		obj.coordinate = kwargs.pop('coordinate', None)

		return obj

	def __array_finalize__(self, obj):
		if obj is None: return
		self.coordinate = getattr(obj, 'coordinate', None)

	def __mul__(self, other):
		'''The multiplication operand applies scaling of vectors.  For
		dot product or cross product, use `.dot` or `.cross` instead.
		'''
		arr1 = np.rollaxis(self.view(np.ndarray),-1)
		arr2 = np.asarray(other)
		return Vector(arr1*arr2, axis=0)

	def __rmul__(self, other):
		'''To satisfy the commutative rule for scaling multiplication'''
		return self.__mul__(other)

	def __eq__(self, other):
		comp = self.view(np.ndarray)==np.asarray(other)
		if isinstance(comp, np.ndarray):
			comp = comp.all(axis=-1)
		return comp

	def __ne__(self, other):
		eq = self.__eq__(other)
		if isinstance(eq, np.ndarray):
			return ~eq
		else:
			return not eq

	def __str__(self):
		d = self.view(np.ndarray)
		d = np.rollaxis(d, -1)
		s = d.__str__().split('\n')
		if len(s) == 1:
			return s[0]
		else:
			s[0] = s[0][1:]
			s[1] = s[1][1:]
			s[2] = s[2][1:-1]
			s = '\n'.join(s)
			return s

	def __repr__(self):
		d = self.view(np.ndarray)
		d = np.rollaxis(d, -1)
		return d.__repr__().replace('array','Vector')

	def __len__(self):
		if self.shape == ():
			raise TypeError("'Vector' object with a scalar value has no len()")
		else:
			return super(Vector, self).__len__()

	@property
	def ndim(self):
		return self.view(np.ndarray).ndim-1

	@property
	def shape(self):
		return self.view(np.ndarray).shape[:-1]

	@property
	def x(self):
		'''Cartesian x in 1-D array'''
		return np.rollaxis(self.view(np.ndarray),-1,0)[0]

	@property
	def y(self):
		'''Cartesian y in 1-D array'''
		return np.rollaxis(self.view(np.ndarray),-1,0)[1]

	@property
	def z(self):
		'''Cartesian z in 1-D array'''
		return np.rollaxis(self.view(np.ndarray),-1,0)[2]

	@property
	def xyz(self):
		return np.rollaxis(self.view(np.ndarray),-1,0)

	@property
	def r(self):
		'''Spherical r in 1-D array'''
		return self.norm()

	@property
	def theta(self):
		'''Spherical theta (radiance) in 1-D array
		0 <= theta <= pi'''
		return np.arctan2(self.rho, self.z)

	@property
	def phi(self):
		'''Spherical phi (radiance) in 1-D array
		0 <= phi < 2 pi'''
		return np.arctan2(self.y, self.x) % (2*np.pi)

	@property
	def sph(self):
		'''Spherical (r, phi, theta) in 3xN array'''
		return np.asarray([self.r, self.phi, self.theta])

	@property
	def lat(self):
		'''Latitude (radiance) in 1-D array
		-pi/2 <= lat <= pi/2'''
		return np.arctan2(self.z, self.rho)

	@property
	def lon(self):
		'''longitude (radiance) in 1-D array
		0 <= lon < 2 pi'''
		return self.phi

	@property
	def geo(self):
		'''Geographic (r, lon, lat) in 3xN array'''
		return np.asarray([self.r, self.lon, self.lat])

	@property
	def rho(self):
		'''Cylindrical rho in 1-D array'''
		return np.sqrt(self.x*self.x+self.y*self.y)

	@property
	def cyl(self):
		'''Cylindrical (rho, phi, z) in 3xN array'''
		return np.asarray([self.rho, self.phi, self.z])

	def norm(self, order=2):
		'''Compute the normal of vector(s)'''
		import numbers
		if not isinstance(order, numbers.Integral):
			raise TypeError('`order` must be an integer type.')
		if order < 1:
			raise ValueError('`order` must be a positive integer.')
		return (np.abs(self.x)**order+np.abs(self.y)**order+np.abs(self.z)**order)**(1./order)

	def dot(self, other):
		'''dot product with another vector

		If `other` is not a Vector type, then it will be converted to a
		Vector type first.  The multiplication follows numpy array
		broadcast rules.
		'''
		if not isinstance(other, Vector):
			other = Vector(other)
		return (self.view(np.ndarray)*other.view(np.ndarray)).sum(axis=-1)

	def cross(self, other):
		'''cross product with other vector(s)

		If `other` is not a Vector type, then it will be converted to a
		Vector type first.  The cross product follows numpy array
		broadcast rules'''
		if not isinstance(other, Vector):
			other = Vector(other)
		x = self.y*other.z-self.z*other.y
		y = self.z*other.x-self.x*other.z
		z = self.x*other.y-self.y*other.x
		return Vector(x, y, z)

	def reshape(self, *var):
		if isinstance(var[0],tuple):
			var = (3,)+var[0]
		else:
			var = (3,)+var
		v = np.rollaxis(self.view(np.ndarray),-1)
		v = v.reshape(*var)
		return Vector(v, axis=0)

	def flatten(self):
		return self.reshape(-1)

	def vsep(self, v2, axis=-1, type=0, deg=False, directional=False):
		'''Angular separation to another vector.  Calculation follows
		numpy array broadcast rules
		'''
		if not isinstance(v2, Vector):
			v2 = Vector(v2, axis=axis, type=type)
		angle = np.arccos(self.dot(v2)/(self.norm()*v2.norm()))
		if directional:
			pi2 = np.pi*2
			zcomp = self.x*v2.y - self.y*v2.x
			wz = zcomp < 0
			if angle[wz].size > 0:
				if hasattr(angle, '__iter__'):
					angle[wz] = pi2 - angle[wz]
				else:
					angle = pi2 - angle
			wz = zcomp == 0
			if angle[wz].size > 0:
				xcomp = self.y*v2.z - self.z*v2.y
				wx = xcomp < 0
				if angle[wz&wx].size > 0:
					if hasattr(angle, '__iter__'):
						angle[wz&wx] = pi2 - angle[wx&wz]
					else:
						angle = pi2 - angle
				wx = xcomp == 0
				if angle[wx].size > 0:
					ycomp = self.z*v2.x - self.x*v2.z
					wy = ycomp < 0
					if angle[wz&wx&wy].size > 0:
						if hasattr(angle, '__iter__'):
							angle[wz&wx&wy] = pi2 - angle[wz&wx&wy]
						else:
							angle = pi2 - angle
		if deg:
			angle = np.rad2deg(angle)
		return angle

	def rot(self, phi, axis=2, deg=True):
		'''Rotate vector(s) along an axis

		`phi` must be a scalar.  Broadcast is not supported'''
		if deg:
			phi = np.deg2rad(phi)
		return VectRot(rotm(phi, axis=axis).T)*self

	def eular(self, phi, theta, psi, deg=True):
		'''Rotate vector(s) by three Eular angles

		Angles `phi`, `theta`, and `psi` must be scalars.  Broadcast is
		not supported'''
		if deg:
			phi = np.deg2rad(phi)
			theta = np.deg2rad(theta)
			psi = np.deg2rad(psi)
		return VectRot(eularm(phi, theta, psi).T)*self

	def astable(self, type=0):
		typecode = self._choose_type(type)
		names = self._types[typecode]['colnames']
		if typecode == 0:
			c1, c2, c3 = self.xyz
		elif typecode == 1:
			c1, c2, c3 = self.sph
		elif typecode == 2:
			c1, c2, c3 = self.cyl
		elif typecode == 3:
			c1, c2, c3 = self.geo
		return Table((c1.flatten(),c2.flatten(),c3.flatten()), names=names)

	@staticmethod
	def _choose_type(ctype):
		if ctype in [0,1,2,3]:
			return ctype
		if isinstance(ctype, str):
			ctype = ctype.lower()
		idx = [ctype in x['code'] for x in Vector._types]
		if True in idx:
			return idx.index(True)
		else:
			raise ValueError('Undefined coordinate type %s' % ctype)

	@staticmethod
	def sph2xyz(r, phi, theta):
		z = r*np.cos(theta)
		rho = r*np.sin(theta)
		x = rho*np.cos(phi)
		y = rho*np.sin(phi)
		return x, y, z

	@staticmethod
	def cyl2xyz(rho, phi, z):
		x = rho*np.cos(phi)
		y = rho*np.sin(phi)
		return x, y, z

	@staticmethod
	def geo2xyz(r, lon, lat):
		z = r*np.sin(lat)
		rho = r*np.cos(lat)
		x = rho*np.cos(lon)
		y = rho*np.sin(lon)
		return x, y, z


def empty(shape):
	'''Generate empty vector

	v1.0.0 : 12/14/2016, JYL @PSI
	'''
	if not hasattr(shape,'__iter__'):
		shape = [shape]
	return Vector(np.empty(list(shape)+[3]))


def empty_like(a):
	'''Generate empty vector like the shape of input
	v1.0.0 : 12/14/2016, JYL @PSI
	'''
	return empty(a.shape)


def ones(shape):
	'''Generate unit vector
	v1.0.0 : 12/14/2016, JYL @PSI
	'''
	v = zeros(shape)
	v.view(np.ndarray)[...,-1] = 1.
	return v


def ones_like(a):
	'''Generate unit vector like the shape of input
	v1.0.0 : 12/14/2016, JYL @PSI
	'''
	return ones(a.shape)


def zeros(shape):
	'''Generate zero vector
	v1.0.0 : 12/14/2016, JYL @PSI
	'''
	v = empty(shape)
	v[:] = 0
	return v


def zeros_like(a):
	'''Generate zero vector like the shape of input
	v1.0.0 : 12/14/2016, JYL @PSI
	'''
	return zeros(a.shape)


def stack(vects, axis=0):
	'''Similar to numpy.stack, but for Vector'''
	shps = [x.shape for x in vects]
	for sh in shps[1:]:
		if sh != shps[0]:
			raise ValueError('all input vectors must have the same shape')
	d = [x.view(np.ndarray) for x in vects]
	s = np.stack(d, axis=axis)
	return Vector(s)


def concatenate(vects, axis=0):
	'''Similar to numpy.concatenate, but for Vectors'''
	vects = [Vector(x) for x in vects]
	ndims = np.array([x.ndim for x in vects])
	if ndims.max() != ndims.min():
		raise ValueError('all the input arrays must have same number of dimensions')
	shps = [np.delete(np.array(Vector(x).shape),axis) for x in vects]
	for sh in shps[1:]:
		if sh != shps[0]:
			raise ValueError('all the input vector dimensions except for the concatenation axis must match exactly')
	d = [x.view(np.ndarray) for x in vects]
	vout = np.concatenate(d, axis=axis)
	return Vector(vout)


def delete(vect, obj, axis=None):
	'''Similar to numpy.delete, but for Vector'''
	if axis is not None:
		if axis >= vect.ndim:
			raise ValueError('axis {0} is out of bound [{1},{2})'.format(axis, 0, vect.ndim))
	if axis is None:
		vect = vect.flatten()
		axis = 0
	d = vect.view(np.ndarray)
	vout = np.delete(d, obj, axis=axis)
	return Vector(vout)


def insert(vect, obj, values, axis=None):
	'''Similar to numpy.insert, but for Vector'''
	if axis is not None:
		if axis >= vect.ndim:
			raise ValueError('axis {0} is out of bound [{1},{2})'.format(axis, 0, vect.ndim))
	if axis is None:
		vect = vect.flatten()
		axis = 0
	d = vect.view(np.ndarray)
	v = Vector(values).view(np.ndarray)
	vout = np.insert(d, obj, v, axis=axis)
	return Vector(vout)


def append(vect, values, axis=None):
	'''Similar to numpy.append, but for Vector'''
	if axis is None:
		vect = vect.flatten()
		axis = 0
	return concatenate((vect, values), axis=axis)


def squeeze(vect, axis=None):
	'''Similar to numpy.squeeze, but for Vector'''
	if axis is not None:
		if axis >= vect.ndim:
			raise ValueError('axis {0} is out of bound [{1},{2})'.format(axis, 0, vect.ndim))
	return Vector(np.squeeze(vect.view(np.ndarray), axis=axis))


class VectRot(np.ndarray):
	'''Vector rotation class in 3-D space, including rotation and scaling

	Initialization
	--------------

	By another VectRot class instance
	By rotation matricies, and scaling factors

	To apply a rotation to a Vector:
		r = VectRot(m)
		v1 = r.inner(v)
	or equivalently
		v1 = r(v)
	or equivalently
		v1 = r*v
	Both r and v can be array of rotations and vectors, respectively.

	Operands
	--------

	Multiplication (*, .inner, or function call by class instance):
	  Scale VectRot, rotate a Vector, or combine rotation with another
	  VectRot.  Note that __rmul__ is defined as the same as __mul__ but
	  with the transposed rotation.
	Power (**): Only defined with the power index is a scalor integer type.
	All other operands are blocked (+, -, /)

	All operations supports broadcast.


	v1.0.0 : JYL @PSI, 2/21/2016
	'''

	def __new__(cls, *var, **kwargs):

		if len(var) != 1:
			raise TypeError('{0} takes 1 argument ({1} given)'.format(cls, len(var)))

		scale = kwargs.pop('scale', None)

		# if a VectRot instance, return a copy
		if isinstance(var[0], VectRot):
			return var[0].copy()

		# initialize with numpy array
		data = np.asarray(var[0])
		if data.shape[-2:] != (3,3):
			raise ValueError('the shape of the last two dimentions of input array must be 3, length {0} received'.format(data.shape[-2:]))

		# attach scaling factors
		if scale is not None:
			scale = np.asarray(scale)
			s = np.stack([scale]*3,axis=-1)
			s = np.stack([s]*3,axis=-1)
			data = data*s

		# generate object
		return data.view(VectRot)

	def __str__(self):
		d = self.view(np.ndarray)
		d = np.rollaxis(d, -1)
		d = np.rollaxis(d, -1)
		s = d.__str__()
		s = [x[1:] for x in s.split('\n')]
		s[2] = s[2][:-1]
		s = '\n'.join(s)
		return s

	def __repr__(self):
		d = self.view(np.ndarray)
		d = np.rollaxis(d, -1)
		d = np.rollaxis(d, -1)
		return d.__repr__().replace('array', 'VectRot')

	def __len__(self):
		if self.shape == ():
			raise TypeError("'VectRot' object with a scalar value has no len()")
		else:
			return super(VectRot, self).__len__()

	def __add__(self, v):
		raise TypeError('add is not defined for VectRot type')

	def __radd__(self, v):
		self.__add__(v)

	def __sub__(self, v):
		raise TypeError('sub is not defined for VectRot type')

	def __rsub__(self, v):
		self.__sub__(v)

	def __mul__(self, v):
		return self.inner(v)

	def __rmul__(self, v):
		return self.T.inner(v)

	def __div__(self, v):
		raise TypeError('div is not defined for VectRot type')

	def __rdiv__(self, v):
		self.__div__(v)

	def __pow__(self, v):
		if hasattr(v, '__iter__'):
			raise TypeError('power with non-scaler not defined for VectRot type')
		import numbers
		if not isinstance(v, numbers.Integral):
			raise TypeError('power can only be performed with integer types')
		out = self.copy()
		while v>1:
			out = out.inner(out)
			v -= 1
		return out

	@property
	def ndim(self):
		return self.view(np.ndarray).ndim-2

	@property
	def shape(self):
		return self.view(np.ndarray).shape[:-2]

	@property
	def det(self):
		from numpy.linalg import det
		return det(self)

	@property
	def T(self):
		return VectRot(np.rollaxis(self.view(np.ndarray),-1,-2))

	def inner(self, v):
		'''Inner product of the VectRot type with another variable.

		Type `v`        Operation           Return
		-----------     ----------------    -------
		numpy array     scaling             VectRot
		Vector          Vector rotation     Vector
		VectRot         Combined VectRot    VectRot

		Support numpy broadcast rules.'''

		if isinstance(v, Vector):
			d = np.stack([v.view(np.ndarray)]*3,axis=-2)
			y = (self.view(np.ndarray)*d).sum(axis=-1)
			return Vector(y)
		elif isinstance(v, VectRot):
			d = np.rollaxis(v.view(np.ndarray),-1,-2)
			d = np.stack([d]*3, axis=-3)
			s = np.stack([self.view(np.ndarray)]*3, axis=-2)
			y = (s*d).sum(axis=-1)
			return VectRot(y)
		else:
			v = np.asarray(v)
			d = np.stack([v]*3, axis=-1)
			d = np.stack([d]*3, axis=-1)
			y = VectRot(self.view(np.ndarray)*d)
			return y

	def __call__(self, v):
		return self.inner(v)


class CoordinateError(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)


class CoordinateSystem(np.ndarray):

	def __new__(cls, *args, **kwargs):
		if len(args) == 1:
			if isinstance(args[0], CoordinateSystem):
				return args[0].copy()
		self = Vector(*args, **kwargs).asarray().view(cls)
		self.coordinate = kwargs.pop('coordinate', None)
		self.name = kwargs.pop('name', '')
		if self.coordinate is not None:
			self.level = self.coordinate.level+1
		else:
			self.level = 0
		return self

	def __array_finalize__(self, obj):
		if obj is None: return
		self.coordinate = getattr(obj, 'coordinate', None)
		self.name = getattr(obj, 'name', None)
		self.level = getattr(obj, 'level', 0)

	def __getitem__(self, i):
		return Vector(self.view(np.ndarray)[i])

	def __setitem__(self, i, v):
		self.view(np.ndarray)[i] = v

	def __repr__(self):
		return self.view(np.ndarray).__repr__().replace('array','CoordinateSystem')

	def __len__(self):
		return 1

	def __add__(self, other):
		self._raise_operator_error('+', other)

	def __radd__(self, other):
		self.__add__(other)

	def __sub__(self, other):
		self._raise_operator_error('-', other)

	def __rsub__(self, other):
		self.__sub__(other)

	def __mul__(self, other):
		'''Need to confirm the order of matrix multiplication'''
		m = np.asarray(self)
		if isinstance(other, CoordinateSystem):
			return CoordinateSystem(self.view(np.ndarray)*other.view(np.ndarray))
		elif isinstance(other, Vector):
			v = np.rollaxis(other.asarray(),-1,-2)  # shape = [..., 3, :]
		else:
			v = np.asarray(other)
		return Vector(m.dot(v),axis=-2)

	def __rmul__(self, other):
		other = np.asarray(other)
		if other.shape == (3,3):
			other = CoordinateSystem(other)
			return other*self
		self._raise_operator_error('*', other)

	def __div__(self, other):
		self._raise_operator_error('/', other)

	def __rdiv__(self, other):
		self._raise_operator_error('/', other)

	def __eq__(self, other):
		if (self.view(np.ndarray) == np.asarray(other)).all():
			return True
		return False

	def _raise_operator_error(self, op, other):
		raise TypeError("unsupported operand type(s) for "+op+": {0} and {1}".format(type(self),type(other)))

	def transform(self, coord):
		'''TO BE FINISHED'''
		out = self.copy()
		while True:
			outcoord = out.coordinate
			if outcoord == coord:
				return out
			if outcoord is None:
				raise CoordinateError('coordinate not found')
			out * out.coordinate


def rotm(phi, axis=2):
	'''
 Calculate a numpy matrix to transform a vector to a new coordinate
 that rotates from the old coordinate along a coordinate axis

 Parameters
 ----------
 phi : floating point
   Angle of rotation [deg]
 axis: integer 0, 1, or 2
   Axis of rotation.   0 for x-axis, 1 for y-axis, 2 for z-axis

 Returns
 -------
 numpy matrix, the transformation matrix from old coordinate to new
 coordinate.

 v1.0.0 : JYL @PSI, August 4, 2013.
 v1.0.1 : JYL @PSI, 2/19/2016
   Use np.deg2rad to perform deg to rad conversion
 v1.0.2 : JYL @PSI, 2/21/2016
   Change the default unit of angle `phi` to radiance to simplify future
     revision with astropy.units
	'''

	cosphi = np.cos(phi)
	sinphi = np.sin(phi)
	m = np.matrix([[cosphi,sinphi,0],[-sinphi,cosphi,0],[0,0,1]])
	if axis == 0:
		m = np.roll(np.roll(m,1,axis=0),1,axis=1)
	if axis == 1:
		m = np.roll(np.roll(m,-1,axis=0),-1,axis=1)
	return m


def rotvm(v, phi):
	'''Return the matrix to rotate a vector around `v` by `phi`'''
	pass


def eularm(phi, theta, psi):
	'''
 Calculate the coordinate transformation matrix to the new coordinate
 rotated with respect to the old coordinate with three Eular angles

 Parameters
 ----------
 phi, theta, psi : floating point
   Three Eular angles [deg]

 Returns
 -------
 Returns a numpy matrix to transform from old coordinate to the
 new (rotated) coordinate

 v1.0.0 : JYL @PSI, August 4, 2013
	'''
	return rotm(psi,axis=2).dot(rotm(theta,axis=0)).dot(rotm(phi,axis=2))


def rotv(v, phi, axis=2):
	'''
 Rotate vector(s) along a specified axis

 Parameters
 ----------
 v  : array-like floating point
   Input 3-vector or column 3-vectors to be rotated
 phi : array-like floating point
   Angle(s) to rotate [deg]
 axis: integer 0, 1, or 2
   Axis to rotate.  0 for x-axis, 1 for y-axis, 2 for z-axis

 Returns
 -------
 Returns the rotated vector

 Note
 ----
 If 'phi' contains more than 1 element, then all input vector(s) will
 be rotated by all the angles in 'phi'.  The returned array will have
 one more dimension than the input 'v'.  The expansion of array
 happens either along the row direction if input 'v' only contains a
 single vector, or along the layer direction if input 'v' contains
 many column vectors.

 v1.0.0 : JYL @PSI, August 4, 2013
	'''

	if not hasattr(phi, '__iter__'):
		return rotm(phi,axis).T.dot(v)

	nphi = np.size(phi)
	if np.size(v) == 3:
		v1 = np.empty((3,nphi))
		for i in range(nphi):
			v1[:,i] = rotv(v,phi[i],axis)
	else:
		v1 = np.expand_dims(v,axis=0)
		v1 = np.pad(v1, ((0,nphi-1),(0,0),(0,0)), mode='constant')
		for i in range(nphi):
			v1[i,:,:] = rotv(v,phi[i],axis)

	return v1


def eularv(v, phi, theta, psi):
	'''
 Rotate vector(s) by three Eular angles

 Parameters
 ----------
 v  : array-like floating point
   Input 3-vector(s)
 phi, theta, psi : floating point
   Three Eular angles [deg]

 Returns
 -------
 Returns the rotated vector(s)

 v1.0.0 : JYL @PSI, August 4, 2013
	'''

	return eularm(phi,theta,psi).T.dot(v)


def twovec(axdef, indexa, plndef, indexp):
	'''
 Calculate the transformation matrix to a new coordinate defined by
 two input vectors.

 Parameters
 ----------
 axdef: array-like floating point
   A 3-vector of (x,y,z), or 2-vector of (RA, Dec) that defines one of
   the principal axes of the new coordinate frame.
 indexa: integer 0, 1, or 2
   Specify which of the three coordinate axes contains axdef.  0 for
   x-axis, 1 for y-axis, and 2 for z-axis
 plndef: array-like floating point
   A 3-vector of (x,y,z), or 2-vector of (RA, Dec) that defines (with
   axdef) a principal plane of the new coordinate frame.
 indexp: integer 0, 1, or 2
   Specify the second axis of the principal frame determined by axdef
   and plndef

 Returns
 -------
 Returns the transformation matrix that convert a vector from the old
 coordinate to the coordinate frame defined by the input vectors.

 Notes
 -----
 This routine is directly translated form SPICE lib routine twovec.f
 (cf. SPICE manual
 http://www.nis.lanl.gov/~esm/idl/spice-dlm/spice-t.html#TWOVEC)

 The indexing of array elements are different in FORTRAN (that SPICE
 is originally based) from Python.  Here we used 0-based index.

 Note that the original twovec.f in SPICE toolkit returns matrix that
 converts a vector in the new frame to the original frame, opposite to
 what is implemented here.

 v1.0.0 : JYL @PSI August 6, 2013
 v1.0.1 : JYL @PSI December 22, 2013
   Minor revision
   Accept 2-vectors in (RA, Dec) as input
	'''

	axdef, plndef = np.asarray(axdef).flatten().copy(), np.asarray(plndef).flatten().copy()
	if axdef.shape[0] == 2:
		axdef = sph2xyz(axdef)
	if plndef.shape[0] == 2:
		plndef = sph2xyz(plndef)

	from numpy.linalg import norm
	if norm(np.cross(axdef,plndef)) == 0:
		raise RuntimeError('The input vectors AXDEF and PLNDEF are linearly correlated and can\'t define a coordinate frame.')

	M = np.eye(3)
	i1, i2, i3 = indexa % 3, (indexa+1) % 3, (indexa+2) % 3

	M[i1,:] = axdef.flatten()/norm(axdef)
	if indexp == i2:
		xv = np.cross(axdef,plndef)
		M[i3,:] = xv/norm(xv)
		xv = np.cross(xv,axdef)
		M[i2,:] = xv/norm(xv)
	else:
		xv = np.cross(plndef,axdef)
		M[i2,:] = xv/norm(xv)
		xv = np.cross(axdef,xv)
		M[i3,:] = xv/norm(xv)

	return M


def vecsep(v1, v2, directional=False, axis=0, row=False, deg=True):
	'''
 Calculate directional separation (angle) from vector v1 to vector v2

 Parameters
 ----------
 v1, v2  : 1-D or 2D array-like, floating point
   3-vector(s) (x,y,z) or 2-vector(s) of (RA, Dec) or (lon, lat).  Can either
   contain the same number of vectors, or one single vector
 directional: Bool, optional
   Specify whether the result angle is directional or undirectional
 axis : int
   The axis that defines vectors
 row : bool, optional
   If `True`, then the input vectors are considered row-vectors.  This
   keyword overrides `axis`.
 deg : bool, optional
   If `True`, return angles in degrees.  Otherwise in radiance

 Returns
 -------
 numpy array of float
 Returns the angle(s) between v1 and v2, in [0, pi).  If `directional`
 is `True`, then the angle is measured from v1 to v2 in [0, pi)
 following right-hand rule with z-axis.  See Notes.

 Notes
 -----
 If v1 and v2 are coplanar with z-axis, then x-axis will be used as
 the reference pole for directional calculation.  If v1 and v2 are in
 x-z plane, then y-axis will be the reference pole.

 v1.0.0 : JYL @PSI, August 9, 2013
 v1.0.1 : JYL @PSI, December 22, 2013
   Remove debug keyword `loop`
   Add keyword `row` to accept row vectors
   Add the step to check the validity of input vectors
 v1.0.2 : JYL @PSI, October 8, 2014
   Returns a vector if takes only one pair of input vectors
 v1.1.0 : JYL @PSI, January 23, 2015
   Add keyword `axis`
   Prepare to deprecate keyword `row`
   Optimized flow
   Increased the flexibility in terms of accepted data type
 v2.0 : JYL @PSI, 2/20/2016
   Use Vector class `.vsep` to calculate vector separation
	'''
	if row:
		axis = 1
	if axis != 0:
		v1 = np.rollaxis(v1, axis)
		v2 = np.rollaxis(v2, axis)
	if len(v1) == 2:
		b2, b3 = v1
		b1 = np.ones_like(b2)
		v1 = Vector(b1, np.deg2rad(b2), np.deg2rad(b3), type='geo')
	elif len(v1) == 3:
		v1 = Vector(v1, axis=0)
	else:
		raise ValueError('first vector is invalid')
	if len(v2) == 2:
		b2, b3 = v2
		b1 = np.ones_like(b2)
		v2 = Vector(b1, np.deg2rad(b2), np.deg2rad(b3), type='geo')
	elif len(v2) == 3:
		v2 = Vector(v2, axis=0)
	else:
		raise ValueError('second vector is invalid')
	return v1.vsep(v2, directional=directional, deg=deg)


def xyz2sph(v, axis=0, row=False, zenith=False):
	'''
 Convert input vector(s) from Cartesian coordinates to spherical
 coordinate

 Parameters
 ----------
 v  : 1-D or 2-D array-like number
   3-vector or column 3-vectors in a 2-D array
 row : bool, optional
   If `True`, then the input vectors are considered row-vectors
 zenith : Bool, optional
   If set True, then the Dec will be replaced by zenith angle where
   it starts from 0 at +z-axis increases to 180 deg in the -z-axis.

 Returns
 -------
 Returns column vectors of the r, RA, and Dec for input vector(s)
 RA and Dec are in[deg]

 v1.0.0 : JYL @PSI, August 9, 2013
 v1.0.1 : JYL @PSI, December 22, 2013
   Corrected a bug to process the shapes of input array correctly
 v1.0.2 : JYL @PSI, 2/19, 2016
   Add keyword `axis`
 v2.0.0 : JYL @PSI, 2/20/2016
   Use Vector object to perform conversion
	'''

	if row:
		axis = 1
	v = Vector(v,axis=axis)
	if zenith:
		vout = v.sph
	else:
		vout = v.geo
	vout[1] = np.rad2deg(vout[1])
	vout[2] = np.rad2deg(vout[2])
	return vout


	#v = np.asarray(v).copy().astype(float)
	#vdim = v.ndim
	#if vdim not in [1,2]:
	#	raise RuntimeError('Input vectors must be 1-D or 2-D array-like')
	#if row:
	#	axis = 1
	#if axis != 0:
	#	v = colvec(v, axis)
	#if v.shape[0] != 3:
	#	raise RuntimeError('Input vectors must be 3-vector(s)')
	#if vdim == 1:
	#	v = v.reshape(3,-1)
	## r
	#r = np.sqrt(v[0,:]*v[0,:]+v[1,:]*v[1,:]+v[2,:]*v[2,:])
	## ra
	#nv = v.shape[1]
	#ra = np.empty(nv)
	#ra[v[0,:]==0] = 90.
	#ra[(v[0,:]==0) & (v[1,:]<0)] = 270.
	#ra[(v[0,:]==0) & (v[1,:]==0)] = 0.
	#nz = v[0,:].nonzero()
	#ra[nz] = np.arctan2(v[1,nz],v[0,nz]) * 180/np.pi
	#ra = (ra+360) % 360  # normalize to [0,360)
	## dec
	#dec = np.empty(nv)
	#dec[(v[0,:]==0) & (v[1,:]==0)] = 90.
	#dec[(v[0,:]==0) & (v[1,:]==0) & (v[2,:]<0)] = -90.
	#dec[(v[0,:]==0) & (v[1,:]==0) & (v[2,:]==0)] = 0.
	#nz = (v[0,:]!=0) | (v[1,:]!=0)
	#dec[nz] = np.arctan2(v[2,nz],np.sqrt(v[0,nz]*v[0,nz]+v[1,nz]*v[1,nz])) * #180/np.pi
	#if zenith:
	#	dec = 90-dec
	#sph = np.concatenate((r.reshape(1,-1),ra.reshape(1,-1), dec.reshape(1,-1)))
	#if vdim == 1:
	#	sph = sph.flatten()
	#return sph


def sph2xyz(v, axis=0, row=False, zenith=False):
	'''
 Convert input vector(s) from spherical coordinate to Cartesian
 coordinate

 Parameters
 ----------
 v  : 1-D or 2D array-like numbers
   Vector or column vectors in a 2-D array.  It can be either a 3xN
   array with each column containing (r, RA, Dec), or a 2xN array with
   each column containing (RA, Dec).  RA and Dec are in [deg]
 row : bool, optional
   If `True`, then the input vectors are considered row-vectors
 zenith : Bool, optional
   If set True, then the Dec will be replaced by zenith angle where it
   starts from 0 at +z-axis increases to 180 deg in the -z-axis.

 Returns
 -------
 Returns column vectors of the (x,y,z) for input vector(s).  If only
 (RA, Dec) available in the input vector, then they are assumed to
 have unity length.

 v1.0.0 : JYL @PSI, August 9, 2013
 v1.0.1 : JYL @PSI, December 22, 2013
   Improved the check of robustness for input array
 v1.0.2 : JYL @PSI, 2/19, 2016
   Add keyword `axis`
 v2.0.0 : JYL @PSI, 2/20/2016
   Use Vector object to perform calculation
	'''

	if row:
		axis = 1
	if axis != 0:
		v = np.rollaxis(v, axis)
	b2, b3 = v
	b2 = np.deg2rad(b2)
	b3 = np.deg2rad(b3)
	b1 = np.ones_like(b2)
	if zenith:
		ctype = 'sph'
	else:
		ctype = 'geo'
	v = Vector(b1, b2, b3, type=ctype)
	return v.xyz

	#v = np.asarray(v).copy().astype(float)
	#vdim = v.ndim
	#if vdim not in [1,2]:
	#	raise RuntimeError('Input vectors must be 1-D or 2-D array-like')
	#if row:
	#	axis = 1
	#if axis != 0:
	#	v = np.rollaxis(v, axis)
	#if v.shape[0] not in [2,3]:
	#	raise RuntimeError('Input vectors must be 2- or 3-vector(s)')
	#if v.shape[0] == 2:
	#	v = v.reshape(2,-1)
	#	v = np.pad(v,((1,0),(0,0)),mode='constant',constant_values=(1.,))
	#elif v.shape[0] == 3:
	#	v = v.reshape(3,-1)
	#if zenith:
	#	v[2,:] = 90.-v[2,:]
	#v[1:,:] *= np.pi/180
	#cosra, sinra, cosdec, sindec = np.cos(v[1,:]), np.sin(v[1,:]), np.cos(v[2,:]), np.sin(v[2,:])
	#x, y, z = v[0,:]*cosdec*cosra, v[0,:]*cosdec*sinra, v[0,:]*sindec
	#xyz = np.concatenate((x.reshape(1,-1),y.reshape(1,-1),z.reshape(1,-1)))
	#if vdim == 1:
	#	xyz = xyz.flatten()
	#return xyz


def cel2ecl(v, axis=0, row=False, obliquity=23.4393):
	'''
 Convert celestial (equatorial) coordinates to ecliptic coordinates

 Parameters
 ----------
 v  : array-like numbers
   Vector or column vectors.  See sph2xyz.
 row : bool, optional
   If `True`, then the input vectors are considered row-vectors
 obliquity : float, optional
   Obliquity of the Earth.

 Returns
 -------
 Returns column vectors of the same form as input vector(s), but in
 ecliptic coordinate frame.

 v1.0.0 : JYL @PSI, August 11, 2013
 v1.0.1 : JYL @PSI, December 22, 2013
   Significantly simplified the program
 v1.0.2 : JYL @PSI, April 5, 2014
   Found a bug in calling sph2xyz
 v1.0.3 : JYL @PSI, 2/19, 2016
   Add keyword `axis`
	'''

	mx = np.array(\
		[[1, 0, 0], \
	     [0, np.cos(obliquity*np.pi/180), np.sin(obliquity*np.pi/180)], \
	     [0, -np.sin(obliquity*np.pi/180), np.cos(obliquity*np.pi/180)]])

	out = xyz2sph(mx.dot(sph2xyz(v, axis=axis, row=row)))
	#if np.asarray(v).shape[axis] == 2:
	#	out = out[1:]

	return out


def ecl2cel(v, axis=0, row=False, obliquity=23.4393):
	'''
 Convert ecliptic coordinates to celestial (equatorial) coordinates

 Parameters
 ----------
 v  : array-like numbers
   Vector or column vectors.  See sph2xyz.
 row : bool, optional
   If `True`, then the input vectors are considered row-vectors
 obliquity : floating point Obliquity of the Earth.  Default is
   23.4393 deg.

 Returns
 -------
 Returns column vectors of the same form as input vector(s), but in
 ecliptic coordinate frame.

 v1.0.0 : JYL @PSI, August 11, 2013
 v1.0.1 : JYL @PSI, December 22, 2013
   Significantly simplified the program
 v1.0.2 : JYL @PSI, 2/19/2016
   Add keyword `axis`

	'''

	mx = np.array(\
		[[1, 0, 0], \
	     [0, np.cos(obliquity*np.pi/180), -np.sin(obliquity*np.pi/180)], \
	     [0, np.sin(obliquity*np.pi/180), np.cos(obliquity*np.pi/180)]])

	out = xyz2sph(mx.dot(sph2xyz(v, axis=axis, row=row)))
	#if np.asarray(v).shape[axis] == 2:
	#	out = out[1:]

	return out


def paplane(los, pa):
	'''
 Calculate the plane that contains the specified position angle for
 particular line of sight.

 Parameters
 ----------
 los : array-like, floating point
   The (RA, Dec) [deg], or (x, y, z) of line of sight
 pa  : array-like, floating point
   The position angles to be calculated [deg east of north]

 Returns
 -------
 Array-like, floating point
 Returns the normal direction of the plane(s) containing the position
 angle and line of sight in (RA, Dec) [deg]

 v1.0.0 : JYL @PSI, November 2013
 v1.0.1 : JYL @PSI, December 22, 2013
   Corrected a bug in calculating the `pav`, position angle vector
   Optimized by replacing iterative call with loop
	'''

	los1 = np.asarray(los).flatten()
	ns = los1.shape[0]
	if ns == 2:
		los1 = sph2xyz(los1)
	m = twovec(-los1, 2, [0.,0,1], 0)

	if not hasattr(pa, '__iter__'):
		pas = [pa]
	else:
		pas = pa

	norms = []
	for p in pas:
		pav = m.T.dot(sph2xyz([p, 0.]))
		norm = np.cross(los1, pav)
		if ns == 2:
			norm =  xyz2sph(norm)[1:]
		norms.append(norm)

	if not hasattr(pa, '__iter__'):
		return norms[0]

	return np.array(norms).T


def vecpa(los, vec, row=False):
	'''
 Calculate the projected position angle of a vector for a given
 line-of-sight vector

 Parameters
 ----------
 los : array-like, floating point
   The (RA, Dec) [deg], or (x,y,z) of line of sight
 vec : 1-D or 2-D array-like, floating point
   The (RA, Dec) [deg], or (x,y,z) of vector(s)
 row : bool, optional
   If `True`, then the input vectors are considered row-vectors

 Returns
 -------
 array-like, floating point
 Returns the vector(s) containing the position angle and aspect angle.
 Position angle is measured from N to E.  Aspect angle is measured
 from sky plane towards the input vector(s), with positive value
 pointing towards observer and negative value pointing away.

 v1.0.0 : JYL @PSI, November 2013
 v1.0.1 : JYL @PSI, December 22, 2013
   Add parameter checking
   Optimized program by replacing iterative call with loop
	'''

	vec = np.asarray(vec).copy().astype(float)
	if vec.ndim not in [1,2]:
		raise RuntimeError('Input vectors must be 1-D or 2-D array-like')
	if row:
		vec = vec.T
	if vec.shape[0] not in [2,3]:
		raise RuntimeError('Input vectors must be 2- or 3-vector(s)')
	if vec.shape[0] == 2:
		vec = sph2xyz(vec)

	los = np.asarray(los).flatten()
	if los.shape[0] == 2:
		los = sph2xyz(los)
	m = twovec(-los, 2, [0.,0.,1.], 0)

	if vec.ndim == 1:
		return xyz2sph(m.dot(vec))[1:]

	return np.asarray([xyz2sph(m.dot(v))[1:] for v in vec.T]).T


def vect2proj(vect, viewpt, pa=0.):
	'''Convert vector coordinates to its coordinates in a parallel
	projected plane defined by `viewpt`.

	The projection plane is defined as +x along horizontal direction
	towards right, +y along verticle direction towards up.  The +z
	axis completes the right-hand system towards viewer.

	Parameter
	---------
	vect : Vector
	  The vector to be projected
	viewpt : Vector
	  The vector of the view point
	pa : number, optional
	  Position angle of the z-axis in projection plane, measured in
	  radiance from up to left (ccw).

	Return
	------
	A Vector instance containing the projected coordinates of the input
	Vector.

	v1.0.0 : JYL @PSI, 2/21/2016
	'''
	# Rotate along z-axis by sub-observer longitude
	m1 = VectRot(rotm(-viewpt.lon-np.pi/2, axis=2).T)
	# Rotate along x-axis by sub-observer azimuth
	m2 = VectRot(rotm(-viewpt.theta, axis=0).T)
	# Rotate along z-axis by position angle
	m3 = VectRot(rotm(pa, axis=2).T)
	return m3*m2*m1*vect


def proj2body(vect, viewpt, pa=0.):
	'''Convert the vector coordinates in a parallel projected plane
	defined by `viewpt` to their body-fixed coordiantes.

	Parameters
	----------
	vect : Vector
	  The vector with coordinates in a parallel projected plane.  Note
	  that the z-coordinates in this vector is ignored.
	viewpt : Vector
	  The vector of the view point
	pa : number, optional
	  Position angle of the z-axis in projection plane, measured in
	  degrees from up to left (ccw).

	Return
	------
	A Vector instance contaning the coordinates of the input vector
	in the body-fixed frame.  NOTE: Only the x- and y-coordinates of
	the returned vector is valid.  The valid z-coordinates can only
	be calculated with the actual shape of the body.

	This program is essentially the inverse of `vect2proj`.

	v1.0.0 : JYL @PSI, 2/21/2016
	'''
	m1 = VectRot(rotm(-viewpt.lon-np.pi/2, axis=2))
	m2 = VectRot(rotm(-viewpt.theta, axis=0))
	m3 = VectRot(rotm(pa, axis=2))
	return m1*m2*m3*vect


def sphere_vert(r=1., nlon=360, nlat=181):
	'''Generate a Vector instance containing the verticies of a sphere

	v1.0.0 : JYL @PSI, 2/21/2016
	'''
	lonstep = 360/nlon
	latstep = 179/(nlat-2)
	lat, lon = makenxy(-89, 89, nlat-2, 0, 360-lonstep, nlon)
	v = Vector(np.ones_like(lat)*r, np.deg2rad(lon), np.deg2rad(lat), type='geo')
	v = insert(v, 0, Vector([0,0,-1.]))
	v = append(v, Vector([0,0,1.])[np.newaxis])
	return v


def ellipsoid_vert(a=1., b=1., c=1., nlon=360, nlat=181):
	'''Generate a Vector insert containing the verticies of a triaxial
	ellipsoid

	v1.0.0 : JYL @PSI, 2/21/2016
	'''
	v = sphere_vert(nlon=nlon, nlat=nlat)
	v = Vector(v.x*a, v.y*b, v.z*c)
	return v


#class Sphere(object):
#	'''Spherical body'''
#	def __init__(self, r=1., nlon=360, nlat=181):
#		self.r = r
#		self.nlon = nlon
#		self.nlat = nlat
#		self.verticies = sphere_vert(r, nlon, nlat)
#
#
#class Ellipsoid(object):
#	'''Triaxial ellipsoid'''
#	def __init__(self, a=1., b=1., c=1., nlon=360, nlat=181):
#		self.a = a
#		self.b = b
#		self.c = c
#		self.nlon = nlon
#		self.nlat = nlat
#		self.verticies = ellipsoid_vert(a, b, c, nlon, nlat)
#
#	@property
#	def normal(self):
#		return Vector(2*self.verticies.x/self.a**2, 2*self.verticies.y/self.b**2, 2*self.verticies.z/self.c**2)


def xy2lonlat(r, viewpt, pxlscl, pa=0., imsz=(1024, 1024), center=None):
	'''Convert (x,y) coordinate of a CCD to body-fixed (lon,lat) for a
	sphere or an ellipsoid

	r : number of sequence of numbers
	  If a scalar or a 1-element sequence, it's the radius of a sphere.
	  If a 2- or 3-element sequence, then it defines the (a, c) or
	    (a, b, c) of an ellipsoid.
	viewpt : Vector
	  The view point vector in body-fixed frame of the object.  Only
	  parallel projection is considered, so the distance of viewer
	  `viewpt.norm()` doesn't matter.
	pxlscl : scalar number
	  The size of pixel in the same unit as `body`.
	pa : scalar
	  Position angle of image plane, measured from up to left (ccw).
	imsz : 2-element sequence
	  Image size (y, x)
	center : 2-element sequence
	  The CCD coordinates of body center

	Return
	------
	lon, lat : two arrays
	  Each array elements contains the longitude and latitude of
	  corresponding pixel in the body-fixed frame.   The size of
	  arrays is defined by `imsz`.

	Algorithm
	---------
	For a sphere, calculate the z-coordinates for each (x,y) position,
	then convert the (x,y,z) to the body-fixed frame based on `viewpt`
	for their (lon, lat).

	For an ellipsoid, convert the image plane (x,y,0) to body-fixed
	frame coordinate (x',y',z'), based on `viewpt`.  Then find the
	intersection of the elllipsoid and the line parallel to the image
	plane normal and passing each (x',y',z') by solving a quadratic
	equation.  The (lon, lat) are calculated based on the coordinates
	of intersection.

	v1.0.0 : JYL @PSI, 2/23/2016
	'''
	imsz = np.asarray(imsz)
	if center is None:
		center = (imsz-1.)/2
	else:
		center = np.asarray(center)

	# CCD coordinates
	yarr, xarr = makenxy(-center[0]*pxlscl, (imsz[0]-1-center[0])*pxlscl, imsz[0], -center[1]*pxlscl, (imsz[1]-1-center[1])*pxlscl, imsz[1])

	# set up shape
	sphere = False
	if hasattr(r, '__iter__'):
		if len(r) == 1:
			sphere = True
		elif len(r) == 2:
			a, c = r
			b = a
		elif len(r) == 3:
			a, b, c = r
		else:
			raise Warning('`r` has {0} elements, the first three elements define the triaxial ellpsoid shape, others are discarded')
			a, b, c = r[:3]
	else:
		sphere = True

	# calculate lon/lat
	if sphere:  # for a sphere
		z = np.sqrt(r*r-xarr*xarr-yarr*yarr)
		w = np.isfinite(z)
		v = proj2body(Vector(xarr[w], yarr[w], z[w]), viewpt, pa=pa)
		lon = np.zeros_like(xarr)*np.nan
		lat = np.zeros_like(yarr)*np.nan
		lon[w] = v.lon
		lat[w] = v.lat
		return lon, lat
	else:  # for an ellipsoid
		vb = proj2body(Vector(xarr,yarr,np.zeros_like(xarr)), viewpt, pa=pa)
		n = viewpt
		a2 = 1./a**2
		b2 = 1./b**2
		c2 = 1./c**2
		p1 = (np.ones_like(vb.x)*(n.x**2*a2+n.y**2*b2+n.z**2*c2)).flatten()
		p2 = 2*(n.x*vb.x*a2+n.y*vb.y*b2+n.z*vb.z*c2).flatten()
		p3 = (vb.x*vb.x*a2+vb.y*vb.y*b2+vb.z*vb.z*c2-1).flatten()
		t = (np.ones_like(p1)*np.nan).flatten()
		for i in range(len(t)):
			s = quadeq(p1[i], p2[i], p3[i])
			if s is not None:
				if len(s) == 2:
					t[i] = s.max()
				else:
					t[i] = s[0]
		w = np.isfinite(t)
		x = n.x*t[w]+vb.x.flatten()[w]
		y = n.y*t[w]+vb.y.flatten()[w]
		z = n.z*t[w]+vb.z.flatten()[w]
		vect = Vector(x, y, z)
		lon = np.ones_like(t)*np.nan
		lon[w] = vect.lon
		lat = np.ones_like(t)*np.nan
		lat[w] = vect.lat
		return lon.reshape(imsz), lat.reshape(imsz)


def lonlat2xy(lon, lat, r, viewpt, pa=0., center=None, pxlscl=None, deg=True):
	'''Convert the body-fixed (lon, lat) coordinates to the corresponding
	(x,y) pixel position in a CCD

	lon, lat : array-like
	  The longitude and latitude to be converted.  They must have the
	  same shape.
	r : number of sequence of numbers
	  If a scalar or a 1-element sequence, it's the radius of a sphere.
	  If a 2- or 3-element sequence, then it defines the (a, c) or
	    (a, b, c) of an ellipsoid.
	viewpt : Vector
	  The view point vector in body-fixed frame of the object.  Only
	  parallel projection is considered, so the distance of viewer
	  `viewpt.norm()` doesn't matter.
	pa : scalar, optional
	  Position angle of the z-axis in image plane, measured from up to
	  left (ccw).
	center : 2-element sequence, optional
	  The CCD coordinates (y0,x0) of body center.  If `None`, then the
	  body center will be at (0,0) image coordinates
	pxlscl : scalar number, optional
	  The size of pixel in the same unit as `body`.  If `None`, then
	  the pixel size will be assumed to be 1 with the same unit as
	  `r`.
	deg : bool, optional
	  The unit of input `lon` and `lat`.

	Return
	------
	x, y : two arrays
	  Two arrays of the same shape as `lon` and `lat` containing the
	  corresponding (x, y) image coordinates.

	Algorithm
	---------
	Calculate the (x,y,z) for input (lon,lat) using the shape model
	defined by `r`, discard those with surface normal pi/2 away from
	`viewpt`, convert to image plane, return (x,y)

	v1.0.0 : JYL @PSI, 2/24/2016
	'''
	lon = np.asarray(lon).astype(float)
	lat = np.asarray(lat).astype(float)
	if lon.shape != lat.shape:
		raise ValueError('`lon` and `lat` must have the same shape, {0} {1} received'.format(lon.shape, lat.shape))

	# set up shape
	if hasattr(r, '__iter__'):
		if len(r) == 1:
			a = r[0]
			b = a
			c = a
		elif len(r) == 2:
			a, c = r
			b = a
		elif len(r) == 3:
			a, b, c = r
		else:
			raise Warning('`r` has {0} elements, the first three elements define the triaxial ellpsoid shape, others are discarded')
			a, b, c = r[:3]
	else:
		a = r
		b = a
		c = a

	# calculate projection
	if deg:
		lon = np.deg2rad(lon)
		lat = np.deg2rad(lat)
		pa = np.deg2rad(pa)
	coslon = np.cos(lon)
	sinlon = np.sin(lon)
	coslat = np.cos(lat)
	sinlat = np.sin(lat)
	v = Vector(a*coslon*coslat, b*sinlon*coslat, c*sinlat)
	angle = Vector(v.x/a**2, v.y/b**2, v.z/c**2).vsep(viewpt)
	w = angle<(np.pi/2)  # only keep where normal is within pi/2 of viewpt
	v = vect2proj(v[w], viewpt, pa=pa)
	x = np.ones_like(lon)*np.nan
	y = np.ones_like(lon)*np.nan
	x[w] = v.x
	y[w] = v.y

	# add pixel scale and pixel center
	if pxlscl is not None:
		x /= pxlscl
		y /= pxlscl
	if center is not None:
		x += center[1]
		y += center[0]

	return x, y


def xy2iea(*args, **kwargs):
	'''Calculate i, e, alpha for a CCD image of a triaxial ellpsoidal object.

	Parameters are similar to `xy2lonlat`, with an additional positional
	parameter of `vsun` for the vector of the Sun in body-fixed frame.
	See `xy2lonlat` for detailed information.

	Returns (i, e, a).

	v1.0.0 : JYL @PSI
	'''

	lon,lat = xy2lonlat(*args, **kwargs)

