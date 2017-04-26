'''
Function module provides support to anaytic functions and
numerically defined functions, and the basic operatures such as add,
subtract, multiplication, division, and power.  It also provides some
some convenience features such as converting to array, table, and
making plot.
'''

import numpy as np
import astropy.units as u
import abc
from .core import condition

__all__ = ['Function', 'AnalyticFunction', 'NumericFunction', 'Spectrum', 'specadd', 'specsub']


class cfunc(object):
	'''Constant function class'''

	def __init__(self, c):
		import numbers
		assert isinstance(c, numbers.Number)
		self._c = c

	def __call__(self, *var):
		if len(var) == 1:
			if hasattr(var[0],'__init__'):
				return np.ones_like(var[0])*self._c
			else:
				return self._c
		y = 0.
		for v in var:
			y = y*np.asarray(v)
		return np.ones_like(y)*self._c

	def __str__(self):
		return '<constant function c = {0}>'.format(self._c)

	def __repr__(self):
		return '<constant function c = {0}>'.format(self._c)


zeros = cfunc(0.)

ones = cfunc(1.)


def _eval_func(func, *var):
	'''
 Recursively evaluate a function expression in a list

 Parameters
 ----------
 func : list with 3 elements
   [func1, op, func2]: func1 and func2 are two function references,
   numbers, or lists like `func` itself, and op is an operater in
   [+, -, *, /, **]
 *var : parameter list to be passed to functions

 Return
 ------
 Evaluates the functions in the list recursively and return the result
   result = func1(*var) op func2(*var)

 v1.0.0 : JYL @PSI, December 7, 2014
	'''
	import numbers
	if isinstance(func, list):
		op = func[1]
		v1 = _eval_func(func[0], *var)
		v2 = _eval_func(func[2], *var)
		if op is '+':
			return v1+v2
		elif op is '-':
			return v1-v2
		elif op is '*':
			return v1*v2
		elif op is '**':
			return v1**v2
		elif op is '/':
			return v1/v2
		else:
			raise TypeError('{0} is not a valid operater'.format(op))
	else:
		if hasattr(func, '__call__'):
			return func(*var)
		else:
			assert isinstance(func, numbers.Number), 'a number or a callable function is expected, get a {0}'.format(type(func))
			return func


class Function(object):
	'''Abstract Function class to define a function f(x, y, ...)

	Properties
	----------
	name : str
	  Name of function
	mode : str
	  Mode of function, either 'analytic' or 'numeric'
	ndim : int
	  Number of dimensions
	xlabel, label : str or str array
	  Labels of function base and function value
	base : ndarray
	  Function base (for numeric function)
*	baseunit, unit : astropy Unit
*	  The units of function base and function value
*	baseequiv, equiv :
*	  Equivalencies of function base unit and value unit
	errs : ndarray
	  The measurement errors of function base and/or function value

	Methods
	-------

	'''

	__metaclass__ = abc.ABCMeta

	# Data
	@property
	def base(self):
		return self._base
	@base.setter
	def base(self, value):
		from .util import islengthone
		import numbers
		if self._ndim == 1:
			self._base = value*u.dimensionless_unscaled
			if self._mode == 'analytic':
				self._value = self.__call__(self._base)
			else:
				self._setfunc(self._base, self._value)
		else:
			assert not islengthone(value), 'Size of `base` must be greater than `ndim` = {0}'.format(self.ndim)
			assert len(value) == self.ndim, 'The length of `base` must be the same as `ndim` = {0}'.format(self.ndim)
			self._base = []
			nv = np.size(value[0])
			for i in range(self.ndim):
				assert np.size(value[i]) == nv, 'all parameters in `base` must have the same size'
				self._base.append(value[i]*u.dimensionless_unscaled)
			if self._mode == 'analytic':
				self._value = self.__call__(*tuple(self._base))
			else:
				self._setfunc(self._base, self._value)
		if self._mode == 'analytic':
			if not isinstance(self._value, u.quantity.Quantity):
				self._value *= u.dimensionless_unscaled
			if self._unit is not None:
				assert self._unit.is_equivalent(self._value.unit), 'unequivalent unit in `self.unit`'
				self._value = self._value.to(self._unit)
			else:
				self._unit = self._value.unit

	@property
	def baseunit(self):
		if self._base is not None:
			if self._ndim == 1:
				return self._base.unit
			else:
				out = []
				for i in range(self._ndim):
					out.append(self._base[i].unit)
				return out
		else:
			return u.dimensionless_unscaled
	@baseunit.setter
	def baseunit(self, value):
		self.SetBaseUnit(value)

	@property
	def unit(self):
		from .util import condition
		return condition(self._value == None, u.dimensionless_unscaled, self._value.unit)
	@unit.setter
	def unit(self, value):
		self.SetUnit(value)

	@property
	def errs(self):
		return self._errs
	@errs.setter
	def errs(self, value):
		self._errs = value*u.dimensionless_unscaled

	@property
	def ndim(self):
		return self._ndim

	@abc.abstractmethod
	def __init__(self):
		'''Initialize class
		'''
		self.name = None	# function name
		self.xlabel = None	# label of independent variable
		self.label = 'y'	# label of function
		self._func = None	# function form
		self._base = None	  # function base (useful for numerical mode)
		self._baseequiv = None	# unit equivalence for base
		self._value = None	# function values (useful for numerical mode)
		self._unit = None	# unit of function
		self._equiv = None	# unit equivalence for function
		self._mode = ''	# 'numeric' or 'analytic'
		self._ndim = None	# number of independent variable

	def __call__(self, *var):
		'''Calculate spectrum at independent variable `var`'''
		from .util import ulen
		if len(var) != self._ndim:
			raise ValueError('function takes {0} arguments ({1} given)'.format(self._ndim, len(var)))
		if self._mode == 'analytic':
			return _eval_func(self._func, *var)
		elif self._mode == 'numeric':
			return self._func(*var)
		else:
			return ValueError('unrecognized function mode: `analytic` or `numeric` expected, {0} found'.format(self.mode))

	def __neg__(self):
		return type(self)._operation(0, self, '-', False)

	def __add__(self, other):
		return type(self)._operation(self, other, '+', False)

	def __radd__(self, other):
		return type(self)._operation(other, self, '+', True)

	def __sub__(self, other):
		return type(self)._operation(self, other, '-', False)

	def __rsub__(self, other):
		return type(self)._operation(other, self, '-', True)

	def __mul__(self, other):
		return type(self)._operation(self, other, '*', False)

	def __rmul__(self, other):
		return type(self)._operation(other, self, '*', True)

	def __div__(self, other):
		return type(self)._operation(self, other, '/', False)

	__truediv__ = __div__

	def __rdiv__(self, other):
		return type(self)._operation(other, self, '/', True)

	__rtruediv__ = __rdiv__

	def __pow__(self, other):
		return type(self)._operation(self, other, '**', False)

	@abc.abstractmethod
	def __str__(self): pass

	@abc.abstractmethod
	def __repr__(self): pass

	def __array__(self, *var):
		import numbers
		from .util import condition, ulen
		if len(var) == 0:
			assert (self._base != None) and (ulen(self._base) != 0), '`self.base` is not defined'
			x = self._base
			v = self._value
		else:
			x = var
			v = self.__call__(*var)
		if self.ndim > 1:
			for i in range(self.ndim):
				if isinstance(x[i].value, numbers.Number):
					x[i] = [x[i].value]*x[i].unit
			v = [v.value]*v.unit
		return np.asarray(np.vstack((x, v)))

	@abc.abstractmethod
	def __copy__(self): pass

	@abc.abstractmethod
	def _setfunc(self, base, value): pass

	def Copy(self):
		return self.__copy__()

	def SetBaseUnit(self, value, equiv=None):
		assert self._base != None, '`base` is undefined'
		self._baseequiv = equiv
		xunit = self.baseunit
		if self._ndim == 1:
			value = u.Unit(value)
			if u.dimensionless_unscaled not in [value, xunit]:
				assert value.is_equivalent(self.baseunit, equivalencies=self._baseequiv), 'unequivalent unit'
			self._base = self._base.to(value)
		else:
			assert len(value) == self._ndim, 'size of input must be the same of `ndim` = {0}'.format(self._ndim)
			for i in range(self._ndim):
				v = u.Unit(value[i])
				if u.dimensionless_unscaled not in [v, xunit[i]]:
					assert v.is_equivalent(xunit[i], equivalencies=self._baseequiv), 'unequivalent unit in the {0}th unit'.format(i)
				self._base[i] = self._base[i].to(v)
		if self._mode == 'numeric':
			self._setfunc(self._base, self._value)

	def SetUnit(self, value, equiv=None):
		self._equiv = equiv
		value = u.Unit(value)
		if self._unit is not None:
			assert value.is_equivalent(self._unit, equivalencies=self._equiv), 'unequivalent unit'
		self._unit = value
		if self._value != None:
			self._value = self._value.to(value, equivalencies=self._equiv)
		if self._mode == 'numeric':
			self._setfunc(self._base, self._value)

	def AsTable(self, *var):
		from astropy import table
		import numbers
		from .util import condition, ulen
		if len(var) == 0:
			assert (self._base != None) and (ulen(self._base) != 0), '`self.base` is not defined'
			x = self._base
			v = self._value
		else:
			x = var
			v = self.__call__(*var)
		if self._ndim == 1:
			if isinstance(x.value, numbers.Number):
				c1 = [x.value]*x.unit
				c2 = [v.value]*v.unit
			else:
				c1 = x
				c2 = v
			tbl = table.Table((c1, c2), names=[self.xlabel, self.label])
		else:
			tbl = table.Table()
			for i in range(self.ndim):
				c = condition(isinstance(x[i].value, numbers.Number), [x[i].value]*x[i].unit, x[i])
				tbl.add_column(table.Column(c, name=self.xlabel[i]))
			c = condition(isinstance(v.value, numbers.Number), [v.value]*v.unit, v)
			tbl.add_column(table.Column(c, name=self.label))
		return tbl

	def AsArray(self, *var): return self.__array__(*var)

	def Plot(self, *var, **kwarg):
		assert self.ndim <= 2, 'Cannot plot dat for `ndim` > 2'
		from matplotlib import pyplot as plt
		from .util import pplot, condition, ulen
		if len(var) == 0:
			assert (self._base != None) and (ulen(self._base) != 0), '`self.base` is not defined'
			x = self._base
			v = self._value
		else:
			x = var[0]
			v = self.__call__(*var)
		ax = kwarg.pop('ax', None)
		if ax == None:
			ax = plt.gca()
		if self.ndim == 1:
			ax.plot(x.value, v.value, **kwarg)
			xunit = condition(x.unit == u.dimensionless_unscaled, '', ' ['+str(x.unit)+']')
			yunit = condition(v.unit == u.dimensionless_unscaled, '', ' ['+str(v.unit)+']')
			pplot(ax, xlabel=self.xlabel+xunit, ylabel=self.label+yunit)
		else:
			print 'plotting 2-D figure'

	def CopyProperties(self, other):
		assert isinstance(other, type(self)), 'Can only copy properties from the same class or inherited class'
		from copy import deepcopy
		for k in self.__dict__:
			self.__dict__[k] = deepcopy(other.__dict__[k])

	@staticmethod
	@abc.abstractmethod
	def _operation(f1, f2, op, swap):
		raise NotImplementedError('function not implemented')


class AnalyticFunction(Function):
	'''Analytic function class'''

	def __init__(self, func, ndim, base=None):
		super(AnalyticFunction, self).__init__()
		self._mode = 'analytic'
		assert isinstance(ndim, int), '`ndim` must be an integer'
		self._ndim = ndim
		self._func = func
		if self._ndim == 1:
			self.xlabel = 'x'
		else:
			self.xlabel = []
			for i in range(self._ndim):
				self.xlabel.append('x'+`i`)
		if base != None:
			self.base = base

	def __str__(self):
		from .util import condition
		keys = [('name', condition(self.name == '', "''", self.name)),
				('ndim', self._ndim),
				('xlabel', self.xlabel),
				('label', condition(self.label == '', "''", self.label)),
				('base', condition(self._base == None, 'Undefined', self._base)),
				('value', condition(self._value == None, 'Undefined', self._value)),
				('unit', condition(self.unit == u.dimensionless_unscaled, 'dimensionless', self.unit))]
		parts = ['{0} = {1}'.format(k,v) for k,v in keys]
		return '<AnalyticFunction class object>\n'+'\n'.join(parts)

	def __repr__(self):
		return "<AnalyticFunction: name = '{0}', ndim = {1}>".format(self.name, self._ndim)

	def __copy__(self):
		out = AnalyticFunction(self._func, self._ndim)
		out.CopyProperties(self)
		return out

	def _setfunc(self, base, value): pass

	@staticmethod
	def _operation(f1, f2, op, swap):
		import numbers
		from .util import condition
		if swap:
			if f1 == 1:
				pass

		if isinstance(f2, AnalyticFunction):
			assert f1._ndim == f2._ndim, 'Unmatched dimensions for '+op
			if (f1._func == f2._func) and (op in ['+','-','/']):
				if op == '+':
					f2 = 2
					op = '*'
				if op == '-':
					out = condition(swap, f2.Copy(), f1.Copy())
					out._func = zeros
					return out
				if op == '/':
					out = condition(swap, f2.Copy(), f1.Copy())
					out._func = ones
					return out
			else:
				out = AnalyticFunction([f1._func, op, f2._func], f1.ndim)
				if swap:
					out._func[0], out._func[2] = out._func[2], out._func[0]
				out.base = f1._base
		if isinstance(f2, numbers.Number):
			if (f2 == 0.) and (op in ['+','*']):
				if op == '+':
					return f1.Copy()
				if op == '*':
					out = f1.Copy()
					out._func = zeros
					return out
			elif (f2 == 0.) and (op == '-') and (not swap):
				return f1.Copy()
			elif (f2 == 1.) and (op == '/') and (not swap):
				return f1.Copy()
			elif (f2 == 1.) and (op in ['*','**']):
				return f1.Copy()
			else:
				out = AnalyticFunction([f1._func, op, f2], f1.ndim)
			if swap:
				out._func[0], out._func[2] = out._func[2], out._func[0]
			out.base = f1.base
		if isinstance(f2, NumericFunction):
			raise TypeError('Operation with NumericFunction undefined yet.')
		if 'out' not in locals():
			raise TypeError('Undefined operation')
		return out


class NumericFunction(Function):
	'''Numeric function class'''

	def __init__(self, base, errs=None):
		super(NumericFunction, self).__init__()
		self._mode = 'numerical'
		base *= u.dimensionless_unscaled
		if base.shape[0] < 2:
			raise ValueError('The first dimension of `base` must be >= 2')
		self._base = base
		if errs != None:
			errs *= u.dimensionless_unscaled
			if errs.shape != base.shape:
				raise ValueError('`errs` must have the same shape as `base`')
			self._errs = errs
		self._ndim = base.shape[0]-1

		# define interpolation function
		from scipy.interpolate import interp1d
		self._fill_value = 0.
		self._method = 'linear'
		self._bounded = True


class Spectrum1(NumericFunction):
	'''Spectrum class'''

	def __init__(self, x, y, xerr=None, err=None):
		x = (x*u.dimensionless_unscaled).reshape(-1)
		y = (y*u.dimensionless_unscaled).reshape(-1)
		assert x.shape == y.shape, '`x` and `y` must have the same shape'
		base = [x,y]
		if (xerr == None) and (yerr == None):
			errs = None
		else:
			if xerr == None:
				xerr = np.zeros(x.shape)*u.dimensionless_unscaled
			else:
				xerr = (xerr*u.dimensionless_unscaled).reshape(-1)
			if err == None:
				err = np.zeros(x.shape)*u.dimensionless_unscaled
			else:
				err = (err*u.dimensionless_unscaled).reshape(-1)
			assert xerr.shape == err.shape, '`xerr` and `err` must have the same shape'
			errs = [xerr, err]
		super(NumericFunction, self).__init__(basis, errs)
		self._labels = ['Wavelength', 'Spectrum']


class Spectrum(object):
	'''Spectrum class
	'''

	name = None
	xlabel = 'x'
	ylabel = 'y'
	_xunit = u.dimensionless_unscaled
	_unit = u.dimensionless_unscaled
	_kind = 'linear'

	def __init__(self, x, f=None, xerr=None, err=None, bounded=True, fill_value=np.nan, kind='linear', name=None):
		'''Initialize class

		spectrum = Spectrum(values, **kwarg)

		values : 2xN or 3xN or 4xN array-like, or astropy.table.Table
 			object with at least two columns
			The input spectra to be added.
			If 2xN array-like or a table with 2 columns, it will be interpreted
			as (x, f).
			If 3xN array-like or a table with 3 columns, it will be interpreted
			as (x, f, f_err).
			If 4xN array-like or a table with 4 columsn, it will be interpreted
			as (x, f, x_err, f_err)

		or

		spectrum = Spectrum(x, f, **kwarg)

		x, f : array like, or astropy Quantities
		  Spectrum f(x) used to initialize the class

		Other parameters
		----------------
		name : str, optional
		  Name of the spectrum
		bounded : bool, optional
		  If `True`, then if asked for calculate spectrum outside of the
		  boundary of `x`, then a ValueError is raised.  If `False`,
		  then `fill_value` is returned.  See `bounds_error` keyword of
		  scipy.interpolate.interp1d
		fill_value : float, optional
		  See scipy.interpolate.interp1d
		kind : str or int, optional
		  See scipy.interpolate.interp1d
		xlabel, ylabel : str, optional
		  The label of x and y
		'''

		from scipy.interpolate import interp1d
		from astropy import table

		self.name = name
		if f is None:
			if isinstance(x, table.Table):
				ncol = len(x.columns)
				assert ncol>=2, 'Input table must contain >2 columns'
				f = np.asarray(x.columns[1])
				if x.columns[1].unit is not None:
					f *= x.columns[1].unit
				if ncol == 3:
					ferr = np.asarray(x.columns[2])
					if x.columns[2].unit is not None:
						ferr *= x.columns[2].unit
				if ncol > 3:
					xerr = np.asarray(x.columns[2])
					if x.columns[2].unit is not None:
						xerr *= x.columns[2].unit
					ferr = np.asarray(x.columns[3])
					if x.columns[3].unit is not None:
						ferr *= x.columns[3].unit
				if x.columns[0].unit is None:
					x = np.asarray(x.columns[0])
				else:
					x = np.asarray(x.columns[0])*x.columns[0].unit
			else:
				ncol = len(x)
				assert ncol>=2, 'The `len` of input must be >= 2.'
				f = np.asarray(x[1])
				if ncol == 3:
					ferr = np.asarray(x[2])
				if ncol > 3:
					xerr = np.asarray(x[2])
					ferr = np.asarray(x[3])
				x = np.asarray(x[0])

		x *= u.dimensionless_unscaled
		f *= u.dimensionless_unscaled
		assert np.size(x) == np.size(f), 'Initialization Error: `x` and `f` must have the same size.'
		self._xunit = x.unit
		self._unit = f.unit
		self._kind = kind
		self._spfunc = interp1d(x.value, f.value, kind=self._kind, bounds_error=bounded, fill_value=fill_value)
		if xerr is not None:
			self._xerr = (xerr*u.dimensionless_unscaled).to(self._xunit, equivalencies=u.spectral()).value
		else:
			self._xerr = None
		if err is not None:
			self._ferr = (err*u.dimensionless_unscaled).to(self._unit, equivalencies=u.spectral_density(x)).value
		else:
			self._ferr = None

	def __call__(self, x, equiv=u.spectral):
		'''Calculate spectrum at `x`'''
		x = (x*u.dimensionless_unscaled).to(self._xunit, equivalencies=equiv())
		xv = np.asarray(x.value)
		xu = x.unit
		if self._spfunc.bounds_error:
			assert (xv.min() >= self._spfunc.x.min()) and (xv.max() <= self._spfunc.x.max()), '`x` must be in the range of {0} and {1}'.format((self._spfunc.x.min()*self._xunit).to(xu, equivalencies=equiv()), (self._spfunc.x.max()*self._xunit).to(xu, equivalencies=equiv()))
		return self._spfunc(xv)*self._unit

	def __neg__(self):
		out = Spectrum(self.x, -self.y, xerr=self.xerr, err=self.err)
		out.CopyProperties(self)
		return out

	def __add__(self, other):
		return specadd(self, other)

	def __radd__(self, other):
		return specadd(other, self)

	def __sub__(self, other):
		return specsub(self, other)

	def __rsub__(self, other):
		return specsub(other, self)

	def __mul__(self, other):
		pass

	def __rmul__(self, other):
		pass

	def __div__(self, other):
		pass

	def __rdiv__(self, other):
		pass

	def __eq__(self, other):
		pass

	def __ne__(self, other):
		return not self.__eq__(other)

	def __iter__(self):
		data = self.__array__()
		return iter(data.T)

	def __len__(self):
		return len(self._spfunc.y)

	def __repr__(self):
		return self._format_repr()

	def _format_repr(self):
		out = '<Spectrum "'+str(condition(self.name is None, '', self.name))+'": shape='+str(self.__array__().shape)
		if self._xunit is not u.dimensionless_unscaled:
			out += ", xunit='"+str(self._xunit)+"'"
		if self._unit is not u.dimensionless_unscaled:
			out += ", unit='"+str(self._unit)+"'"
		out += '>'
		return out

	def __str__(self):
		return self._format_str()

	def _format_str(self):
		keys = [('name', self.name),
				('points', self.__len__()),
				('xlabel', self.xlabel),
				('xunit', np.where(self._xunit is u.dimensionless_unscaled, None, self._xunit)),
				('ylabel', self.ylabel),
				('unit', np.where(self._unit is u.dimensionless_unscaled, None, self._unit)),
				('x', self._spfunc.x),
				('y', self._spfunc.y)]
		if self._xerr is not None:
			keys.append(('xerr=', self._xerr))
		if self._ferr is not None:
			keys.append(('err=', self._ferr))
		parts = ['{0} = {1}'.format(k,v) for k,v in keys]
		return '<Spectrum class object>\n'+'\n'.join(parts)

	def __array__(self):
		out = [self._spfunc.x, self._spfunc.y]
		if self._xerr is not None:
			out.append(self._xerr)
		if self._ferr is not None:
			out.append(self._ferr)
		return np.asarray(out)

	def __copy__(self):
		if self._xerr is not None:
			xerr = self.xerr
		else:
			xerr = None
		if self._ferr is not None:
			ferr = self.err
		else:
			ferr = None
		out = Spectrum(self.x, self.y, xerr=xerr, err=ferr, bounded=self._spfunc.bounds_error, fill_value=self._spfunc.fill_value)
		out.CopyProperties(self)
		return out

	@property
	def xunit(self):
		'''Unit of x'''
		return self._xunit
	@xunit.setter
	def xunit(self, value):
		self.SetxUnit(value)

	@property
	def unit(self):
		'''Unit of f(x)'''
		return self._unit
	@unit.setter
	def unit(self, value):
		self.SetUnit(value)

	@property
	def bounded(self):
		'''See scipy.interpolate.interp1d'''
		return self._spfunc.bounds_error
	@bounded.setter
	def bounded(self, value):
		assert type(value) is bool
		self._spfunc.bounds_error = value

	@property
	def fill_value(self):
		'''See scipy.interpolate.interp1d'''
		return self._spfunc.fill_value
	@fill_value.setter
	def fill_value(self, value):
		self._spfunc.fill_value = value

	@property
	def kind(self):
		'''See scipy.interpolate.interp1d'''
		return self._kind
	@kind.setter
	def kind(self, value):
		assert value in 'linear nearest zero slinear quadratic cubic'.split(), '`kind` must be in '+str('linear nearest zero slinear quadratic cubic'.split())
		from scipy.interpolate import interp1d
		self._kind = value
		self._spfunc = interp1d(self._spfunc.x, self._spfunc.y, kind=value, bounds_error=self._spfunc.bounds_error, fill_value=self._spfunc.fill_value)
		self._kind = value

	@property
	def x(self):
		out = self._spfunc.x
		if self._xunit is not u.dimensionless_unscaled:
			out *= self._xunit
		return out

	@property
	def y(self):
		out = self._spfunc.y
		if self._unit is not u.dimensionless_unscaled:
			out *= self._unit
		return out

	@property
	def xerr(self):
		if self._xerr is None:
			return None
		out = self._xerr
		if self._xunit is not u.dimensionless_unscaled:
			out *= self._xunit
		return out

	@property
	def err(self):
		if self._ferr is None:
			return None
		out = self._ferr
		if self._unit is not u.dimensionless_unscaled:
			out *= self._unit
		return out

	def SetxUnit(self, value, equiv=None):
		from scipy.interpolate import interp1d
		value = u.Unit(value)
		# If either one is dimensionless, then just set it
		if (self._xunit is u.dimensionless_unscaled) or (value is u.dimensionless_unscaled):
			self._xunit = value
			return
		# Otherwise convert x to new unit before set it
		assert value.is_equivalent(self._xunit, equivalencies=equiv), 'Inequivalent unit [{0}] with the original unit [{1}].'.format(value, self._xunit)
		self._spfunc = interp1d((self.x.to(value, equivalencies=equiv)).value, self.y.value, kind=self._kind, bounds_error=self.bounded, fill_value=self.fill_value)
		if self._xerr is not None:
			self._xerr = (self.xerr.to(value, equivalencies=equiv)).value
		self._xunit = value

	def SetUnit(self, value, equiv=None):
		from scipy.interpolate import interp1d
		value = u.Unit(value)
		# If either one is dimensionless, then just set it
		if (self._unit is u.dimensionless_unscaled) or (value is u.dimensionless_unscaled):
			self._unit = value
			return
		# Otherwise convert x to new unit before set it
		assert value.is_equivalent(self._unit, equivalencies=equiv), 'Inequivalent unit [{0}] with the original unit [{1}].'.format(value, self._unit)
		self._spfunc = interp1d(self._spfunc.x, self.y.to(value, equivalencies=equiv).value, kind=self._kind, bounds_error=self.bounded, fill_value=self.fill_value)
		if self._ferr is not None:
			self._ferr = self.err.to(value, equivalencies=equiv).value
		self._unit = value

	def AsTable(self):
		'''Return spectrum in astropy table'''
		from astropy.table import Table, Column
		spec = Table([self._spfunc.x, self._spfunc.y], names='x f'.split())
		spec['x'].unit = self._xunit
		spec['f'].unit = self._unit
		if self._xerr is not None:
			spec.add_column(Column(self._xerr, name='xerr', unit=self._xunit))
		if self._ferr is not None:
			spec.add_column(Column(self._ferr, name='err', unit=self._unit))
		return spec

	def AsArray(self):
		return self.__array__()

	def Plot(self, **kwarg):
		'''Plot spectrum
		**kwarg: keywords for matplotlib.pyplot.errorbar()'''
		from matplotlib import pyplot as plt
		from jylipy import pplot
		plt.errorbar(self._spfunc.x, self._spfunc.y, yerr=self._ferr, xerr=self._xerr, **kwarg)
		if self._xunit is u.dimensionless_unscaled:
			xlbl = self.xlabel
		else:
			xlbl = self.xlabel+' ['+str(self._xunit)+']'
		if self._unit is u.dimensionless_unscaled:
			ylbl = self.ylabel
		else:
			ylbl = self.ylabel+' ['+str(self._unit)+']'
		pplot(xlabel=xlbl,ylabel=ylbl, title=self.name)

	def Bin(self, bin):
		'''Return a binned spectrum object'''
		from jylipy import rebin
		if self._xerr is None:
			xb = rebin(self._spfunc.x, bin, mean=True)*self._xunit
			xeb = None
		else:
			xb, xeb = rebin(self._spfunc.x, bin, mean=True, weight=1/self._xerr**2)
			xb *= self._xunit
			xeb = np.sqrt(1/xeb)*self._xunit
		if self._ferr is None:
			fb = rebin(self._spfunc.y, bin, mean=True)*self._unit
			xeb = None
		else:
			fb, feb = rebin(self._spfunc.y, bin, mean=True, weight=1/self._ferr**2)
			fb *= self._unit
			feb = np.sqrt(1/feb)*self._unit
		name = `bin`.strip()+'pt_Binned'
		if self.name is not None:
			name = self.name+'_'+name
		return Spectrum(xb, fb, xerr=xeb, err=feb, name=name, bounded=self._spfunc.bounds_error, fill_value=self._spfunc.fill_value, kind=self._kind)

	def Copy(self):
		return self.__copy__()

	def CopyProperties(self, spec):
		'''Copy the properties of the input spectrum'''
		self.name = spec.name
		self.xlabel = spec.xlabel
		self.ylabel = spec.ylabel
		self.bounded = spec.bounded
		self.fill_value = spec.fill_value
		if self.kind != spec.kind:
			self.kind = spec.kind
		self.unit = spec.unit
		self.xunit = spec.xunit

	def Resample(self, x):
		'''Resample the spectrum based on new `x`'''
		pass


def specadd(spec1, spec2):
	'''
 Add two spectra

 Parameters
 ----------
 spec1, spec2 : 2xN or 3xN or 4xN array-like, or astropy.table.Table
 object with at least two columns, or Spectrum object
   The input spectra to be added.
   If 2xN array-like or a table with 2 columns, it will be interpreted
   as (x, f).
   If 3xN array-like or a table with 3 columns, it will be interpreted
   as (x, f, f_err).
   If 4xN array-like or a table with 4 columsn, it will be interpreted
   as (x, f, x_err, f_err)

 Return
 ------
 spectrum : Spectrum
   The sum spectrum, sampled at spec1.x.

 v1.0.0 : JYL @PSI, December 3, 2014
	'''

	from .math import add
	from .util import islengthone, condition

	# Assert at least one input is Spectrum instance
	assert isinstance(spec1, Spectrum) or isinstance(spec2, Spectrum), 'At least one input has to be an instance of Spectrum class.'

	# For the case that one input is a length-one parameter
	if not isinstance(spec1, Spectrum):
		if islengthone(spec1):
			if isinstance(spec1, list):
				spec1 = spec1[0]
			out = Spectrum(spec2.x, spec1*u.dimensionless_unscaled+spec2.y, xerr=spec2.xerr, err=spec2.err)
			out.CopyProperties(spec2)
			out.name = 'sum_spec1_'+condition(spec2.name is None, 'spec2', spec2.name)
			return out
	if not isinstance(spec2, Spectrum):
		if islengthone(spec2):
			if isinstance(spec2, list):
				spec2 = spec2[0]
			out = Spectrum(spec1.x, spec2*u.dimensionless_unscaled+spec1.y, xerr=spec1.xerr, err=spec1.err)
			out.CopyProperties(spec1)
			out.name = 'sum_'+condition(spec1.name is None, 'spec1', spec1.name)+'_spec2'
			return out

	# Other cases
	if not isinstance(spec1, Spectrum):
		try:
			spec1 = Spectrum(spec1)
		except:
			raise ValueError('Input parameter error.')
	if not isinstance(spec2, Spectrum):
		try:
			spec2 = Spectrum(spec2)
		except:
			raise ValueError('Input Parameter error.')

	name = 'sum_'+condition(spec1.name is None, 'spec1', spec1.name)+'_'+condition(spec2.name is None, 'spec2', spec2.name)

	spec2.unit = spec1.unit
	sp2r = spec2(spec1.x)
	if (spec1.x == spec2.x).all():
		if (spec1.err is not None) and (spec2.err is not None):
			spsum, err = add(spec1.y, sp2r, err1=spec1.err, err2=spec2.err)
		else:
			spsum = spec1.y+spec2.y
			err = None
	else:
		spsum = spec1.y+spec2.y
	if spec1.xerr is not None:
		xerr = spec1.xerr
	else:
		xerr = None
	out = Spectrum(spec1.x, spsum, xerr=xerr, err=err)
	out.CopyProperties(spec1)
	out.name = name
	return out


def specsub(spec1, spec2):
	'''
 Subtract two spectra

	'''

	assert isinstance(spec1, Spectrum) or isinstance(spec2, Spectrum), 'At least one input has to be an instance of Spectrum class.'
	if isinstance(spec2, Spectrum):
		return specadd(spec1, -spec2)
	else:
		if islengthone(spec2):
			if isinstance(spec2, list):
				spec2 = spec2[0]
			return specadd(spec1, -spec2)
		else:
			try:
				spec2 = Spectrum(spec2)
			except:
				raise ValueError('Input parameter error.')
			return specadd(spec1, -spec2)


# # Pre-defined solar spectra
from astropy.io import ascii
# # E490-AM0
# e490source = '/Users/jyli/work/references/Sun/E490_00a_AM0.tab'
# solar_e490 = Spectrum(ascii.read(e490source))
# solar_e490.xunit = 'um'
# solar_e490.unit = 'W m-2 um-1'
# solar_e490.name = 'Solar Spectrum E490-AM0'
# solar_e490.xlabel = 'Wavelength'
# solar_e490.ylabel = 'Solar Flux'
# # SUSIM
# susimsource = '/Users/jyli/work/references/Sun/susim_hires_norm_nocomments.tab'
# solar_susim = Spectrum(ascii.read(susimsource))
# solar_susim.xunit = 'nm'
# solar_susim.unit = 'W m-2 um-1'
# solar_susim.name = 'Solar Spectrum SUSIM'
# solar_susim.xlabel = 'Wavelength'
# solar_susim.ylabel = 'Solar Flux'


