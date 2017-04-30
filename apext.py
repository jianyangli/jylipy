'''astropy extension

Extended modules include:

table
time

v0.0.1 : 5/31/2015, JYL @PSI
'''


from astropy import table, units, modeling, time, modeling, constants, nddata
from astropy.io import fits, ascii
from astropy.modeling import Fittable1DModel, Parameter
import numpy as np


__all__ = [
	'table', 'time', 'modeling', 'units', 'fits', 'ascii', 'modeling', 'constants', 'nddata',  # Original astropy modules
	'Table', 'Column', 'show_table_in_browser', # Table
	'TimeET', 'TimeYDay', 'TimeYDayT', 'EpochTimeFormat', 'Time', 'month_number', 'month_name', # Time
	'MPFitter'  # Fitter
	]



#---------------------------------------------------------------------
# Extended Table and related classes and functions
#---------------------------------------------------------------------

class Table(table.Table):
	'''Extended astropy.table.Table class.

	Extended functions:

	__setitem__ : When assign values to a column, accepts astropy
	  Quantity and set the unit correctly.

	.getcolumn(k) : Returns astropy Qauntity for column k if the unit
	  is not None.

	.asdict() : Returns a copy of the table as an OrderedDict instance

	.query(keycol, key, valuecol=None, logic='and', op='==') :
	  Query the table column based on `key` in `keycol` and return
	  columns or table

	.index(keycol, key, logic='and', op='==') :
	  Similar to .query, but returns row indices

	.show_in_browser() : Same as the original method, but with the
	  default jsviewer=True
	'''

	def __setitem__(self, k, v):
		super(Table, self).__setitem__(k, v)
		if isinstance(v, units.Quantity):
			self[k].unit = v.unit

	def getcolumn(self, k):
		'''Return column with unit if possible'''
		v = self[k]
		if v.unit is not None:
			v = v.data*v.unit
		else:
			v = v.data
		return v

	def asdict(self):
		'''Return a copy of the table as a dictionary'''
		from collections import OrderedDict
		out = OrderedDict()
		for k in list(self.keys()):
			out[k] = self[k].data
			if self[k].unit is not None:
				out[k] *= self[k].unit
		return out

	def query(self, keycol, key, valuecol=None, logic='and', op='=='):
		index = self.index(keycol, key, logic=logic, op=op)
		if valuecol is None:
			return self[index]
		return self[valuecol][index]

	def index(self, keycol, key, logic='and', op='=='):
		if hasattr(keycol, '__iter__') and hasattr(key, '__iter__'):
			if len(keycol) != len(key):
				raise ValueError('`keycol` and `key` must have the same length')
		elif hasattr(keycol,'__iter__'):
			key = [key]*len(keycol)
		elif hasattr(key,'__iter__'):
			keycol = [keycol]*len(key)
		else:
			return self._col_index(self[keycol], key, op).nonzero()
		if hasattr(op, '__iter__'):
			if len(op) != len(keycol):
				raise ValueError('`op` must have the same length as `keycol`')
		else:
			op = [op]*len(keycol)
		if logic == 'and':
			idx = np.repeat(True, len(self))
			for c, k, o in zip(keycol, key, op):
				idx &= self._col_index(self[c], k, o)
		elif logic == 'or':
			idx = np.repeat(False, len(self))
			for c, k, o in zip(keycol, key, op):
				idx |= self._col_index(self[c], k, o)
		else:
			raise ValueError('unrecognized `logic` {0}'.format(logic))
		return idx.nonzero()

	@staticmethod
	def _col_index(c, k, op='=='):
		if op == '==':
			return (c == k)
		if op == '>':
			return (c >  k)
		if op == '<':
			return (c <  k)
		if op == '>=':
			return (c >= k)
		if op == '<=':
			return (c <= k)
		if op == 'is':
			return (c is k)
		if op == '!=':
			return (c != k)

	def show_in_browser(self, *args, **kwargs):
		kwargs['jsviewer'] = kwargs.pop('jsviewer', True)
		return super(Table, self).show_in_browser(*args, **kwargs)

	def update(self, tbl, col, sort=None):
		'''Update table with the new table based on key column.

		Parameters
		----------
		tbl : Table
		  New table
		col : str or int
		  The key column
		sort : str or list of str
		  Sort the table after update.  See Table.sort()

		Program will compare the key columns in two tables.  If a key
		exists, then the corresponding row will be updated using the
		content in the new table.  If a key does not exist, then the
		corresponding row will be inserted.  If the new table contains
		new columns, then the new columns will be added.

		v1.0.0 : JYL @PSI, 2/26/2015
		'''
		# check parameters
		if not isinstance(tbl, Table):
			raise TypeError('a Table instance is expected, {0} received'.format(type(tbl)))
		from .core import isinteger
		if isinteger(col):
			col = list(self.keys())[col]
		elif isinstance(col, str):
			pass
		else:
			raise TypeError('a string or integer is expected, {0} received'.format(type(col)))

		# search and add new columns
		from collections import OrderedDict
		nn = 0
		for k in list(tbl.keys()):
			if k not in list(self.keys()):
				self.add_column(Column(np.zeros(len(self)),dtype=tbl[k].dtype,name=k))
				nn += 1

		# go through all rows in new table
		selfkeys = list(self.keys())
		if self.mask is None:
			selfmask = np.zeros((len(self), len(list(self.keys()))),dtype=bool)
		else:
			selfmask = np.asarray(self.mask).view(bool).reshape(len(self),len(list(self.keys())))
		if nn >0 :
			selfmask[:,-nn:] = True
		if tbl.mask is None:
			tblmask = np.zeros((len(tbl), len(list(tbl.keys()))),dtype=bool)
		else:
			tblmask = np.asarray(tbl.mask).view(bool).reshape(len(tbl),len(list(tbl.keys())))
		newkeys = list(tbl.keys())
		oldkeys = list(self.keys())
		for c in tbl:
			if c[col] in self[col]:  # update the row
				idx = self.index(col, c[col])[0][0]
				for k in newkeys:
					self[idx][k] = c[k]
					selfmask[idx][oldkeys.index(k)] = tblmask[c.index][newkeys.index(k)]
			else:  # add new row
				from copy import deepcopy
				r = deepcopy(self[0])
				m = [True]*len(selfkeys)
				for k in list(tbl.keys()):
					r[k] = c[k]
					m[selfkeys.index(k)] = False
				self.add_row(r)
				selfmask = np.concatenate((selfmask, np.asarray(m)[np.newaxis,:]))

		if selfmask.any():
			if self.mask is None:
				super(Table, self).__init__([self[x] for x in list(self.keys())], masked=True)
			self.mask[:] = selfmask.T

		if sort is not None:
			self.sort(sort)


class Column(table.Column):

	def index(self, key, op='=='):
		if op == '==':
			return (self == key).nonzero()
		if op == '>':
			return (self > key).nonzero()
		if op == '<':
			return (self < key).nonzero()
		if op == '>=':
			return (self >= key).nonzero()
		if op == '<=':
			return (self <= key).nonzero()
		if op == 'is':
			return (self is key).nonzero()
		if op == '!=':
			return (self != key).nonzero()


def show_table_in_browser(table_file, ext=1, **kwargs):
	'''Display a table file in browser

	It accepts all keywords accepted by ascii.read() and
	Table.show_in_browser().  An additional keyword `ext` is to specify
	the FITS extension if input file is a FITS file.'''
	# separate show table keywards from table reading keywords
	show_kwargs = {}
	for k, d in [('css', 'table,th,td,tr,tbody {border: 1px solid black; border-collapse: collapse;}'), ('max_lines', 5000), ('jsviewer', True), ('jskwargs', {'use_local_files': True}), ('tableid', None), ('browser', 'default')]:
		show_kwargs[k] = kwargs.pop(k, d)
	# read in table and display it
	if table_file.split('.')[-1].lower() in ['fit','fits']:
		return Table(fits.open(table_file)[ext].data, **kwargs).show_in_browser(**show_kwargs)
	else:
		from .core import ascii_read
		return ascii_read(table_file, **kwargs).show_in_browser(**show_kwargs)



#---------------------------------------------------------------------
# Extended Time and related classes and functions
#---------------------------------------------------------------------

class TimeET(time.TimeFromEpoch):
	'''
	Ephemeris time used by NAIF SPICE: seconds from 2000-01-01T12:00:00 TDB
	'''

	name = 'et'
	unit = 1.0 / 86400
	epoch_val = '2000-01-01T12:00:00.0000000'
	epoch_val2 = None
	epoch_scale = 'tdb'
	epoch_format = 'isot'


class TimeYDay(time.TimeString):
	'''
	Time format YYYY-DDD HH:MM:SS.SSS
	'''

	name = 'y_day'
	subfmts = (('date_hms',
				'%Y-%j %H:%M:%S',
				'{year:d}-{yday:03d} {hour:02d}:{min:02d}:{sec:02d}'),
			   ('date_hm',
				'%Y-%j %H:%M',
				'{year:d}-{yday:03d} {hour:02d}:{min:02d}'),
			   ('date',
				'%Y-%j',
				'{year:d}:{yday:03d}'))


class TimeYDayT(time.TimeString):
	'''
	Time format YYYY-DDDTHH:MM:SS.SSS
	'''

	name = 'y_dayt'
	subfmts = (('date_hms',
				'%Y-%jT%H:%M:%S',
				'{year:d}-{yday:03d}T{hour:02d}:{min:02d}:{sec:02d}'),
			   ('date_hm',
				'%Y-%j %H:%M',
				'{year:d}-{yday:03d}T{hour:02d}:{min:02d}'),
			   ('date',
				'%Y-%j',
				'{year:d}:{yday:03d}'))


class EpochTimeFormat(TimeET):
	'''This class is to simplify the programming with general epoch time.
	User should specify the epoch by

		EpochTimeFormat.epoch_val = '2015-01-01T00:00:00'

	The corresponding format in Time class is 'epoch'.
	The default time string format is 'isot'.  If the epoch time is
	specified in another format, then the corresponding epoch_format
	must be specified:

		EpochTimeFormat.epoch_format = str

	The default epoch of this class is the same as 'et'.  See TimeET.
	'''
	name = 'epoch'


class Time(time.Time):
	'''
	Extended astropy Time class

	New formats:
	u'et': NAIF SPICE ephemeris time
	u'epoch': Epoch time from users specified epoch.  See EpochTimeFormat
	u'y_day': Time format yyyy-ddd hh:mm:ss.sss
	u'y_dayt': Time format yyyy-dddThh:mm:ss.ssss
	'''

	def __init__(self, *args, **kwargs):
		self.FORMATS['et'] = TimeET
		self.FORMATS['epoch'] = EpochTimeFormat
		self.FORMATS['y_day'] = TimeYDay
		self.FORMATS['y_dayt'] = TimeYDayT
		super(Time, self).__init__(*args, **kwargs)


def month_number(x, case=True):
	'''Convert month name/abbr to month number'''

	import calendar

	if not case:
		x = x[0].upper()+x[1:].lower()
	try:
		n = list(calendar.month_name).index(x)
	except:
		try:
			n = list(calendar.month_abbr).index(x)
		except:
			raise ValueError('Wrong month name/abbr')
	return n


def month_name(x, abbr=False):
	'''Convert month number to month name or abbr'''

	import calendar

	if abbr:
		return calendar.month_abbr[x]
	else:
		return calendar.month_name[x]


#---------------------------------------------------------------------
# astropy fitter class for MPFIT
#---------------------------------------------------------------------

import mpfit
from astropy.modeling.fitting import _FitterMeta, DEFAULT_MAXITER, DEFAULT_ACC, DEFAULT_EPS, _validate_model
import six
@six.add_metaclass(_FitterMeta)
class MPFitter(object):
	'''
	Wrapper for mpfit.

	Attributes
	----------
	fit_info : dict
		The `scipy.optimize.leastsq` result for the most recent fit (see
		notes).

	Notes
	-----
	The ``fit_info`` dictionary contains the values returned by
	`scipy.optimize.leastsq` for the most recent fit, including the values from
	the ``infodict`` dictionary it returns. See the `scipy.optimize.leastsq`
	documentation for details on the meaning of these values. Note that the
	``x`` return value is *not* included (as it is instead the parameter values
	of the returned model).

	Additionally, one additional element of ``fit_info`` is computed whenever a
	model is fit, with the key 'param_cov'. The corresponding value is the
	covariance matrix of the parameters as a 2D numpy array.  The order of the
	matrix elements matches the order of the parameters in the fitted model
	(i.e., the same order as ``model.param_names``).
	'''

	supported_constraints = ['fixed', 'tied', 'bounds']
	'''
	The constraint types supported by this fitter type.
	'''

	def __init__(self):
		self.fit_info = {'nfev': None,
						 'fvec': None,
						 'fjac': None,
						 'ipvt': None,
						 'qtf': None,
						 'message': None,
						 'ierr': None,
						 'param_jac': None,
						 'param_cov': None}
		self.code = {-18: 'a fatal execution error has occurred.  More information may be available in .fit_info[''errmsg''].',
					 -16: 'a parameter or function value has become infinite or an undefined number.  Possibly a consequence of numerical overflow in the user''s model function, which must be avoided.',
					 'res': 'error in either MYFUNCT or ITERPROC may return to terminate the fitting process (see description of MPFIT_ERROR common below).  If either MYFUNCT or ITERPROC set ERROR_CODE to a negative number, then that number is returned in STATUS.',
					 0 : 'improper input parameters.',
					 1 : 'both actual and predicted relative reductions in the sum of squares are at most FTOL.',
					 2 : 'relative error between two consecutive iterates is at most XTOL',
					 3 : 'conditions for STATUS = 1 and STATUS = 2 both hold.',
					 4 : 'the cosine of the angle between fvec and any column of the jacobian is at most GTOL in absolute value.',
					 5 : 'the maximum number of iterations has been reached',
					 6 : 'FTOL is too small. no further reduction in the sum of squares is possible.',
					 7 : 'XTOL is too small. no further improvement in the approximate solution x is possible.',
					 8 : 'GTOL is too small. fvec is orthogonal to the columns of the jacobian to machine precision.',
					 9 : 'A successful single iteration has been completed, and the user must supply another "EXTERNAL" evaluation of the function and its derivatives.  This status indicator is neither an error nor a convergence indicator.'}

		super(MPFitter, self).__init__()

	def objective_function(self, fps, model=None, y=None, weights=None, fjac=None, **args):
		'''
		Function to minimize.

		Parameters
		----------
		fps : list
		    parameters returned by the fitter
		model : astropy.modeling.FittableModel
		    model to fit
		y : measurements to be fitted
		weights : array, optional
		    weights
		fjac : optional
		**args : keyword arguments
		    x1 = None
		    x2 = None
		    ...

		'''

		if y is None:
			raise ValueError('data to be fitted must be provided in keyword `y`')
		if model is None:
			raise ValueError('model to be fitted must be provided in keyword `model`')
		model = model.copy()
		model.parameters = fps
		var = []
		for i in range(len(list(args.keys()))):
			x = args.pop('x'+str(i), None)
			if x is None:
				raise ValueError('input coordinates must be provided in keywords `x#`')
			var.append(x)
		if weights is None:
			obj =  np.ravel(model(*var) - y)
		else:
			obj = np.ravel(weights * (model(*var) - y))

		if fjac is None:
			return 0, obj
		else:
			#print fjac.shape
			pderiv = np.squeeze(model.fit_deriv(*((var,)+tuple(model.parameters))))
			#print np.asarray(pderiv).T.shape
			#fjac[:] = np.asarray(pderiv).T.flatten()
			#print np.concatenate((obj[np.newaxis,:],np.asarray(pderiv)),axis=0).T.shape
			return 0, obj, np.asarray(pderiv).T #,pderiv
			#return 0, np.concatenate((obj[np.newaxis,:],np.asarray(pderiv)),axis=0)

	def __call__(self, model, *var, **kwargs):
		'''
		Fit data to this model.

		Parameters
		----------
		model : `~astropy.modeling.FittableModel`
		   model to fit
		*var : arrays
		   input coordinates
		weights : array, optional
		   weights
		ftol : float, optional
		   a nonnegative input variable. Termination occurs when both
           the actual and predicted relative reductions in the sum of
           squares are at most FTOL (and STATUS is accordingly set to
           1 or 3).  Therefore, FTOL measures the relative error
           desired in the sum of squares.  Default: 1D-10
        gtol : float, optional
           a nonnegative input variable. Termination occurs when the
           cosine of the angle between fvec and any column of the
           jacobian is at most GTOL in absolute value (and STATUS is
           accordingly set to 4). Therefore, GTOL measures the
           orthogonality desired between the function vector and the
           columns of the jacobian.  Default: 1D-10
		maxiter : int, optional
		   maximum number of iterations
		resdamp : float, optional
		   a scalar number, indicating the cut-off value of
           residuals where "damping" will occur.  Residuals with
           magnitudes greater than this number will be replaced by
           their logarithm.  This partially mitigates the so-called
           large residual problem inherent in least-squares solvers
           (as for the test problem CURVI, http://www.maxthis.com/-
           curviex.htm).  A value of 0 indicates no damping.
           Default: 0
        xtol : float, optional
           a nonnegative input variable. Termination occurs when the
           relative error between two consecutive iterates is at most
           XTOL (and STATUS is accordingly set to 2 or 3).  Therefore,
           XTOL measures the relative error desired in the approximate
           solution.  Default: 1D-10
		verbose : bool, optional

		Returns
		-------
		model_copy : `~astropy.modeling.FittableModel`
		   a copy of the input model with parameters set by the fitter
		'''

		from .core import condition

		weights = kwargs.pop('weights', None)
		verbose = kwargs.pop('verbose', True)
		model_copy = _validate_model(model, self.supported_constraints)

		# prepare parinfo
		parinfo = []
		p0 = []
		for pname in model_copy.param_names:
			par = getattr(model_copy, pname)
			info = {}
			info['value'] = par.value
			info['fixed'] = par.fixed
			info['limits'] = [par.min, par.max]
			info['limited'] = [par.min is not None, par.max is not None]
			info['parname'] = pname
			info['step'] = 0
			info['mpside'] = 0
			info['mpmaxstep'] = 0
			info['tied'] = condition(par.tied, par.tied, '')
			info['mpprint'] = 1
			parinfo.append(info)
			p0.append(par.value)

		# prepare function coordinates
		functkw = {'model': model_copy, 'weights': weights}
		#functkw['fjac'] = getattr(model_copy, 'fit_deriv', None)
		var = list(var)
		functkw['y'] = var.pop()
		for i in range(len(var)):
			functkw['x'+str(i)] = var[i]

		autoderivative = kwargs.pop('autoderivative', condition(getattr(model_copy, 'fit_deriv', None), 0, 1))

		m = mpfit.mpfit(self.objective_function, functkw=functkw, parinfo=parinfo, quiet=not verbose, autoderivative=autoderivative, **kwargs)

		self.fit_info['ierr'] = m.status
		self.fit_info['message'] = m.errmsg
		self.fit_info['param_cov'] = m.covar
		self.fit_info['chisq'] = m.fnorm
		self.fit_info['dof'] = m.dof
		self.fit_info['perror'] = m.perror
		self.fit_info['serror'] = m.perror*np.sqrt(m.fnorm/m.dof)
		self.fit_info['niter'] = m.niter

		if verbose:
			if m.status > 0:
				print('Success with code {0}: '.format(m.status))
				print(self.code[m.status])

		model_copy.parameters = m.params
		return model_copy

	@staticmethod
	def _wrap_deriv(params, model, weights, x, y, z=None):
		"""
		Wraps the method calculating the Jacobian of the function to account
		for model constraints.

		`scipy.optimize.leastsq` expects the function derivative to have the
		above signature (parlist, (argtuple)). In order to accommodate model
		constraints, instead of using p directly, we set the parameter list in
		this function.
		"""
		if any(model.fixed.values()) or any(model.tied.values()):

			if z is None:
				full_deriv = np.array(model.fit_deriv(x, *model.parameters))
			else:
				full_deriv = np.array(model.fit_deriv(x, y, *model.parameters))

			pars = [getattr(model, name) for name in model.param_names]
			fixed = [par.fixed for par in pars]
			tied = [par.tied for par in pars]
			tied = list(np.where([par.tied is not False for par in pars],
								  True, tied))
			fix_and_tie = np.logical_or(fixed, tied)
			ind = np.logical_not(fix_and_tie)

			if not model.col_fit_deriv:
				full_deriv = np.asarray(full_deriv).T
				residues = np.asarray(full_deriv[np.nonzero(ind)])
			else:
				residues = full_deriv[np.nonzero(ind)]

			return [np.ravel(_) for _ in residues]
		else:
			if z is None:
				return model.fit_deriv(x, *params)
			else:
				return [np.ravel(_) for _ in model.fit_deriv(x, y, *params)]


#---------------------------------------------------------------------
# astropy models
#---------------------------------------------------------------------

class Sine1DModel(Fittable1DModel):
	'''
 1-D sine wave model.  Similar to astropy.modeling.functional_models.Sine1D,
 but accepts three parameters: amplitude, frequency, phase, and DC

 Model formula:
 		f(x) = A * sin(2 * pi * (f * x + x0)) + DC
 A: amplitude (peak-to-peak is 2A);
 f: frequency (number of oscillation in 1 unit);
 x0: initial phase (between 0 and 1);
 DC: average level of oscillation.

 v1.0.0 : JYL @PSI, October 9, 2014
	'''

	A = Parameter('A')
	f = Parameter('f')
	x0 = Parameter('x0')
	DC = Parameter('DC')
	linear = False

	@staticmethod
	def evaluate(x, A, f, x0, DC):
		return A*np.sin(2*np.pi*(f*x+x0))+DC

	@staticmethod
	def deriv(x, A, f, x0, DC):
		d_A = np.sin(2*np.pi*(f*x+x0))
		d_f = A*np.cos(2*np.pi*(f*x+x0))*2*np.pi*x
		d_x0 = A*np.cos(2*np.pi*f*x+x0)*2*np.pi
		if hasattr(DC, '__iter__'):
			d_DC = np.ones_like(DC)
		else:
			d_DC = 1.
		return [d_A, d_f, d_x0, d_DC]
