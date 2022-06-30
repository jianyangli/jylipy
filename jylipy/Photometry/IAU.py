'''Implementation of IAU HG Phase function model

v1.0.0 : JYL @PSI, December 22, 2013
'''


from astropy.modeling import Parameter
import astropy.units as u
import numpy as np, numbers
from .core import PhaseFunction
from .hapke import DiskInt5
from ..core import condition
from sbpy import photometry
from sbpy.calib import solar_fluxd
from functools import wraps


__all__ = ['HG', 'HG1G2', 'HG12_Pen16', 'HG12', 'HG3P']


solar_fluxd.set({'V': -26.77 * u.mag})


def default_keyword(**default):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            for k, v in default.items():
                v_default = kwargs.pop(k, v)
                kwargs[k] = v_default
            return f(*args, **kwargs)
        return wrapper
    return decorator


class HG(photometry.HG):

    @default_keyword(wfb='V')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def toHapke(self, radi, magsun=-26.74):
        return HG2Hapke(self, radi, magsun)


class HG1G2(photometry.HG1G2):

    @default_keyword(wfb='V')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HG12_Pen16(photometry.HG12_Pen16):

    @default_keyword(wfb='V')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


HG12 = HG12_Pen16


HG3P = HG12  # for backward compatability


def geoalb(model, radi, magsun=-26.74):
    '''Calculate the geometric albedo for IAU HG model.

 Parameters
 ----------
 model : HG, Parameter, or a number
   The IAU HG model, or the H model parameter
 radi : number or astropy Quantity
   The radius of object.  If number, then the unit is km
 magsun : number, or astropy Quantity, optional
   Magnitude of the Sun

 Returns
 -------
 Number, the geometric albedo of an object

 v1.0.0 : JYL @PSI, December 22, 2013
 v1.0.1 : JYL @PSI, January 6, 2015
   Add type check for input parameters
    '''

    if isinstance(model, (HG, HG3P, HG12)):
        hh = model.H.value
    elif isinstance(model, Parameter):
        hh = model.value
    elif isinstance(model, numbers.Number):
        hh = model
    else:
        raise TypeError('an instance of HG, Parameter, or a number is expected, got a %s' % type(model))

    if isinstance(radi, u.Quantity):
        if radi.unit != u.km:
            radi = radi.to(u.km).value
        else:
            radi = radi.value
    elif isinstance(radi, numbers.Number):
        pass
    else:
        raise TypeError('an astropy quantity or a number is expected, got a %s' % type(radi))

    from ..core import mag2alb
    return mag2alb(hh, radi, magsun=magsun)


def bondalb(model, radi, magsun=-26.74):
    '''Calculate Bond albedo

 Parameters
 ----------
 See geoalb.

 Returns
 -------
 Number, Bond albedo of an object

 v1.0.0 : JYL @PSI, December 22, 2013
    '''
    return geoalb(model, radi, magsun)*model.phaseint()


def HG2Hapke(model, radi, magsun=-26.74):
    '''Convert IAU HG model to Hapke model parameters

 Parameters
 ----------
 model : HG class, or a tuple with two Parameter class or numbers
   The IAU HG model or the (H, G) parameters
 radi : number or astropy Quantity
   The radius of object [km]
 magsun : number, or astropy Quantity, optional
   Magnitude of the Sun

 Returns
 -------
 Hapke.DiskInt5 instance

 Notes
 -----
 The procedure follows Verbiscer and Veverka (1995).

 v1.0.0 : JYL @PSI, December 22, 2013
 v1.0.1 : JYL @PSI, January 6, 2015
   Add type check for input parameters
   Change return from a Hapke parameter dictionary to a Hapke disk
   integrated phase function model `DiskInt5` instance.
    '''

    if isinstance(model, HG):
        hh, gg = model.H.value, model.G.value
    elif hasattr(model,'__iter__'):
        hh, gg = model
        if isinstance(hh, Parameter):
            hh = hh.value
        elif isinstance(hh, numbers.Number):
            pass
        else:
            raise TypeError('an astropy quantity or a number is expected, got a %s' % type(hh))
        if isinstance(gg, Parameter):
            gg = gg.value
        elif isinstance(gg, numbers.Number):
            pass
        else:
            raise TypeError('an astropy quantity or a number is expected, got a %s' % type(gg))
    else:
        raise TypeError('a HG or a tuple of two Parameters or numbers is expected, got a %s' % type(model))

    if isinstance(radi, u.Quantity):
        if radi.unit != u.km:
            radi = radi.to(u.km).value
        else:
            radi = radi.value
    elif isinstance(radi, numbers.Number):
        pass
    else:
        raise TypeError('an astropy quantity or a number is expected, got a %s' % type(radi))

    g = -0.291 + 0.189*gg - 0.256*gg*gg + 0.147*gg*gg*gg
    B0 = 3.11 - 9.08*gg + 10.98*gg*gg - 5.07*gg*gg*gg
    h =  0.0516 - 0.00042*gg - 0.042*gg*gg
    pv = geoalb(hh, radi, magsun)
    apv = 0.006 + 0.65*pv
    bpv = 0.062 + 2.36*pv
    cpv = 0.045 - pv + 7.4*pv*pv - 4.9*pv*pv*pv
    dpv = 0.027 - 0.404*pv + 1.51*pv*pv
    w = apv + bpv*gg - cpv*gg*gg + dpv*gg*gg*gg

    return DiskInt5(w, g, 0., B0, h)


def fitHG(alpha, measure, model=0, error=None, par=None, maxiter=1000, verbose=False):
    '''Fit IAU HG models with data

 Parameters
 ----------
 alpha, measure : array-like, number
   Phase angle [deg] and measured phase function
 model : number or string, optional
   The model to be fitted.  Value must be in
   {0, 'HG', 'hg', 1, 'HG3P', 'hg3p', 2, 'HG12', 'hg12'}
 error : array-like, number, optional
   Measurement error
 par : tuple of numbers, optional
   Initial parameters
 maxiter : number, optional
   Maximum number of iteration
 verbose : Bool, optional

 Returns
 -------
 (Model, uncertainties, fit_info):
   Model: The best fit HG class.
   uncertaintes: Uncertianties of best-fit model parameters.
   fit_info: The fit_info dictionary from fitter class.  Note that
     it contains an additional element 'red_chisq' that stores the
     reduced chisq of the fit.

 v1.0.0 : JYL @PSI, December 22, 2013
 v1.0.0 : JYL @PSI, October 27, 2014
   Revised the calculation of covarance using the fit_info['param_cov']
     in the fitter.
   Changed function return
   Deprecated keyword `covar`
   Added keyword `maxiter`
    '''

    #from astropy.modeling.fitting import LevMarLSQFitter
    from ..core import MPFitter

    alpha = np.asarray(alpha).flatten()
    measure = np.asarray(measure).flatten()
    if error is not None:
        error = np.asarray(error).flatten()
        weights = 1/error
    else:
        weights = None
    if alpha.shape[0] != measure.shape[0]:
        raise RuntimeError('Input parameters must have the same number of elements')
    if error is not None and error.shape[0] != alpha.shape[0]:
        raise RuntimeError('Error array must have the same number of element as data')

    if par is None:
        h = measure.min()
        if model in [0, 'HG', 'hg']:
            par = (h, 0.2)
        elif model in [1, 'HG3P', 'hg3p']:
            par = (h, 0.4, 0.4)
        elif model in [2, 'HG12', 'hg12']:
            par = (h, 0.5)
        else:
            raise ValueError("`model` must be [0, 'HG', 'hg', 1, 'HG3P', 'hg3p', 2, 'HG12', 'hg12'], {0} received".format(model))

    if model in [0, 'HG', 'hg']:
        m0 = HG(*par)
    elif model in [1, 'HG3P', 'hg3p']:
        m0 = HG3P(*par)
    elif model in [2, 'HG12', 'hg12']:
        m0 = HG12(*par)

    f = MPFitter()  #LevMarLSQFitter()
    m = f(m0, alpha, measure, weights=weights, maxiter=maxiter, verbose=verbose)
    fit_info = f.fit_info.copy()

    if fit_info['ierr'] > 0:
        if verbose:
            print('Fit successful.\n'+fit_info['message'])
        if weights is None:
            fit_info['red_chisq'] = np.sum((m(alpha)-measure)**2)/(len(alpha)-2)
        else:
            fit_info['red_chisq'] = np.sum(((m(alpha)-measure)*weights)**2)/(len(alpha)-2)
        return m, np.sqrt(np.diag(fit_info['param_cov'])), fit_info
    else:
        if verbose:
            print('Fit may not be successful!\n'+fit_info['message'])
        return m, None, fit_info

