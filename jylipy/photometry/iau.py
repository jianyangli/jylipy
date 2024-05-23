"""Wapper for the IAU phase function models implemented in
`sbpy.photometry`.
"""


from functools import wraps
import numpy as np, astropy.units as u
from astropy.modeling import Parameter
from sbpy import photometry, calib, bib
from .hapke import DiskInt5


__all__ = ['HG', 'HG1G2', 'HG12_Pen16', 'HG12', 'HG3P', 'default_keyword',
           'fitHG']


calib.solar_fluxd.set({'V': -26.77 * u.mag})


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

    @bib.cite({'method': '1995Icar..115..369V'})
    def toHapke(self):
        """Convert IAU HG model to Hapke 5-parameter model

        Reference
        ---------
        Verbiscer, A.J., Veverka, J., 1995. Icarus 115, 369.

        """
        hh = self.H.value
        gg = self.G.value
        g = -0.291 + 0.189*gg - 0.256*gg*gg + 0.147*gg*gg*gg
        B0 = 3.11 - 9.08*gg + 10.98*gg*gg - 5.07*gg*gg*gg
        h =  0.0516 - 0.00042*gg - 0.042*gg*gg
        pv = self.geomalb.value
        apv = 0.006 + 0.65*pv
        bpv = 0.062 + 2.36*pv
        cpv = 0.045 - pv + 7.4*pv*pv - 4.9*pv*pv*pv
        dpv = 0.027 - 0.404*pv + 1.51*pv*pv
        w = apv + bpv*gg - cpv*gg*gg + dpv*gg*gg*gg
        return DiskInt5(w, g, 0., B0, h)


class HG1G2(photometry.HG1G2):

    @default_keyword(wfb='V')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HG12(photometry.HG12):

    @default_keyword(wfb='V')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HG12_Pen16(photometry.HG12_Pen16):

    @default_keyword(wfb='V')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


HG3P = HG12_Pen16  # for backward compatability


def fitHG(alpha, measure, model=0, error=None, par=None, maxiter=1000, verbose=False):
    '''Utility function to fit IAU HG models with data

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

