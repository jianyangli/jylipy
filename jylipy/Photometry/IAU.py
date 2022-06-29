'''Implementation of IAU HG Phase function model

v1.0.0 : JYL @PSI, December 22, 2013
'''


from astropy.modeling import Parameter
from astropy import units
import numpy as np, numbers
from .core import PhaseFunction
from .hapke import DiskInt5
from ..core import condition

__all__ = sorted('HG geoalb bondalb HG2Hapke fitHG HG3P HG12'.split())


class spline(object):
    '''Spline with function values at nodes and the first derivatives at
    both ends.  Outside the data grid the extrapolations are linear based
    on the first derivatives at the corresponding ends.
    '''

    def __init__(self, x, y, dy):
        x = np.asarray(x)
        y = np.asarray(y)
        dy = np.asarray(dy)
        self.x, self.y, self.dy = x, y, dy
        n = len(y)
        h = x[1:]-x[:-1]
        r = (y[1:]-y[:-1])/(x[1:]-x[:-1])
        B = np.zeros((n-2,n))
        for i in range(n-2):
            k = i+1
            B[i,i:i+3] = [h[k], 2*(h[k-1]+h[k]), h[k-1]]
        C = np.empty((n-2,1))
        for i in range(n-2):
            k = i+1
            C[i] = 3*(r[k-1]*h[k]+r[k]*h[k-1])
        C[0] = C[0]-dy[0]*B[0,0]
        C[-1] = C[-1]-dy[1]*B[-1,-1]
        B = B[:,1:n-1]
        from numpy.linalg import solve
        dys = solve(B, C)
        dys = np.array([dy[0]] + [tmp for tmp in dys.flatten()] + [dy[1]])
        A0 = y[:-1]
        A1 = dys[:-1]
        A2 = (3*r-2*dys[:-1]-dys[1:])/h
        A3 = (-2*r+dys[:-1]+dys[1:])/h**2
        self.coef = np.array([A0, A1, A2, A3]).T
        self.polys = []
        from numpy.polynomial.polynomial import Polynomial
        for c in self.coef:
            self.polys.append(Polynomial(c))
        self.polys.insert(0, Polynomial([1,self.dy[0]]))
        self.polys.append(Polynomial([self.y[-1]-self.x[-1]*self.dy[-1], self.dy[-1]]))

    def __call__(self, x):
        x = np.asarray(x)
        out = np.zeros_like(x)
        idx = x < self.x[0]
        if idx.any():
            out[idx] = self.polys[0](x[idx])
        for i in range(len(self.x)-1):
            idx = (self.x[i] <= x ) & (x < self.x[i+1])
            if idx.any():
                out[idx] = self.polys[i+1](x[idx]-self.x[i])
        idx = (x >= self.x[-1])
        if idx.any():
            out[idx] = self.polys[-1](x[idx])
        return out


class HG(PhaseFunction):
    '''
 IAU HG phase function model

 The IAU-HG phase function is defined following Bowell et al. (1989)
 in Asteroids II, pp 524-556.

 v1.0.0 : JYL @PSI, December 22, 2013
 v1.0.1 : JYL @PSI, October 09, 2014
   Removed the `param_dim` parameter for super class Initialization.
     This parameter is to be deprecated.
   Add constraints to G: 0.0<G<1.0
 v1.0.2 : JYL @PSI, October 27, 2014
   Revised for the new features in astropy 0.4.2
   Optimized the calculation for fit_deriv
 v1.0.3 : JYL @PSI, January 6, 2015
   Add class methods `GeoAlb`, `BondAlb`, `toHapke`
    '''

    H = Parameter(default=0)
    G = Parameter(default=0.12)

    @staticmethod
    def hgphi(alpha, i):
        '''Core function in IAU HG phase function model

     Parameters
     ----------
     alpha : number or numpy array of number
       Phase angle [deg]
     i : int in [1, 2]
       Choose the form of function

     Returns
     -------
     numpy array of float

     Note
     ----
     See Bowell et al. (1989), Eq. A4.
        '''

        assert i in [1,2]
        a, b, c = [3.332, 1.862], [0.631, 1.218], [0.986, 0.238]
        alpha = np.deg2rad(alpha)
        alpha_half = alpha*0.5
        sin_alpha = np.sin(alpha)
        tan_alpha_half = np.tan(alpha_half)
        w = np.exp(-90.56 * tan_alpha_half * tan_alpha_half)
        phiis = 1 - c[i-1]*sin_alpha/(0.119+1.341*sin_alpha-0.754*sin_alpha*sin_alpha)
        phiil = np.exp(-a[i-1] * tan_alpha_half**b[i-1])
        return w*phiis + (1-w)*phiil

    @staticmethod
    def evaluate(alpha, hh, gg):
        return hh-2.5*np.log10((1-gg)*HG.hgphi(alpha, 1)+gg*HG.hgphi(alpha,2))

    @staticmethod
    def fit_deriv(alpha, hh, gg):
        if hasattr(alpha,'__iter__'):
            ddh = np.ones_like(alpha)
        else:
            ddh = 1.
        phi1, phi2 = HG.hgphi(alpha,1), HG.hgphi(alpha,2)
        ddg = -1.085736205*(-phi1+phi2)/((1-gg)*phi1+gg*phi2)
        return [ddh, ddg]

    def phaseint(self, steps=5000):
        '''
     Calculate phase integral

     Parameters
     ----------
     steps : The number of steps of phase angles [deg] for numerical
         integration.

     Returns
     -------
     The phase integral

     Notes
     -----
     This program calculates phase integral numerically.

     v1.0.0 : JYL @PSI, December 22, 2013
        '''

        pha = np.linspace(0,180,steps)
        f = 10**(-0.4*HG.evaluate(pha, self.H, self.G))
        f /= f[0]
        dpha = np.empty(steps)
        dpha[:] = np.pi/(steps-1)
        dpha[[0,-1]] /= 2
        return 2*(f*np.sin(pha*np.pi/180)*dpha).sum()

    def GeoAlb(self, radi, magsun=-26.74):
        return geoalb(self, radi, magsun)

    def BondAlb(self, radi, magsun=-26.74):
        return bondalb(self, radi, magsun)

    def toHapke(self, radi, magsun=-26.74):
        return HG2Hapke(self, radi, magsun)


class HG3P(PhaseFunction):
    '''IAU 3-parameter model (Muinonen et al., 2010)

    Phase angles are in degrees'''

    H = Parameter(default=0)
    G1 = Parameter(default=0.5)
    G2 = Parameter(default=0.2)

    from scipy.interpolate import interp1d, PchipInterpolator

    phi1v = np.deg2rad([7.5, 30., 60, 90, 120, 150]),[7.5e-1, 3.3486016e-1, 1.3410560e-1, 5.1104756e-2, 2.1465687e-2, 3.6396989e-3],[-1.9098593, -9.1328612e-2]
    phi1 = spline(*phi1v)
    phi2v = np.deg2rad([7.5, 30., 60, 90, 120, 150]),[9.25e-1, 6.2884169e-1, 3.1755495e-1, 1.2716367e-1, 2.2373903e-2, 1.6505689e-4],[-5.7295780e-1, -8.6573138e-8]
    phi2 = spline(*phi2v)
    phi3v = np.deg2rad([0.0, 0.3, 1., 2., 4., 8., 12., 20., 30.]),[1., 8.3381185e-1, 5.7735424e-1, 4.2144772e-1, 2.3174230e-1, 1.0348178e-1, 6.1733473e-2, 1.6107006e-2, 0.],[-1.0630097, 0]
    phi3 = spline(*phi3v)

    @staticmethod
    def evaluate(ph, h, g1, g2):
        ph = np.deg2rad(ph)
        return h-2.5*np.log10(g1*HG3P.phi1(ph)+g2*HG3P.phi2(ph)+(1-g1-g2)*HG3P.phi3(ph))

    @staticmethod
    def fit_deriv(ph, h, g1, g2):
        if hasattr(ph, '__iter__'):
            ddh = np.ones_like(ph)
        else:
            ddh = 1.
        ph = np.deg2rad(ph)
        phi1 = HG3P.phi1(ph)
        phi2 = HG3P.phi2(ph)
        phi3 = HG3P.phi3(ph)
        dom = (g1*phi1+g2*phi2+(1-g1-g2)*phi3)
        ddg1 = -1.085736205*(phi1-phi3)/dom
        ddg2 = -1.085736205*(phi2-phi3)/dom
        return [ddh, ddg1, ddg2]

    def phaseint_num(self, steps=5000):
        '''
     Calculate phase integral numerically

     Parameters
     ----------
     steps : The number of steps of phase angles [deg] for numerical
         integration.

     Returns
     -------
     The phase integral

     Notes
     -----
     This program calculates phase integral numerically.

     v1.0.0 : JYL @PSI, December 22, 2013
        '''

        pha = np.linspace(0,150,steps)
        f = 10**(-0.4*self(pha))
        f /= f[0]
        dpha = np.empty(steps)
        dpha[:] = np.pi/(steps-1)
        dpha[[0,-1]] /= 2
        return 2*(f*np.sin(np.deg2rad(pha))*dpha).sum()

    @property
    def phaseint(self):
        '''Phase integral, q
        Based on Muinonen et al. (2010) Eq. 22'''
        return 0.009082+0.4061*self.G1+0.8092*self.G2

    @property
    def phasecoeff(self):
        '''Phase coefficient, k
        Based on Muinonen et al. (2010) Eq. 23'''
        return -(30*self.G1+9*self.G2)/(5*np.pi*float(self.G1+self.G2))

    @property
    def oeamp(self):
        '''Opposition effect amplitude, psi-1
        Based on Muinonen et al. (2010) Eq. 24)'''
        tmp = float(self.G1+self.G2)
        return (1-tmp)/tmp

    def GeoAlb(self, radi, magsun=-26.74):
        return geoalb(self, radi, magsun)

    def BondAlb(self, radi, magsun=-26.74):
        return bondalb(self, radi, magsun)


class HG12(PhaseFunction):
    '''IAU HG12 model (Muinonen et al., 2010)
    Phase angles are in degrees
    '''

    H = Parameter(default=0.)
    G12 = Parameter(default=0.5)

    @staticmethod
    def G1(G12):
        return condition(G12<0.2, 0.7527*G12+0.06164, 0.9529*G12+0.02162)

    @staticmethod
    def G2(G12):
        return condition(G12<0.2, -0.9612*G12+0.6270, -0.6125*G12+0.5572)

    @staticmethod
    def evaluate(ph, h, g):
        g1 = HG12.G1(g)
        g2 = HG12.G2(g)
        return HG3P.evaluate(ph, h, g1, g2)

    @staticmethod
    def fit_deriv(ph, h, g):
        if hasattr(ph, '__iter__'):
            ddh = np.ones_like(ph)
        else:
            ddh = 1.
        g1 = HG12.G1(g)
        g2 = HG12.G2(g)
        ph = np.deg2rad(ph)
        phi1 = HG3P.phi1(ph)
        phi2 = HG3P.phi2(ph)
        phi3 = HG3P.phi3(ph)
        dom = (g1*phi1+g2*phi2+(1-g1-g2)*phi3)
        ddg = -1.085736205*((phi1-phi3)*condition(g<0.2,0.7527,0.9529)+(phi2-phi3)*condition(g<0.2,-0.9612,-0.6125))/dom
        return [ddh, ddg]

    def phaseint_num(self, steps=5000):
        '''
     Calculate phase integral numerically

     Parameters
     ----------
     steps : The number of steps of phase angles [deg] for numerical
         integration.

     Returns
     -------
     The phase integral

     Notes
     -----
     This program calculates phase integral numerically.

     v1.0.0 : JYL @PSI, December 22, 2013
        '''

        pha = np.linspace(0,150,steps)
        f = 10**(-0.4*self(pha))
        f /= f[0]
        dpha = np.empty(steps)
        dpha[:] = np.pi/(steps-1)
        dpha[[0,-1]] /= 2
        return 2*(f*np.sin(np.deg2rad(pha))*dpha).sum()

    @property
    def phaseint(self):
        '''Phase integral, q
        Based on Muinonen et al. (2010) Eq. 22'''
        return 0.009082+0.4061*self.G1(self.G12)+0.8092*self.G2(self.G12)

    @property
    def phasecoeff(self):
        '''Phase coefficient, k
        Based on Muinonen et al. (2010) Eq. 23'''
        G1 = self.G1(self.G12)
        G2 = self.G2(self.G12)
        return -(30*G1+9*G2)/(5*np.pi*(G1+G2))

    @property
    def oeamp(self):
        '''Opposition effect amplitude, psi-1
        Based on Muinonen et al. (2010) Eq. 24)'''
        tmp = self.G1(self.G12)+self.G2(self.G12)
        return (1-tmp)/tmp

    def GeoAlb(self, radi, magsun=-26.74):
        return geoalb(self, radi, magsun)

    def BondAlb(self, radi, magsun=-26.74):
        return bondalb(self, radi, magsun)


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

    if isinstance(radi, units.Quantity):
        if radi.unit != units.km:
            radi = radi.to(units.km).value
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

    if isinstance(radi, units.Quantity):
        if radi.unit != units.km:
            radi = radi.to(units.km).value
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

