'''
Disk-resolved photometric models and fitting.

Packages used:
numpy, matplotlib.pyplot, mpfit, JYLPy.utilities

Written by Jian-Yang Li (Planetary Science Institute)
'''

from ..core import *
from .core import PhotometricModel, HG1
from ..plotting import *
from .. import mesh
import numpy as np, copy
from astropy.modeling import Fittable1DModel, Fittable2DModel, FittableModel, Parameter


'''Hapke model suite

* All angles are in radiance by default unless otherwise specified.
* The goal is to support astropy quantity for input, but this is not
  fully tested
'''


def r0(w,deriv=False):
    '''
    Spherical albedo r0 = (1-gamma)/(1+gamma), where gamma = sqrt(1-w)

    v1.0.0 : JYL @PSI, October 30, 2014
    '''
    gamma = np.sqrt(1-w)
    if not deriv:
        return (1-gamma)/(1+gamma)
    else:
        return 1/((1+gamma)*(1+gamma)*gamma)


class HapkeK(Fittable1DModel):
    '''
    Hapke roughness correction K for disk-integrated phase function
    Based on Table 12.1, Hapke (2012) book.

    v1.0.0 : JYL @PSI, October 29, 2014
    '''

    theta = Parameter(default=20., description='Roughness', bounds=(0., 60.))

    t = np.deg2rad(np.array([0.,10,20,30,40,50,60]))
    a = np.deg2rad(np.array([0.,2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]))

    # Table 12.1, Hapke (2012) book
    K = np.array([[1., 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                  [1., .997, .991, .984, .974, .961, .943],
                  [1., .994, .981, .965, .944, .918, .881],
                  [1., .991, .970, .943, .909, .866, .809],
                  [1., .988, .957, .914, .861, .797, .715],
                  [1., .986, .947, .892, .825, .744, .644],
                  [1., .984, .938, .871, .789, .692, .577],
                  [1., .982, .926, .846, .748, .635, .509],
                  [1., .979, .911, .814, .698, .570, .438],
                  [1., .974, .891, .772, .637, .499, .366],
                  [1., .968, .864, .719, .566, .423, .296],
                  [1., .959, .827, .654, .487, .346, .231],
                  [1., .946, .777, .575, .403, .273, .175],
                  [1., .926, .708, .484, .320, .208, .130],
                  [1., .894, .617, .386, .243, .153, .094],
                  [1., .840, .503, .290, .175, .107, .064],
                  [1., .747, .374, .201, .117, .070, .041],
                  [1., .590, .244, .123, .069, .040, .023],
                  [1., .366, .127, .060, .032, .018, .010],
                  [1., .128, .037, .016, .0085, .0047, .0026],
                  [1., .0,   .0,   .0,   .0,   .0,   .0]])
    from scipy.interpolate import RectBivariateSpline
    Kmodel = RectBivariateSpline(a, t, K)

    # Numerically derived d_K/d_theta
    dK = np.array([[  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
                   [  0.000, -0.029, -0.037, -0.048, -0.065, -0.086, -0.122],
                   [  0.000, -0.060, -0.083, -0.105, -0.132, -0.173, -0.256],
                   [  0.000, -0.093, -0.140, -0.173, -0.217, -0.281, -0.376],
                   [  0.000, -0.131, -0.216, -0.276, -0.331, -0.410, -0.535],
                   [  0.000, -0.161, -0.276, -0.350, -0.420, -0.513, -0.635],
                   [  0.000, -0.187, -0.331, -0.430, -0.511, -0.604, -0.715],
                   [  0.000, -0.227, -0.399, -0.513, -0.607, -0.686, -0.754],
                   [  0.000, -0.274, -0.486, -0.617, -0.706, -0.753, -0.752],
                   [  0.000, -0.332, -0.599, -0.744, -0.791, -0.783, -0.735],
                   [  0.000, -0.418, -0.746, -0.880, -0.858, -0.777, -0.677],
                   [  0.000, -0.543, -0.923, -1.008, -0.889, -0.730, -0.594],
                   [  0.000, -0.719, -1.138, -1.108, -0.861, -0.641, -0.496],
                   [  0.000, -0.982, -1.371, -1.132, -0.771, -0.528, -0.383],
                   [  0.000, -1.344, -1.584, -1.052, -0.635, -0.411, -0.282],
                   [  0.000, -1.793, -1.700, -0.860, -0.496, -0.301, -0.210],
                   [ -0.267, -2.213, -1.642, -0.603, -0.363, -0.197, -0.156],
                   [ -2.047, -2.409, -1.314, -0.364, -0.237, -0.113, -0.099],
                   [ -5.222, -2.272, -0.696, -0.205, -0.116, -0.054, -0.046],
                   [ -8.920, -1.916, -0.500, -0.127, -0.011, -0.025,  0.000],
                   [-11.016, -0.841,  0.000,  0.000,  0.000,  0.000,  0.000]])
    dKmodel = RectBivariateSpline(a, t, dK)

    @staticmethod
    def evaluate(alpha, theta):
        if hasattr(alpha,'__iter__') and (len(alpha) >1):
            k = np.zeros_like(alpha)
            st = alpha.argsort()
            k[st] = HapkeK.Kmodel(alpha[st], theta, grid=False)
        else:
            k = HapkeK.Kmodel(alpha, theta)
        k = np.clip(k, 0., 1.)
        return k

    @staticmethod
    def fit_deriv(alpha, theta):
        if hasattr(alpha, '__iter__') and (len(alpha) > 1):
            dk = np.zeros_like(alpha)
            st = alpha.argsort()
            dk[st] = HapkeK.dKmodel(alpha[st], theta, grid=False)
        else:
            dk = HapkeK.dKmodel(alpha, theta)
        #k = np.clip(k, 0.)
        return dk


class HapkeU(Fittable1DModel):
    '''
    Hapke roughness correction U for disk-integrated phase function
    Based on Eq. 12.60 in Hapke (2012) book.

    The unit of roughness parameter, theta, is degrees.

    v1.0.0 : JYL @PSI, October 29, 2014
    '''

    w = Parameter(default=0.1, description='Single Scattering Albedo', bounds=(0., 1.))
    theta = Parameter(default=20., description='Roughness', bounds=(0., 60.))

    @staticmethod
    def evaluate(x, w, theta):
        r = r0(w)
        return 1-theta*r*(0.048+0.0041*theta+(0.33-0.0049*theta)*r)

    @staticmethod
    def fit_deriv(x, w, theta):
        r = r0(w)
        dr = r0(w,deriv=True)
        d_theta = -r*(0.048+0.0082*theta+r*(0.33-0.0098*theta))
        d_w = -theta*((0.048+0.0041*theta)+(0.66-0.0098*theta)*r)*dr
        return [d_w, d_theta]


class SHOE_Base(Fittable1DModel):
    '''Base class for SHOE model'''

    Bs0 = Parameter(default=1.0, description='Amplitude of shadow-hiding opposition effect', min=0.)
    hs = Parameter(default=0.01, description='Width of shadow-hiding opposition effect', min=1e-10)

    @property
    def hwhm(self):
        return np.rad2deg(self.hs.value)*2


class SHOE(SHOE_Base):
    '''
    Shadow-hiding opposition effect model, the exact expression.
    See Eqs. 9.21 in Hapke (2012) book

    SHOE = 1+Bs0*Bs(alpha, hs)

    v1.0.0 : JYL @PSI, October 30, 2014
    v1.0.1 : JYL @PSI, January 6, 2015
    Separate the approximate solution to another class SHOE_approx
    '''

    @staticmethod
    def Bs(alpha, hs):
        '''Eq. 9.21b in Hapke (2012) book'''
        from scipy.special import erf
        bs = np.ones_like(alpha)
        y = np.tan(alpha/2)/hs
        z1 = y >= 0.04  # can be calculated directly
        z2 = ~z1  # has to be calculated with Tylor expansion
        yr = np.ones_like(y)
        z3 = y != 0
        yr[z3] = 1/y[z3]
        yr[~z3] = np.inf
        yrsqrt = np.sqrt(yr)
        if z1.any():
            bs[z1] = 2*np.sqrt(np.pi)*yrsqrt[z1]*np.exp(yr[z1])*(erf(2*yrsqrt[z1])-erf(yrsqrt[z1]))+np.exp(-3*yr[z1])-1
        if z2.any():
            y = y[z2]
            y2 = y*y
            y3 = y*y2
            y4 = y2*y2
            bs[z2] = 1-y+1.5*y2-3.75*y3+6.5625*y4+(0.125*y-4.6875e-2*y2+5.5859375e-2*y3-2.5634765625e-2*y4)*np.exp(-3*yr[z2])
        return bs

    @staticmethod
    def evaluate(alpha, Bs0, hs):
        return 1+Bs0*SHOE.Bs(alpha, hs)

    @staticmethod
    def fit_deriv(alpha, Bs0, hs):
        dB0 = SHOE.Bs(alpha, hs)
        from scipy.special import erf
        dh = np.ones_like(alpha)
        y = np.tan(alpha/2)/hs
        z1 = y >= 0.04  # can be calculated directly
        z2 = ~z1  # has to be calculated with Tylor expansion
        yr = np.ones_like(y)
        z3 = y != 0
        yr[z3] = 1/y[z3]
        yr[~z3] = np.inf
        yrsqrt = np.sqrt(yr)
        if z1.any():
            dh[z1] = Bs0*(np.sqrt(np.pi)*np.exp(yr[z1])*(1+2*yr[z1])*(erf(2*yrsqrt[z1])-erf(yrsqrt[z1]))+yrsqrt[z1]*(np.exp(-3.*yr[z1])-2))*yrsqrt[z1]/hs
        if z2.any():
            y = y[z2]
            y2 = y*y
            dh[z2] = Bs0*(y-3.*y2+np.exp(-3.*yr[z2])*(-0.375+1.5625e-2*y+5.859375e-3*y2))/hs
        return [dB0, dh]


class SHOE_approx(SHOE_Base):
    '''
    Shadow-hiding opposition effect model, the approximate solution
    See Eq. 9.22 in Hapke (2012) book

    SHOE = 1+B0*Bs(alpha, h)

    v1.0.0 : JYL @PSI, January 6, 2015
    '''

    @staticmethod
    def Bs(alpha, hs):
        '''Eq. 9.22, Hapke (2012) book'''
        return 1/(1+np.tan(alpha/2)/hs)

    @staticmethod
    def evaluate(alpha, Bs0, hs):
        return 1+Bs0*SHOE_approx.Bs(alpha, hs)

    @staticmethod
    def fit_deriv(alpha, Bs0, hs):
        dB0 = SHOE_approx.Bs(alpha, hs)
        a = np.tan(alpha/2)
        dh = Bs0*a/(a+hs)**2
        return [dB0, dh]


class CBOE_Base(Fittable1DModel):
    '''Base class for CBOE model'''

    Bc0 = Parameter(default=1.0, description='Amplitude of coherent backscatter opposition effect', min=0.)
    hc = Parameter(default=0.01, description='Width of coherent backscatter opposition effect', min=1e-10)

    @property
    def hwhm(self):
        return np.rad2deg(self.hc.value)*0.66


class CBOE(CBOE_Base):
    '''
    Coherent backscatter opposition effect model
    See Eq. 33 in Hapke (2002) Icarus 157, 523-534.
    '''

    @staticmethod
    def Bc(alpha, hc):
        '''Eq. 9.21b in Hapke (2012) book'''
        y = np.tan(alpha/2)/hc
        z1 = y != 0
        z2 = ~z1
        bc = np.zeros_like(y)
        bc[z1] = 0.5*(1.+(1.-np.exp(-y[z1]))/y[z1])/(1+y[z1])**2
        bc[z2] = 1.
        return bc

    @staticmethod
    def evaluate(alpha, Bc0, hc):
        return 1+Bc0*CBOE.Bc(alpha, hc)


class H(Fittable1DModel):
    '''Chandrasekhar's H function

    Based on the approximate formulae in Hapke (1981) or Hapke (2002,
    Icarus 157, 523)).  The form of formula is chosen by keyword
    'version', where the default is '02'.
    '''

    w = Parameter(default=0.1, description='Single Scattering Albedo', bounds=(0., 1.))

    @staticmethod
    def evaluate(x, w, version='02'):
        x = np.asarray(x)
        assert (x>=0.).all() & (x<=1.).all()
        if version == '02':
            r = r0(w)
            hh = np.zeros_like(x)
            z1 = x!=0
            z2 = ~z1
            hh[z1] = 1./(1-w*x[z1]*(r+(1-2*r*x[z1])*np.log((1+x[z1])/x[z1])/2))
            hh[z2] = 1.
            return hh
        elif version == '81':
            return (1+2*x)/(1+2*x*np.sqrt(1-w))


def hg(alpha, b, deriv=False):
    '''One-parameter forward-scattering Henyey-Greenstein function.
            -1 < b < 0: Backscattering
            b == 0: Isotropic
            0 < b < 1.: Forward scattering

    If `deriv' is set True, then return the partial derivative w/r to parameter

    Note:
        If restrict 0<b<1, then this function only represents the forward
        scatteirng case.  The backscattering case with 0<b<1 can be represented
        as hg(alpha, -b), and the derivative is -hg(alpha, -b, deriv=True).
    '''
    if b==0:
        hg = np.ones_like(alpha)
    else:
        cosa = np.cos(alpha)
        b2 = b*b
        dom_base = (1+2*b*cosa+b2)
        dom = dom_base**1.5
        hg = (1-b2)/dom
    if not deriv:
        return hg
    else:
        if b==0:
            dhg = np.zeros_like(alpha)
        else:
            dhg = -2*b/dom - 3*(cosa+b)*hg/dom_base
        return dhg


class HenyeyGreenstein1P(Fittable1DModel):
    '''One-parameter Henyey-Greenstein model

    The Henyey-Greenstein function can take either a single-term form or
    double-term form, depending on the parameters passed.
    The single-term HG function has the form as in Hapke (2012) book, Eq. 6.5:
                                 (1-g**2)
        HG_1(alpha) = -----------------------------
                       (1+2*g*cos(alpha)+g**2) **1.5
    where -1<g<1.  g=0 means isotripic scattering, g>0 forward scattering,
    and g<0 backward scattering.
    '''

    g = Parameter(default=-0.3, description='Henyey-Greenstein Asymmetry Factor', bounds=(-1.,1.))

    @staticmethod
    def evaluate(x, g):
        return hg(x, g)

    @staticmethod
    def fit_deriv(x, g):
        return hg(x, g, deriv=True)


class HenyeyGreenstein2P(Fittable1DModel):
    '''Two-parameter Henyey-Greenstein model

    The two-parameter HG function has the form as in Hapke (2012), Eq. 6.7a:
                                    (1-b**2)
        HG_b(alpha; b) = -----------------------------
                          (1-2*b*cos(alpha)+b**2) **1.5
                                    (1-b**2)
        HG_f(alpha; b) = -----------------------------
                          (1+2*b*cos(alpha)+b**2) **1.5
        HG_2(alpha) = (1+c)/2 * HG_b(alpha; b) + (1-c)/2 * HG_f(alpha; b)

    The HG_b describes backward lobe and the HG_f describes forward lobe.
    '''

    b = Parameter(default=0.2, description='Henyey-Greenstein Asymmetry Factor', bounds=(0.,1.))
    c = Parameter(default=0., description='Henyey-Greenstein Forward-backward Partition Parameter', bounds=(-1.,1))

    @staticmethod
    def evaluate(x, b, c):
        if c==-1.:
            return hg(x, b)
        elif c==1.:
            return hg(x, -b)
        hgf = hg(x, b)
        hgb = hg(x, -b)
        return 0.5 * ((1-c)*hgf + (1+c)*hgb)

    @staticmethod
    def fit_deriv(x, b, c):
        hgf = hg(x, b)
        hgb = hg(x, -b)
        ddc = (hgb-hgf)*0.5
        if c==-1:
            ddb = hg(x, b, deriv=True)
        elif c==1:
            ddb = -hg(x, -b, deriv=True)
        else:
            ddb = 0.5*((1-c)*hg(x, b, deriv=True)+(1+c)*-hg(x, -b, deriv=True))
        return [ddb, ddc]


class HenyeyGreenstein3P(Fittable1DModel):
    '''Three-parameter Henyey-Greenstein model

    The three-parameter HG function has the form as in Hapke (2012), Eq.
    6.7b:
        HG_3(alpha) = (1+c)/2 * HG_b(alpha; b1) + (1-c)/2 * HG_f(alpha; b2)
    The range of values of parameters for two-parameter and three-parameter
    HG functions are: 0<=b, b1, b2<=1, and -1<c<1.
    '''

    b1 = Parameter(default=0.2, description='Henyey-Greenstein forward scattering parameter', bounds=(0.,1.))
    b2 = Parameter(default=0.4, description='Henyey-Greenstein backward scattering parameter', bounds=(0.,1.))
    c = Parameter(default=0., description='Forward-backward partition parameter', bounds=(-1.,1.))

    @staticmethod
    def evaluate(x, b1, b2, c):
        if c==-1.:
            return hg(x, b1)
        elif c==1.:
            return hg(x, -b2)
        hgf = hg(x, b1)
        hgb = hg(x, -b2)
        return 0.5 * ((1-c)*hgf + (1+c)*hgb)

    @staticmethod
    def fit_deriv(x, b1, b2, c):
        hgf = hg(x, b1)
        hgb = hg(x, -b2)
        ddc = (hgb-hgf)*0.5
        if c==-1:
            ddb1 = hg(x, b1, deriv=True)
            ddb2 = np.zeros_like(x)
        elif c==1:
            ddb1 = np.zeros_like(x)
            ddb2 = -hg(x, -b2, deriv=True)
        else:
            ddb1 = 0.5*(1-c)*hg(x, b1, deriv=True)
            ddb2 = 0.5*(1+c)*-hg(x, -b2, deriv=True)
        return [ddb1, ddb2, ddc]


def calc_psi(i, e, alpha, cos=False):
    '''
 Calculate the angle between incidence and emission planes.

 Parameters
 ----------
 i : array-like float
   Incidence angle in degrees or cosines
 e : array-like float
   Emission angle in degrees or cosines
 alpha : array-like float
   Phase angle in degrees
 cos : bool, optional
   If True, then the input i and e are the cosines of incidence and
   emission angles.  Default is False.

 Returns
 -------
 A numpy array of the angles between incidence and emission planes.
 The unit is degrees.

 Notes
 -----
 If the incidence and emission planes are nearly co-plane by ~3e-6 deg,
 then the results may be round-off to 0 or 180 deg.
    '''

    i_cp, e_cp, ph_cp = nparr(i, e, alpha)

    if cos:
        mu0, mu = i_cp, e_cp
        i_cp, e_cp = np.rad2deg(np.arccos(mu0)), np.rad2deg(np.arccos(mu))
    else:
        mu0, mu = np.cos(np.deg2rad(i_cp)), np.cos(np.deg2rad(e_cp))
    si, se = np.sqrt(1-mu0*mu0), np.sqrt(1-mu*mu)

    sise = si*se
    zeros = sise == 0
    sise[zeros] = 1.

    # cos(psi)
    cospsi = (np.cos(ph_cp*np.pi/180)-mu0*mu)/sise

    # When cos(psi) is too close to 1 or -1, set it to 1 or -1
    ww = np.abs(1-np.abs(cospsi)) < 1e-4
    cospsi[ww] = np.sign(cospsi[ww])
    psi = np.rad2deg(np.arccos(cospsi))
    psi[zeros] = 0.  # if i or e is 0, set the angle to be 0

    # When i or e is 0, i+e must be equal to phase.  Otherwise set it to nan.
    psi[(si*se == 0) * (np.abs(i_cp+e_cp-ph_cp) > 1e-10)] = np.nan

    return psi


def calc_pha(inc, emi, psi, cos=False):
    '''
 Calculate phase angle from incidence, emission angles and the angle
 between incidence and emission planes.

 Parameters
 ----------
 i : array-like float
   Incidence angle in degrees or cosines
 e : array-like float
   Emission angle in degrees or cosines
 psi : array-like float
   Angle between incidence and emission planes, in degrees.
 cos : bool, optional
   If True, then the input i and e are the cosines of incidence and
   emission angles.  Default is False.

 Returns
 -------
 A numpy array containing the phase angles in degrees.
    '''

    icp, ecp, psicp = nparr(inc, emi, psi)

    if cos:
        mu0, mu = icp, ecp
    else:
        mu0, mu = cos(icp*np.pi/180), cos(ecp*np.pi/180)

    si, se = np.sqrt(1-mu0*mu0), np.sqrt(1-mu*mu)

    psicp *= np.pi/180

    return np.arccos(mu0*mu+si*se*np.cos(psi*np.pi/180))*180/np.pi


def hfunc(w, x, version='02'):
    '''
 Calculates Chandrasekhar's H function.

 Based on the approximate formulae in Hapke (1981) or Hapke (2002,
 Icarus 157, 523)).  The form of formula is chosen by keyword
 'version', where the default is '02'.

 Input can take scalors or lists or numpy arrays.  If both parameters
 are lists or numpy arrays, then they have to have the same length
    '''

    # wcp, xcp = nparr(w, x)
    wcp = np.asarray(w)
    xcp = np.asarray(x)

    if version == '02':
        gamma = np.sqrt(1-wcp)
        r0 = (1-gamma)/(1+gamma)
        xcp[xcp == 0] = np.NaN
        hh = 1./(1-wcp*xcp*(r0+(1-2*r0*xcp)*np.log((1+xcp)/xcp)/2))
        hh[np.isnan(hh)] = 1.
        return hh
    elif version == '81':
        return (1+2*xcp)/(1+2*xcp*np.sqrt(1-wcp))


def hg_func(alpha, *g):
    '''
 Calculates the Henyey-Greenstein single-particle phase function.

 Usage::
 sppf = hg_func(alpha, g)
 sppf = hg_func(alpha, b, c)
 sppf = hg_func(alpha, b1, b2, c)

 Parameters
 ----------
 alpha : array-like float
   Phase angle in degrees
 g : float
   Parameter of single-term HG function
 b, c : float
   Parameters of double-parameter HG function
 b1, b2, c : float
   Parameter of three-parameter HG function

 Returns
 -------
 Program returns a numpy array containing the HG phase function for the
 corresonding input parameters.

 Notes
 -----
 The Henyey-Greenstein function can take either a single-term form or
 double-term form, depending on the parameters passed.
 The single-term HG function has the form as in Hapke (2012), Eq. 6.5:
                             (1-g**2)
     HG_1(alpha) = -----------------------------
                   (1+2*g*cos(alpha)+g**2) **1.5
 where -1<g<1.  g=0 means isotripic scattering, g>0 forward scattering,
 and g<0 backward scattering.
 The two-parameter HG function has the form as in Hapke (2012), Eq. 6.7a:
                                (1-b**2)
     HG_b(alpha; b) = -----------------------------
                      (1-2*b*cos(alpha)+b**2) **1.5
                                (1-b**2)
     HG_f(alpha; b) = -----------------------------
                      (1+2*b*cos(alpha)+b**2) **1.5
     HG_2(alpha) = (1+c)/2 * HG_b(alpha; b) + (1-c)/2 * HG_f(alpha; b)

 The HG_b describes backward lobe and the HG_f describes forward lobe.
 The three-parameter HG function has the form as in Hapke (2012), Eq.
 6.7b:
     HG_3(alpha) = (1+c)/2 * HG_b(alpha; b1) + (1-c)/2 * HG_f(alpha; b2)
 The range of values of parameters for two-parameter and three-parameter
 HG functions are: 0<=b, b1, b2<=1, and no constraints for c except that
 phase function has to be non-negative everywhere.
    '''

    alpha_cp, gcp = nparr(alpha, g)
    cosa = np.cos(alpha_cp*np.pi/180)

    if len(gcp) == 3:
        b1, b2, c = gcp
        hg_b = (1-b1*b1)/(1-2*b1*cosa+b1*b1)**1.5
        hg_f = (1-b2*b2)/(1+2*b2*cosa+b2*b2)**1.5
        return (1+c)/2*hg_b + (1-c)/2*hg_f
    elif len(gcp) == 2:
        b, c = gcp
        hg_b = (1-b*b)/(1-2*b*cosa+b*b)**1.5
        hg_f = (1-b*b)/(1+2*b*cosa+b*b)**1.5
        return (1+c)/2*hg_b + (1-c)/2*hg_f
    else:
        g1 = gcp[0]
        return (1-g1*g1)/(1+2*g1*cosa+g1*g1)**1.5


def poly_phase(alpha, *par):
    '''
 Calculates the Legendre polynomial single-particle phase function.

 Usage::
 return = poly_phase(alpha, par...)

 Parameters
 ----------
 alpha : array-like float
   Phase angle in degrees
 par : numbers or tuple of numbers
   Coefficients for the Legendre polynomial (starting from the first
   order coefficent.  The order of Legendre polynomial is determined
   by the number of input parameters.  The 0th order coefficient is
   always 1, and omitted from the input coefficient list.

 Returns
 -------
 The Legendre polynomial as normalized single-particle phase function

 Notes
 -----
 The first order Legendre polynomial takes the form:
        P(alpha) = 1 + par[0] * cos(phase)
 The second order Legendre polynomial takes the form:
        P(alpha) = 1 + par[0] * cos(phase) + par[1]* (1.5 * cos(alpha) **2 - 0.5)
    '''
    from numpy.polynomial.legendre import Legendre
    leg = Legendre(np.concatenate((np.array([1.]),np.array(par).reshape(-1))))
    return leg(np.cos(alpha*np.pi/180))


def shoe(alpha, par, form='bowell'):
    '''
 Calculate the SHOE (Shadow-hiding opposition effect) model

 Parameters
 ----------
 alpha : array-like float
   Phase angle in degrees
 par   : tuple of numbers
   Opposition parameters (b0, h), where b0 is the amplitude, and h is
   the width of SHOE
 form : string: 'bowell' or 'hapke', optional
   The form of SHOE model.

 Returns
 -------
 The Shadow-hiding opposition effect model.

 Notes
 -----
 Bowell model is from Bowell et al. (1989) Asteroids II chapter
 Hapke model is from Hapke (1981) original paper
    '''

    b0, h = par
    ph_cp, b0_cp, h_cp = nparr(alpha, b0, h)

    if form == 'bowell':
        if h==0:
            oe = np.ones_like(ph_cp)*b0
        else:
            oe = b0_cp/(1+np.tan(ph_cp*np.pi/360)/h_cp)

        return 1+oe


def cboe(pha, par):
    '''
 Calculate CBOE (coherent-backscatter opposition effect) model

 Parameters
 ----------
 pha : array-like float
   Phase angle in degrees
 par : tuple of numbers
   The opposition parameters (b0, h), where b0 is the amplitude, and h
   is the width of CBOE

 Returns
 -------
 The coherent-backscatter opposition effect model.
    '''

    b0, h = par
    phacp, b0cp, hcp = nparr(pha, b0, h)
    tang2h = np.tan(pha*np.pi/360)/hcp
    zeros = tang2h == 0
    tang2h[zeros] = 0.1
    bcg = 0.5*(1.+(1.-np.exp(-tang2h))/tang2h)/(1+tang2h)**2
    bcg[zeros] = 1.

    return 1.0+b0*bcg


def effang(i, e, psi, theta, cos=False):
    '''
 Calculate the effective scattering angles with specified roughness.

 Parameters
 ----------
 i, e : array-like float
   Incidence angles and emission angles
 psi : array-like float
   Angle between incidence and emission planes, in degrees
 theta : array-like float
   Photometric roughness parameter, in degrees
 cos : bool, optional
   If true, then i and e are the cosines of incidence and emission
   angles

 Returns
 -------
 numpy array of float:
   Returns the effective i.  If cos keyword is set to True, then the
   returned values are cosines of effective i.  In order to calculate
   effective emission angle, one just needs to swap the i and e in the
   call: e_eff = effang(e, i, psi, theta[, cos=False])

 Notes
 -----
 Reference: Hapke (2012), Eqs. 12.46, 12.47, 12.52, and 12.53.
    '''

    def e1(cotx, cottheta):
        '''
        Support functions for computing i_eff and e_eff, all angles
        are in rad, NOT deg
        Ref. Hapke (2012), Eq. 12.45b
        '''

        cotx_cp, cottheta_cp = nparr(cotx, cottheta)
        return np.exp(-2/np.pi*cotx_cp*cottheta_cp)


    def e2(cotx, cottheta):
        '''
        Support functions for computing i_eff and e_eff, all angles
        are in rad, NOT deg
        Ref. Hapke (2012), Eq. 12.45c
        '''

        cotx_cp, cottheta_cp = nparr(cotx, cottheta)
        return np.exp(-1/np.pi*cottheta_cp*cottheta_cp*cotx*cotx)

    icp, ecp, psi_cp, theta_cp = nparr(i, e, psi, theta)
    psi_cp = psi_cp*np.pi/180  # convert to radiance

    # cosine and sine of i and e
    if cos:
        mux1, mux2 = icp, ecp
    else:
        mux1, mux2 = np.cos(icp*np.pi/180), np.cos(ecp*np.pi/180)
    sx1, sx2 = np.sqrt(1-mux1*mux1), np.sqrt(1-mux2*mux2)

    # cotan(theta)
    tantheta = np.tan(theta_cp*np.pi/180)
    cottheta = 1/np.where(tantheta != 0, tantheta, 1.)
    cottheta[tantheta == 0] = np.inf

    # cotan(i)
    cotx1 = mux1/np.where(sx1 != 0, sx1, np.repeat(1,sx1.size))
    cotx1[sx1 == 0] = np.inf

    # cotan(e)
    cotx2 = mux2/np.where(sx2 != 0, sx2, np.repeat(1,sx2.size))
    cotx2[sx2 == 0] = np.inf

    condition = sx1>=sx2
    cot1, cot2 = np.where(condition, cotx1, cotx2), np.where(condition, cotx2, cotx1)
    e1b, e1s = e1(cot1, cottheta), e1(cot2, cottheta)
    e2b, e2s = e2(cot1, cottheta), e2(cot2, cottheta)
    f = np.where(condition, 1., np.cos(psi_cp))
    g = np.where(condition, -1., 1.)

    xi = 1./np.sqrt(1.+np.pi*tantheta*tantheta)
    mue = xi*(mux1+sx1*tantheta*(f*e2b+g*np.sin(psi_cp/2)**2*e2s)/(2-e1b-psi_cp*e1s/np.pi))

    if cos:
        return mue
    else:
        return np.arccos(mue)*180/np.pi


def sfunc(inc, ieff, emi, eeff, psi, theta, cos=False):
    '''
 Calculate the correction function for photometric roughness

 Parameters
 ----------
 inc, emi : array-like float
   Incidence angle and emission angle
 ieff, eeff : array-like float
   Effective incidence angle and emission angle corrected for
   roughness
 psi : array-like float
   Scattering angle between incidence and emission planes, in degrees
 theta : float
   Roughness parameter in degrees
 cos : bool, optional
    If set True, then inc, ieff, emi, eeff are all the cosines of
    corresponding angles.  Default is False, and the angles are in
    degree

 Returns
 -------
 Returns a numpy array containing the roughness correction values for
 corresponding input parameters.

 Notes
 -----
 Ref. Hapke (2012), Eqs. 12.50 and 12.54
    '''

    inc_cp, ieff_cp, emi_cp, eeff_cp, psi_cp, theta_cp = nparr(inc, ieff, emi, eeff, psi, theta)

    if cos:   # inputs are mu and mu0
        mu0, mu0e, mu, mue = inc_cp, ieff_cp, emi_cp, eeff_cp
    else:     # inputs are angles in deg
        mu0, mu0e = np.cos(inc_cp*np.pi/180), np.cos(ieff_cp*np.pi/180)
        mu, mue = np.cos(emi_cp*np.pi/180), np.cos(eeff_cp*np.pi/180)
    si, se = np.sqrt(1-mu0*mu0), np.sqrt(1-mu*mu)

    tantheta = np.tan(theta_cp*np.pi/180)
    cottheta = 1/np.where(tantheta != 0, tantheta, 1.)
    cottheta[tantheta == 0] = np.inf

    xi = 1./np.sqrt(1.+np.pi*tantheta*tantheta)
    fpsi = np.exp(-2*np.tan(psi_cp*np.pi/360))

    mu0e0, mue0 = effang(mu0,mu,0,theta_cp,cos=True), effang(mu,mu0,0,theta_cp,cos=True)

    s1 = np.where(si <= se, mu0/mu0e0, mu/mue0)
    s = mue/mue0*mu0/mu0e0*xi/(1-fpsi+fpsi*xi*s1)

    return s


def rsingle(sca_angle, par, sppf='hg'):
    '''
 Calculate single scattering component based on Hapke (2012) formulism

 Parameters
 ----------
 sca_angle : tuple
   Containing the scattering angles (inc, emi, pha), all in degrees
 params : dictionary
   The parameters of Hapke model.  The form of Hapke model used is
   determined by the parameters contained here and the keyword
   parameter sppf.
   The Hapke parameters used here are:
    'w' : Single-scattering albedo
    'g' : Parameters for single-particle phase function
    'shoe'  : A tuple containing the parameters (b0, h) for SHOE
    'cboe'  : A tuple containing the parameters (b0, h) for CBOE
    'theta' : Hapke roughness parameter
    'phi'   : Hapke porosity parameter
 sppf : string: 'hg' or 'poly', optional
   The form of single-partical phase function.  Accepted values are,
   'hg' - Henyey-Greenstein function; 'poly' - Legendre polynomial
   Default is Henyey-Greenstein function.

 Returns
 -------
 numpy array:
 Return the single particle scattering reflectance

 Notes
 -----
 Note that the single scattering reflectance calculated here does not
 include CBOE and any correction for porosity
    '''

    inc, emi, pha = nparr(sca_angle)
    mu0, mu = np.cos(inc*np.pi/180), np.cos(emi*np.pi/180)

    w = par['w']

    # single-particle phase function
    g = par['g']
    if sppf == 'hg':
        pfunc = hg_func(pha, g)
    elif sppf == 'poly':
        pfunc = poly_phase(pha, g)

    if 'theta' in par:    # correct for roughness
        theta = par['theta']
        psi = calc_psi(mu0, mu, pha, cos=True)
        mu0e = effang(mu0, mu, psi, theta, cos=True)
        mue = effang(mu, mu0, psi, theta, cos=True)
        ss = sfunc(mu0, mu0e, mu, mue, psi, theta, cos=True)
        rs = w/(4*np.pi)*mu0e/(mu0e+mue)*pfunc*ss
    else:
        rs = w/(4*np.pi)*mu0/(mu0+mu)*pfunc

    # shoe
    if 'shoe' in par:
        shpar = par['shoe']
        rs *= shoe(pha, shpar)

    # cboe
    if 'cboe' in par:
        cboe = params['cboe']
        rs *= shoe(pha, cboe)

    # porosity
    if 'phi' in par:
        phi = params['phi']
        phi23 = 1.209*phi**(2./3)
        K = -np.log(1-phi23)/phi23
        rs *= k

    return rs


def misma(sca_angle, par, cos=False, sppf='hg', hfuncver='02'):
    '''
 Calculate the modified IMSA term (MISMA)

 Usage::
 M(mu0, mu) = misma((inc, emi, pha), par, cos=False, sppf = 'hg')

 Parameters
 ----------
 sca_angle : tuple
   Scattering angles (inc, emi, pha), all in degrees
 par : dictionary
   Photometric parameters.  The form of model is determined by both
   the members of par and the keyword sppf.
   Members of par:
      'w'     : Single scattering albedo
      'g'     : Tuple, single particle phase function parameters.  The
         form of SPPF is determined by the number of parameters
         contained in this tuple and keyword sppf.
         if sppf='hg':
            par = g for single-term HG function
            par = (b, c) for two-parameter HG function
            par = (b1, b2, c) for three-parameter HG function
         if sppf='poly':
            par = (b1, b2, ..., ) for the parameters of Legendre
            polynomial
      'theta' : roughness parameter
      'phi'   : porosity parameter
 sppf : string, optional
   Specify the form of sppf.  Possible values:
      'hg': Henyey-Greenstein function (default)
      'poly': Legendre polynomial
 hfuncver : string, optional
   Speficy the form of Chandrasekhar H-function.  Can be '02'
   (default) or '81'

 Returns
 -------
 Returns the MISMA model based on Hapke (2002, Icarus 157, 523-534).

 Notes:
 ------
 The MISMA model is defined in Hapke (2002), Eq. 17:
 M(mu0, mu) = P(mu0)*[H(mu)-1] + P(mu)*[H(mu0)-1] + P*[H(mu)-1][H(mu0)-1]
 The calculations of P(mu0), P(mu), and P follow Eqs. 23, 24, 25, 26, 27.
 Coefficients of Legendre polynomial for various cases of SPPF are
 discussed for 1st order Legendre polynomial, Rayleigh scatters,
 single-term, two- and three-parameter HG functions.
    '''

    from numpy.polynomial.legendre import Legendre

    inc, emi, pha = nparr(sca_angle)
    if cos:
        mu0, mu = inc, emi
    else:
        mu0, mu = np.cos(inc*np.pi/180), np.cos(emi*np.pi/180)

    # if with roughness, then modify incidence and emission angles to effective angles
    if 'theta' in par:
        theta = par['theta']
        psi = calc_psi(mu0, mu, pha, cos=True)
        mu0, mu = effang(mu0, mu, psi, theta, cos=True), effang(mu, mu0, psi, theta, cos=True)

    # calculate three P integrals P(mu0), P(mu), and P
    pmu0 = np.ones(mu0.size, dtype=mu0.dtype)
    pmu = np.ones(mu.size, dtype=mu.dtype)
    p = An = 1.
    g = par['g']
    ph = Legendre(g)
    if sppf == 'poly':
        for i in np.arange(1., len(g)+1, 2):
            An *= -i/(i+1)
            bn = g[i.astype(int)-1]
            pmu0 += An/i*bn*ph.basis(i)(mu0)
            pmu += An/i*bn*ph.basis(i)(mu)
            p += An*An/(i*i)*bn
    elif sppf == 'hg':
        g = np.asarray(g).flatten()
        if len(g) == 1:
            b1, b2, c = g[0], 0, 1
        elif len(g) == 2:
            b1, b2, c = g[0], g[0], g[1]
        b1p = b2p = 1.
        if b1 != 0:
            b1p = 1./b1
        if b2 != 0:
            b2p = 1./b2
        b1_2 = b1*b1
        b2_2 = b2*b2
        for i in np.arange(1., 20, 2):
            An *= -i/(i+1)
            b1p *= b1_2
            b2p *= b2_2
            bn = (i+0.5)*((1-c)*b2p-(1+c)*b1p)
            dpmu0 = An/i*bn*ph.basis(i)(mu0)
            dpmu = An/i*bn*ph.basis(i)(mu)
            dp = An*An/(i*i)*bn
            pmu0 += dpmu0
            pmu += dpmu
            p += dp
            if np.max([dpmu0.max(), dpmu.max(), dp.max()]) < 1e-15:
                break

    # if porosity is present
    if 'phi' in par:
        phi = par['phi']
        if phi != 0:
            phi23 = 1.209*phi**(2./3)
            K = -np.log(1-phi23)/phi23
            mu0 /= K
            mu /= K

    w = par['w']
    hmu0 = hfunc(w, mu0, version=hfuncver)
    hmu = hfunc(w, mu, version=hfuncver)

    return pmu0*(hmu-1) + pmu*(hmu0-1) + p*(hmu0-1)*(hmu-1)


def rmulti(sca_angle, par, cos=False, sppf='hg', mimsa=False):
    '''
 Calculate multiple scattering component based on Hapke (2012) formulism

 Parameters
 ----------
 sca_angle : tuple
   Scattering angles (inc, emi, pha), all in degrees
 params : dictionary
   Parameters of Hapke model.  The form of Hapke model used is
   determined by the parameters contained here and the keyword
   parameter `sppf`.

   The Hapke parameters used here are:
     'w' : Single-scattering albedo
     'g' : Parameters for single-particle phase function
     'shoe'  : A tuple containing the parameters (b0, h) for SHOE
     'cboe'  : A tuple containing the parameters (b0, h) for CBOE
     'theta' : Hapke roughness parameter
     'phi'   : Hapke porosity parameter
 sppf : string, optional
   The form of single-partical phase function.  Accepted values are,
   'hg' - Henyey-Greenstein function; 'poly' - Legendre polynomial
   Default is Henyey-Greenstein function.
 mimsa : bool, optional
   If True, then use the modified IMSA multiple scattering model,
   first introduced in Hapke (2002).  Default is to use the isotropic
   multiple scattering model (IMSA).
    '''

    inc, emi, pha = nparr(sca_angle)

    if cos:
        mu0, mu = inc, emi
    else:
        mu0, mu = np.cos(inc*np.pi/180), np.cos(emi*np.pi/180)

    w = par['w']

    # calculate roughness correction
    if 'theta' in par:
        theta = par['theta']
        psi = calc_psi(mu0, mu, pha, cos=True)
        mu0e, mue = effang(mu0, mu, psi, theta, cos=True), effang(mu, mu0, psi, theta, cos=True)
        ss = sfunc(mu0, mu0e, mu, mue, psi, theta, cos=True)

    if 'phi' in par:
        phi = par['phi']
        phi23 = 1.209*phi**(2./3)
        K = -log(1-phi23)/phi23

    # calculate multiple scattering term
    if mimsa:
        mfunc = misma(sca_angle, par, cos=cos, sppf=sppf)
    else:
        # roughness corrections
        if 'theta' in par:
            xmu0, xmu = mu0e, mue
        else:
            xmu0, xmu = mu0, mu
        # porosity corrections
        if 'phi' in par:
            xmu0 /= K
            xmu /= K
        mfunc = hfunc(w, xmu0)*hfunc(w, xmu)-1

    if 'theta' in par:   # apply roughness correction
        rm = w/(4*np.pi)*mu0e/(mu0e+mue)*mfunc*ss
    else:
        rm = w/(4*np.pi)*mu0/(mu0+mu)*mfunc

    # apply porosity correction
    if 'phi' in par:
        rm *= K

    # apply cboe correction
    if 'cboe' in par:
        cbpar = par['cboe']
        rm *= cboe(pha, cbpar)

    return rm


def bdr(sca_angle, par, sppf='hg', mimsa=False):
    '''
 Calculate bidirectional reflectance (bdr) based on Hapke (2012)
 formulism

 Parameters
 ----------
 sca_angle   : A tuple containing the scattering angles (inc, emi, pha),
     all in degrees
 params      : A dictionary containing the parameters of Hapke model.
     The form of Hapke model used is determined by the parameters
     contained here and the keyword parameter sppf.

     The Hapke parameters used here are:
     'w' : Single-scattering albedo
     'g' : Parameters for single-particle phase function
     'shoe'  : A tuple containing the parameters (b0, h) for SHOE
     'cboe'  : A tuple containing the parameters (b0, h) for CBOE
     'theta' : Hapke roughness parameter
     'phi'   : Hapke porosity parameter
 sppf    : The form of single-partical phase function.  Accepted
     values are, 'hg' - Henyey-Greenstein function; 'poly' - Legendre
     polynomial.  Default is Henyey-Greenstein function.
 mimsa    : If True, then use the modified IMSA multiple scattering
     model, first introduced in Hapke (2002).  Default is to use the
     isotropic multiple scattering model (IMSA).
    '''

    inc, emi, pha = nparr(sca_angle)
    mu0, mu = np.cos(inc*np.pi/180), np.cos(emi*np.pi/180)

    w = par['w']

    # calculate single-particle phase function
    g = par['g']
    if sppf == 'hg':
        ph = hg_func(pha, g)
    elif sppf == 'poly':
        ph = poly_phase(pha, g)

    # calculate roughness correction
    if 'theta' in par:
        theta = par['theta']
        psi = calc_psi(mu0, mu, pha, cos=True)
        mu0e, mue = effang(mu0, mu, psi, theta, cos=True), effang(mu, mu0, psi, theta, cos=True)
        ss = sfunc(mu0, mu0e, mu, mue, psi, theta, cos=True)

    # calculate porosity correction
    if 'phi' in par:
        phi = par['phi']
        phi23 = 1.209*phi**(2./3)
        K = -log(1-phi23)/phi23

    # calculate multiple scattering term, mfunc
    if mimsa:
        mfunc = misma((mu0, mu, pha), par, cos=True, sppf=sppf)
    else:
        # roughness corrections
        if 'theta' in par:
            xmu0, xmu = mu0e, mue
        else:
            xmu0, xmu = mu0, mu
        # porosity corrections
        if 'phi' in par:
            xmu0 /= K
            xmu /= K
        mfunc = hfunc(w, xmu0)*hfunc(w, xmu)-1

    # apply shoe correction
    if 'shoe' in par:
        shpar = par['shoe']
        ph *= shoe(pha, shpar)

    if 'theta' in par:
        bdr = w/(4*np.pi)*mu0e/(mu0e+mue)*(ph+mfunc)*ss
    else:
        bdr = w/(4*np.pi)*mu0/(mu0+mu)*(ph+mfunc)

    # apply cboe correction
    if 'cboe' in par:
        cbpar = par['cboe']
        bdr *= cboe(pha, cbpar)

    # apply porosity correction
    if 'phi' in par:
        bdr *= K

    return bdr


def RADF(sca_angle, par, **kwarg):
    '''
 Calculate radiance factor (RADF).  This routine converts the
 bidirectional reflectance from bdr_hapke() to RADF by multiplying
 the pi factor.

 See bdr_hapke() for more information.

 History
 -------
 10/1/2013, created by JYL @PSI
    '''
    return bdr(sca_angle, par, **kwarg)*np.pi


def REFF(sca_angle, par, **kwarg):
    '''
 Calculate reflectance factor (REFF).  This routine converts the
 bidirectional reflectance from bdr_hapke() to REFF by multiplying
 pi/mu0, where mu0=cos(i).

 See bdr_hapke() for more information.

 History
 -------
 10/1/2013, created by JYL @PSI
    '''
    mu0 = np.cos(nparr(sca_angle[0]*np.pi/180))
    return bdr(sca_angle, par, **kwarg)*np.pi/mu0


def BRDF(sca_angle, par, **kwarg):
    '''
 Calculate bidirectional reflectance distribution function (BRDF).
 This routine converts the bidirectional reflectance from bdr_hapke()
 to BRDF by dividing mu0, where mu0=cos(i).

 See bdr_hapke() for more information.

 History
 -------
 10/1/2013, created by JYL @PSI
    '''
    mu0 = np.cos(nparr(sca_angle[0]*np.pi/180))
    return bdr(sca_angle, par, **kwarg)/mu0


def geoalb(par, sppf='hg'):
    '''
 Calculate the geometric albedo of a spherical body.

 Usage::
 geometric_albedo = geoalb(par, sppf='hg')

 Parameters
 ----------
 par     : Dictionary containing photometric parameters
 sppf    : Keyword to specify the form of single-particle phase function.
     Could be 'hg' (this is the default) for Henyey-Greenstein function
     or 'poly' for Legendre polynomial function

 Returns
 -------
 Returns the geometric albedo for corresponding input parameters

 Notes
 -----
 Based on Hapke (2012), Eqs. 12.58 and 12.60.  Note that there might be
 a typo in Eq. 12.58.  See Hapke (1984, Icarus 59, 41-59), Eq. 65 and 67.
    '''

    w = nparr(par['w'])
    gamma = np.sqrt(1-w)
    r0 = (1-gamma)/(1+gamma)

    if 'theta' in par:
        theta = nparr(par['theta'])*np.pi/180
        c = 1. - (0.048*theta+0.0041*theta*theta)*r0 - (0.33*theta-0.0049*theta*theta)*r0*r0
    else:
        c = 1.

    gg = par['g']
    if sppf == 'hg':
        p0 = hg_func(0., gg)
    elif sppf == 'poly':
        p0 = poly(0., gg)

    if 'shoe' in par:
        b0s, hs = par['shoe']
    else:
        b0s = 0.

    Ap = w*0.125*((1+b0s)*p0-1)+0.5*r0*(1+r0*0.333333333333)*c

    if 'cboe' in par:
        b0c, hc = par['cboe']
        Ap *= 1+b0c

    return Ap


def phasefunc(pha, par, normalize=0., sppf='hg'):
    '''
 Calculate the disk-integrated phase function.

 Usage::
 phase function = phasefunc(pha, par, normalize=0, sppf='hg')

 Parameters
 ----------
 pha     : Array-like, phase angle in degrees
 par     : Dictionary, input parameters
 normalized  : A number in [0., 180), the phase angle that the
    returned phase function will be normalized to.  If it is not
    defined (None) or a number outside of this range, then the
    phase function will not be normalized, and the value at zero
    phase angle will be the geometric albedo.
 sppf    : Specify the single-particle phase function.  Default is
     'hg' for Henyey-Greenstein function.  Can also be 'poly' for
     Legendre polynomial.

 Returns
 -------
 Disk-integrated phase function.
    '''

    pha = np.asarray(pha)
    pha_rad = np.deg2rad(pha)

    w = np.asarray(par['w'])
    gamma = np.sqrt(1-w)
    r0 = (1-gamma)/(1+gamma)

    if 'theta' in par:
        k = HapkeK(np.deg2rad(par['theta']))(pha_rad)
    else:
        k = np.array(1.).reshape(-1)

    gg = par['g']
    if sppf == 'hg':
        p0 = hg_func(0., gg)
        pp = hg_func(pha, gg)
    elif sppf == 'poly':
        p0 = poly_phase(0., gg)
        pp = poly_phase(pha, gg)

    if 'shoe' in par:
        shpar = par['shoe']
        sh = shoe(pha, shpar)
    else:
        sh = np.array(1.).reshape(-1)

    t1 = (w/8.)*(sh*pp-1)+r0*0.5*(1-r0)

    zeroph = pha_rad ==0
    piph = pha == 180.
    pha_rad[zeroph+piph] = 0.1
    t2 = 1-np.sin(pha_rad/2.)*np.tan(pha_rad/2.)*np.log(1./np.tan(pha_rad/4.))
    t2[zeroph] = 1.
    t2[piph] = 0.
    pha_rad[zeroph] = 0.
    pha_rad[piph] = np.pi

    t3 = 2*r0*r0/(3*np.pi)*(np.sin(pha_rad)+(np.pi-pha_rad)*np.cos(pha_rad))

    par1 = par.copy()
    if 'theta' in par1:
        del par1['theta']
    ph = (t1*t2+t3)*k/geoalb(par1)

    if 'cboe' in par:
        ph *= cboe(pha, par['cboe'])

    if (normalize == None) or (normalize < 0) or (normalize >= 180):
        ph *= geoalb(par, sppf=sppf)
    elif normalize != 0:
        ph /= phasefunc(normalize, par, sppf=sppf)

    return ph


def phaseint(par, sppf='hg', steps=5000):
    '''
 Calculate the phase integral based on input Hapke parameters

 Input
 -----
 par     : Dictionary, input parameters
 sppf    : Specify the single-particle phase function.  Default is
     'hg' for Henyey-Greenstein function.  Can also be 'poly' for
     Legendre polynomial.
 steps : The number of steps of phase angles [deg] for numerical
     integration.

 Returns
 -------
 The phase integral

 Notes
 -----
 This program calculates phase integral numerically, based on the
 phase function provided by phasefunc().

 History
 -------
 10/1/2013, created by JYL @PSI
    '''

    pha = np.linspace(0,180,steps)
    f = phasefunc(pha, par, sppf=sppf)
    dpha = np.empty(steps)
    dpha[:] = np.pi/(steps-1)
    dpha[[0,-1]] /= 2
    return 2*(f*np.sin(pha*np.pi/180)*dpha).sum()


def bondalb(par, sppf='hg'):
    '''
 Calculate the Bond albedo based on input Hapke parameters
 See geoalb() for more details.

 Returns
 -------
 The Bond albedo

 History
 -------
 10/2/2013, created by JYL @PSI
    '''

    return geoalb(par, sppf=sppf) * phaseint(par, sppf=sppf)


def mpfit_func(p, sca_angle=None, iof=None, err=None, parnames=None, sppf='hg', mimsa=False, fjac=None):
    '''
 User defined function for mpfit, to be called by mpfit_photo()

 Input
 -----
 p  : array-like, floating point
     Model parameters
 sca_angle : tuple, floating point array-like
     Scattering angles (inc, emi, pha) [deg]
 radf : array-like, floating point
     Measured RADF
 err : array-like, floating point
     Measurement error for the input 'radf'
 parnames : array-like, string
     Names of the corresponding parameters in 'p'
 sppf : string
     Specify the single-particle phase function.  Default is
     'hg' for Henyey-Greenstein function.  Can also be 'poly' for
     Legendre polynomial.
  mimsa : boolean
     If True, then use the modified IMSA multiple scattering
     model, first introduced in Hapke (2002).  Default is to use the
     isotropic multiple scattering model (IMSA).

 Output
 ------
 Returns array-like floating point as the evaluation of weighted
 deviation between model and measurements

 History
 -------
 10/2/2013, created by JYL @PSI
    '''

    if err is None:
        err = np.ones_like(iof)

    # choose a model and construct the model parameter variable
    # populate the dictionary of parameters
    par = {}
    for name, value in zip(parnames, p):
        if name.find('g') != -1:
            if 'g' in par:
                par['g'] += (value,)
            else:
                par['g'] = (value,)
        elif name.find('cboe') != -1:
            if 'cboe' in par:
                par['cboe'] += (value,)
            else:
                par['cboe'] = (value,)
        elif name.find('shoe') != -1:
            if 'shoe' in par:
                par['shoe'] += (value,)
            else:
                par['shoe'] = (value,)
        else:
            par[name] = value

    return (0, (RADF(sca_angle, par, sppf=sppf, mimsa=mimsa)-iof)/err)


def mpfit(sca_angle, iof, par0=None, err=None, sppf='hg', mimsa=False, fixed=None, limits=None, parinfo=None, return_mpfit=False, quiet=False, xtol=1.e-10, gtol=1.e-10, ftol=1.e-10, maxiter=200, **kwarg):
    '''
 Fit Hapke model to measured RADF.

 Input
 -----
 sca_angle : tuple
   The scattering angles (inc, emi, pha), each element being array-like
   floating point [deg]
 radf : array-like, floating point
   The measured RADF.
 par0 : A dictionary containing the parameters of Hapke model.
   The form of Hapke model used is determined by the parameters
   contained here and the keywords 'sppf' and 'mimsa'.  All the
   parameters presented here will be fitted, unless otherwise
   specified by keywords 'fixed' or 'parinfo'.  Can be overrided by
   'parinfo'.
   The Hapke parameters used here are:
     'w' : Single-scattering albedo
     'g' : Parameters for single-particle phase function, a scaler or
       array-like depending on the number of parameters.
     'shoe'  : A tuple containing the parameters (b0, h) for SHOE
     'cboe'  : A tuple containing the parameters (b0, h) for CBOE
     'theta' : Hapke roughness parameter
     'phi'   : Hapke porosity parameter
 err : array-like, floating point
   The measurement uncertainty associated with input 'radf'.
 sppf : string
   The form of single-partical phase function.  Accepted values are,
   'hg' - Henyey-Greenstein function; 'poly' - Legendre polynomial.
   Default is Henyey-Greenstein function.
 mimsa : boolean
   If True, then use the modified IMSA multiple scattering model,
   first introduced in Hapke (2002).  Default is to use the isotropic
   multiple scattering model (IMSA).
 fixed : dictionary, Boolean.
   Specify whether the corresponding parameters should be kept fixed
   in fitting.  If a parameter is not specified, then it will be
   adjusted by default.  This keyword will be overrided by 'parinfo'
   if present.
 limits : dictionary
   The limits of parameters specified by [min, max].  This keyword
   will be overrided by 'parinfo'
 parinfo : Dictionary list
   Same as mpfit() keyword 'parinfo'.  See mpfit() documents for
   details.  If present, then the information contained in this
   keyword will override that in par0.
 return_mpfit : boolean
   If true, then a mpfit class as directly returned by mpfit will be
   returned.
 All keywords accepted by mpfit class (see mpfit document) will be
 accepted here.  Some important ones are listed here:
   ftol = 1e-10
   xtol = 1e-10
   gtol = 1e-10
   damp = 0.0
   maxiter = 200
   nprint = 1
   quiet = 0

 Output
 ------
 If return_mpfit=False (default), returns a dictionary with all the
 best-fit results.
 'par' : dictionary
   The best fit photometric parameters.  It has the same structure and
   elements as keyword 'par0', or as what's specified in 'parinfo'.
 'status' : integer
   The status of fit.  See mpfit.
 'chisq' : floating point
   The best-fit Chi-Sq == total( [ (measurement - model)/sigma ]**2 )
 'rms' : floating point
   Root Mean Square = sqrt(chisq/degree_of_freedom)
 'relrms' : float
   RMS relative to average RADF
 'perror' : dictionary
   Formal 1-sigma errors of best-fit parameters.  It has the same
   structure as 'par'.
 'serror' : dictionary
   Formal scaled 1-sigma errors of the best-fit parameters.
     serror = perror * sqrt(chisq / dof)
   where dof = deg of freedom
     dof = len(radf) - len(parameters_fitted)
 'errmsg' : string
   A string error or warning message
 'niter' : number of iterations completed

 If return_mpfit=True, then a tuple will be returned with the first
 element as above dictionary, the second element as an mpfit class.

 History
 -------
 10/2/2013, created by JYL @PSI
    '''
    import mpfit

    # process parameters to be fitted
    if parinfo != None:
        for info in parinfo:
            if 'parname' not in info:
                errmsg = 'All parameter names is not specified.'
                print('Error: '+errmsg)
                return {'status': -2, 'errmsg': errmsg}
    else:
        if par0 == None:
            errmsg = 'Initial parameter values not specified.'
            print('Error: '+errmsg)
            return {'status': -1, 'errmsg': errmsg}
        else:
            parinfo = []
            info_temp = {'value':0, 'step':0.01}
            for p in par0:
                if hasattr(par0[p],'__iter__'):  # tuple, for 'g', 'shoe', and 'cboe'
                    i = 0
                    for v in par0[p]:
                        info = info_temp.copy()
                        info['parname'] = p+str(i+1)
                        info['value'] = v
                        if fixed is not None and p in fixed and fixed[p][i] != None:
                            info['fixed'] = fixed[p][i]
                        if limits != None and p in limits:
                            info['limits'] = np.array(limits[p][i])
                            info['limited'] = np.array([True]*len(limits[p][i]))
                        parinfo.append(info.copy())
                        i += 1
                else:
                    info = info_temp.copy()
                    info['parname'] = p
                    info['value'] = par0[p]
                    if fixed is not None and p in fixed and fixed[p] != None:
                        info['fixed'] = fixed[p]
                    if limits is not None and p in limits:
                        info['limits'] = np.array(limits[p])
                        info['limited'] = np.array([True]*len(limits[p]))
                    parinfo.append(info.copy())

    if not quiet:
        print()
        print('Fit hapke parameters to input data:')
        print('Number of data points: ', repr(len(iof)).ljust(10))
        print('Incidence angle: from %8.2f to %8.2f' % (sca_angle[0].min(), sca_angle[0].max()))
        print('Emission angle:  from %8.2f to %8.2f' % (sca_angle[1].min(), sca_angle[1].max()))
        print('Phase angle:     from %8.2f to %8.2f' % (sca_angle[2].min(), sca_angle[2].max()))
        print()
        print('Single-scattering phase function: ', end=' ')
        if sppf == 'hg':
            print('Henyey-Greenstein')
        else:
            print('Legendre polynomial')
        print('Modified isotropic multiple scattering approximation: ', end=' ')
        if mimsa:
            print('yes')
        else:
            print('no')
        print()
        print('Parameter setting:')
        print('  name     init      limited?       range      fixed?')
        print('-------- ------- ------------ --------------- --------')
        for info in parinfo:
            print('%6s %9.4f' % (info['parname'],info['value']), end=' ')
            if 'limited' in info:
                for lim in info['limited']:
                    if lim:
                        print('  yes', end=' ')
                    else:
                        print('   no', end=' ')
            if 'limits' in info:
                print('[%7.2f, %7.2f]' % (info['limits'][0], info['limits'][1]), end=' ')
            else:
                print('%14s' % ' ')
            if 'fixed' in info and info['fixed']:
                print('  yes', end=' ')
            else:
                print('   no', end=' ')
            print()
        print()
        print()

    # populate initial values of parameters
    p0, parnames = [], []
    for info in parinfo:
        p0.append(info['value'])
        parnames.append(info['parname'])

    # process measurement errors
    if err == None:
        err = np.ones_like(iof)

    # initialize functkw
    fa = {'sca_angle': sca_angle, 'iof': iof, 'err': err, 'sppf': sppf, 'mimsa': mimsa, 'parnames': parnames}

    #print p0
    #print fa
    #print parinfo

    # fit model
    fitp = mpfit.mpfit(mpfit_func, p0, functkw=fa, parinfo=parinfo, quiet=quiet, xtol=xtol, gtol=gtol, ftol=ftol, maxiter=maxiter, **kwarg)

    if fitp.status >= 1 and fitp.status <= 8:  # successful fit
        dof = len(iof)-len(p0)
        rms = np.sqrt(fitp.fnorm/dof)
        serrors = fitp.perror * rms
        # populate the dictionary to be returned
        par, serror, perror = {}, {}, {}
        for name, value, perr, serr in zip(parnames, fitp.params, fitp.perror, serrors):
            if name.find('g') != -1:
                if 'g' in par:
                    par['g'] += (value,)
                    perror['g'] += (perr,)
                    serror['g'] += (serr,)
                else:
                    par['g'], perror['g'], serror['g'] = (value,), (perr,), (serr,)
            elif name.find('cboe') != -1:
                if 'cboe' in par:
                    par['cboe'] += (value,)
                    perror['cboe'] += (perr,)
                    serror['cboe'] += (serr,)
                else:
                    par['cboe'], perror['cboe'], serror['cboe'] = (value,), (perr,), (serr,)
            elif name.find('shoe') != -1:
                if 'shoe' in par:
                    par['shoe'] += (value,)
                    perror['shoe'] += (perr,)
                    serror['shoe'] += (serr,)
                else:
                    par['shoe'], perror['shoe'], serror['shoe'] = (value,), (perr,), (serr,)
            else:
                par[name], perror[name], serror[name] = value, perr, serr
        if len(par['g']) == 1:
            par['g'], perror['g'], serror['g'] = par['g'][0], perror['g'][0], serror['g'][0]

        # set status message
        if fitp.status == 1:
            errmsg = 'Both actual and predicted relative reductions in the sum of squares are at most ftol (%.5g).' % ftol
        elif fitp.status == 2:
            errmsg = 'Relative error between two consecutive iterates is at most xtol (%.5g).' % xtol
        elif fitp.status == 3:
            errmsg = 'Both actual and predicted relative reductions in the sum of squares are at most ftol (%.5g), and Relative error between two consecutive iterates is at most xtol (%.5g).' % (ftol, xtol)
        elif fitp.status == 4:
            errmsg = 'The cosine of the angle between fvec and any column of the jacobian is at most gtol (%.5g) in absolute value.' % gtol
        elif fitp.status == 5:
            errmsg = 'The maximum number of iterations (%d) has been reached.' % maxiter
        elif fitp.status == 6:
            errmsg = 'ftol (%.5g) is too small. No further reduction in the sum of squares is possible.' % ftol
        elif fitp.status == 7:
            errmsg = 'xtol (%.5g) is too small. No further improvement in the approximate solution x is possible.' % xtol
        elif fitp.status == 8:
            errmsg = 'gtol (%.5g) is too small. fvec is orthogonal to the columns of the jacobian to machine precision.' % gtol

        fit = {'par': par, 'status': fitp.status, 'chisq': fitp.fnorm, 'rms': rms, 'relrms': rms/iof.mean(), 'perror': perror, 'serror': serror, 'errmsg': errmsg, 'niter': fitp.niter}

        if not quiet:
            print()
            print('Fit successful with a status code: %1d' % fitp.status)
            print('  >>> %s' % errmsg)
            print('Total number of iterations: %d' % fitp.niter)
            print('Best-fit Chi-squared: %9.5f' % fitp.fnorm)
            print('Root mean squared (RMS): %8.4f, %6.2f%%' % (rms, rms/iof.mean()*100))
            print()
            print('Final parameters:')
            print('  name    value     perror    serror')
            print(' ------ --------- --------- ---------')
            for p in par:
                if hasattr(par[p], '__iter__'):
                    for v, perr, serr in zip(par[p], perror[p], serror[p]):
                        print('%6s %9.4f %9.4f %9.4f' % (p, v, perr, serr))
                else:
                    print('%6s %9.4f %9.4f %9.4f' % (p, par[p], perror[p], serror[p]))
            print()
            print('Geometric albedo: %7.4f' % geoalb(par))
            print('Bond albedo: %7.4f' % bondalb(par))

    else:  # unsuccessful fit
        if fitp.status == -16:
            errmsg = 'A parameter or function value has become infinite or an undefined number.'
        elif fitp.status == 0:
            errmsg = 'Improper input parameters.'
        else:
            errmsg = 'Undefined errors.'
        fit = {'status': fitp.status}

        if not quiet:
            print()
            print('Fit failed with a status code: %1d' % fitp.status)
            print('  >>> %s' % errmsg)
            print()

    if return_mpfit:
        return (fit, fitp)
    else:
        return fit


def packdata(inc, emi, pha, iof, outfile, append=True, clobber=False):
    '''
 Pack reflectance data to FITS table file that can be directly used by
 hapke.fitfile

 Parameters
 ----------
 inc, emi, pha, iof : arrays
   Arrays containing incidence angle, emission angle, phase angle, and
   reflectance data
 outfile : string, or file object
   Output FITS file
 append : bool, optional
   If `True`, then data will be appended to existing FITS file.  If
   `False`, then data will be written to a new file.  If the file
   already exists, then see `clobber` keyword
 clobber : bool, optional
   If `True`, then if the output file exists, it will be overwritten.
   Otherwise an error message will be displayed by
   jylipy.write_fitstable

 Returns
 -------
 None

 v1.0.0 : JYL @PSI, July 16, 2014
    '''

    from astropy.io import fits
    write_fitstable(outfile, inc, emi, pha, iof, colnames='Inc Emi Pha IoF'.split(), append=append, clobber=clobber)


def loaddata(infile):
    '''
 Load the reflectance data from FITS file

 Parameters
 ----------
 infile : str
   Name of the FITS file that contains the I/F data table, which needs
   to contain at least four columns, with names "Inc", "Emi", "Pha",
   "IoF".  It can contain "IoFErr" column as the measurement errors
   of I/F data.

 Returns
 -------
  (inc, emi, pha, iof) : arrays
  (inc, emi, pha, iof, incerr, emierr, phaerr, ioferr)

 v1.0.0 : JYL @PSI, July 16, 2014
    '''

    from astropy.io import fits
    f = fits.open(infile)
    return f[1].data['Inc'],f[1].data['Emi'],f[1].data['Pha'],f[1].data['IoF']


def fitfile(datafile, **kwarg):
    '''Fit Hapke model using data from a datafile

 Parameters
 ----------
 datafile : str
   Name of the FITS file that contains the I/F data table, which needs
   to contain at least four columns, with names "Inc", "Emi", "Pha",
   "IoF".  It can contain "IoFErr" column as the measurement errors
   of I/F data.
 **kwarg : See `modelfit`

 v1.0.0 : 5/28/2015, JYL @PSI
    '''

    new = kwarg.pop('new', False)

    from astropy.io import fits
    f = fits.open(datafile)
    if new:
        from .core import PhotometricData
        data = PhotometricData(datafile)
        data.unit = 'deg'
        data.cos = False
        inc = data.inc.value
        emi = data.emi.value
        pha = data.pha.value
        iof = data.RADF
        ioferr = None
    else:
        inc = f[1].data['inc']
        emi = f[1].data['emi']
        pha = f[1].data['pha']
        iof = f[1].data['RADF']
        if 'IoFErr' in f[1].data.columns.names:
            ioferr = f[1].data['IoFErr']
        else:
            ioferr = None
    return modelfit(inc, emi, pha, iof, ioferr=ioferr, **kwarg)


def modelfit(inc, emi, pha, iof, ioferr=None, parlist=['w', 'g', 'shoe', 'theta'], sppf='hg', mimsa=False, fixed=None, limits=None, inclim=None, emilim=None, phalim=None, ntrial=100, quiet=False, plot=False, return_all=False, **kwarg):
    '''
 Fit Hapke model for the data in the datafile.

 Input
 -----
 inc, emi, pha, iof : array-like, numbers
   The incidence angle, emission angle, phase angle, and I/F data to
   be fitted
 parlist : array-like, string
   The name of Hapke parameters contained in the model.  This
   parameter is used, together with keywords 'sppf' and 'mimsa' to
   determine the form of Hapke model used.  The available names can
   be:
     'w' : single-scattering albedo
     'g#' : single-scattering phase function.  # specifies how many
       parameters should be included, can be 1, 2, or 3.  If omitted,
       then it is assumed to be g1.
     'shoe' : shadow-hiding opposition effect
     'cboe' : coherent-backscatter opposition effect
     'theta' : macroscopic roughness
     'phi' : porosity parameter
   If parlist is omitted, then a five-parameter version of Hapke model
   with 'w', 'g1', 'shoe', 'theta', using Henyey-Greenstein single-
   scattering phase function and ignore MIMSA.
 sppf : string, 'hg' or 'poly'
   Specify the single-scattering phase function
 mimsa : boolean
   Specify whether use MIMSA or not
 fixed : sequence, floating point
   Specify the values that the corresponding parameters in 'parlist'
   should be fixed.  A None value means not fixed.  For multi-
   parameter parameters, such as 'g2' or 'g3', 'shoe', and 'cboe',
   the corresponding element should be another sequence.
 limits : sequence, floating point
   Pairs of values to specify the lower and upper limit of the
   corresponding parameters in 'parlist'.  None value are interpreted
   as confined in their physical meaningful ranges:
     'w' : [0, 1]
     'g' :
         'sppf' 1-parameter : [-1, 1]
         'sppf' 2-parameter : [[0, 1], [-1, 1]]
         'sppf' 3-parameter : [[0, 1], [0, 1], [-1, 1]]
         'poly' : no bound
     'theta' : [0, 60]
     'phi' : [0, 0.75]
     'shoe' : [[0, 10], [0, 0.2]]
     'cboe' : [[0, 10], [0, 0.2]]
 inclim, emilim, phalim: 2-element sequences, optional
   The lower and upper limit of incidence, emission, and phase angles
 ntrial : integer
   Number of trials with random initial parameters
 quiet : boolean
   If True, all screen print are supressed.
 plot : bool, or a list of figure class
   Plot the statistics of all trials and the goodness of final fit
 return_all : boolean
   If True, then all trial fit results will be returned, together with
   the initial conditions.  See Output for more.
 This function accepts other keywords accepted by hapke.mpfit.

 Output
 ------
 1. A dictionary of the best-fit results.  Same as the return of
   hapke.mpfit().
 2. If keyword return_all is set True, then a tuple
   (best_fitp, all_fitp, par0) will be returned:
   best_fitp: the best-fit results
   all_fitp: a list of all fit results
   par0: a list of initial conditions
 3. If `plot` is set, then a list of matplotlib Figure class will be
   appended as the last member of the returned tuple.

 Note
 ----
 This fitting program uses hapke.mpfit to find the best-fit
 parameters, and will generate 'ntrial' random initial parameter sets
 to make sure a global minimum in Chi-squared space is reached.

 History
 -------
 10/3/2013, created by JYL @PSI
 5/28/2015, JYL @PSI
   Changed name from `fitfile` to `modelfit`.  Reserve `fitfile` as a
     wrapper of this routine to input photometric data from a file
   Add keyword `ioferr`
   Remove keyword `weighted`.  Weighted fit is indicated by `ioferr`
    '''

    # check plot parameter
    if type(plot) is not bool:
        import matplotlib
        if not hasattr(plot,'__iter__'):
            plot = [plot]
        for f in plot:
            if type(f) is not matplotlib.figure.Figure:
                raise TypeError('Please pass only matplotlib.figure.Figure class.')

    # load i/f data
    good = np.ones(len(inc),dtype=bool)
    if inclim != None:
        good = good & (inc>inclim[0]) & (inc<inclim[1])
    if emilim != None:
        good = good & (emi>emilim[0]) & (emi<emilim[1])
    if phalim != None:
        good = good & (pha>phalim[0]) & (pha<phalim[1])
    sca = (inc[good], emi[good], pha[good])
    trial = bdr(sca, {'w':0.5, 'shoe':[1.0, 0.03], 'theta':20, 'g':-0.25})
    val = ~np.isnan(trial)
    sca = ((inc[good])[val], (emi[good])[val], (pha[good])[val])
    iof = (iof[good])[val]

    if ioferr is not None:
        err = ioferr[good][val]
    else:
        err = None

    # generate 'limits' if not provided
    oelim = [[0.,6.],[0.,.5]]  # SHOE or CBOE
    wlim = [0.,1.]  # SSA
    tlim = [0.,60.]  # theta
    philim = [0.,0.75]  # phi
    g1lim = [-1.,1.]  # g for 1PHG
    g2lim = [[0.,1.],[-1.,1.]]  # b, c for 2PHG
    g3lim = [[0.,1.],[0.,1.],[-1.,1.]]  # b1, b2, c for 3PHG
    if limits == None:
        limits = []
        for parname in parlist:
            if parname[0] == 'g':
                if len(parname) == 1:
                    ng = 1
                else:
                    ng = int(parname[1])
                if ng == 1:
                    limits.append(g1lim)
                elif ng == 2:
                    limits.append(g2lim)
                else:
                    limits.append(g3lim)
            elif parname == 'shoe' or parname == 'cboe':
                limits.append(oelim)
            elif parname == 'w':
                limits.append(wlim)
            elif parname == 'theta':
                limits.append(tlim)
            elif parname == 'phi':
                limits.append(philim)

    # generate 'fixed' parameter if not provided
    if fixed == None:
        fixed = []
        for parname in parlist:
            if parname[0] == 'g':
                if len(parname) == 1:
                    ng = 1
                else:
                    ng = int(parname[1])
                if ng == 1:
                    fixed.append(None)
                else:
                    fixed.append([None]*ng)
            elif parname == 'shoe' or parname == 'cboe':
                fixed.append([None]*2)
            else:
                fixed.append(None)

    # prepare parameters for mpfit
    par0, lim, fix = {}, {}, {}
    for parname, l, f in zip(parlist, limits, fixed):
        if parname[0] == 'g':
            parname = 'g'
        lim[parname] = l
        if not hasattr(f, '__iter__'):
            if f == None:
                fix[parname] = False
                par0[parname] = 0.
            else:
                fix[parname] = True
                par0[parname] = f
        else:
            par0[parname] = []
            fix[parname] = []
            for fn in f:
                if fn == None:
                    fix[parname].append(False)
                    par0[parname].append(0.)
                else:
                    fix[parname].append(True)
                    par0[parname].append(fn)

    # loop through all trials
    tr_par0, tr_fitp = [0]*ntrial, [0]*ntrial
    for i in range(ntrial):

        # print information
        if not quiet:
            print()
            print('--------------------------------------------------')
            print('Trial # %d' % i)

        # populate initial guess parameters
        for parname in par0:
            if not hasattr(fix[parname], '__iter__'):
                if not fix[parname]:
                    par0[parname] = np.random.random()*(lim[parname][1]-lim[parname][0])+lim[parname][0]
            else:
                for j in range(len(fix[parname])):
                    if not fix[parname][j]:
                        par0[parname][j] = np.random.random()*(lim[parname][j][1]-lim[parname][j][0])+lim[parname][j][0]

        import copy
        tr_par0[i] = copy.deepcopy(par0)
        tr_fitp[i] = mpfit(sca, iof, err=err, par0=par0, fixed=fix, limits=lim, sppf=sppf, mimsa=mimsa, quiet=quiet, **kwarg)

    # find the best fit
    sta = np.array([x['status'] for x in tr_fitp])
    goodfit = (sta >= 1) & (sta <= 8)
    ngoodfit = np.size(np.where(goodfit))
    if not quiet:
        print()
        print('All trials finished')
        print()
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print()
        print('Total number of trials: %d' % ntrial)
        print('Total number of successful fit: %d' % ngoodfit)
    if ngoodfit == 0:
        print('All attempts failed.  No solution found.')
        if not return_all:
            return {'status': -1}
        else:
            return ({'status': -1}, tr_fitp, tr_par0)
    goodfitp = np.array(tr_fitp)[goodfit]
    goodpar0 = np.array(tr_par0)[goodfit]
    chisq = np.array([x['chisq'] for x in goodfitp])
    # consider all chisq that is within 2% of the minimum chisq
    bests = np.where(abs(chisq-chisq.min())<.02*chisq.min())
    chisq_hist, edge = np.histogram(chisq[bests], bins=10)
    best_bin = chisq_hist.argmax()
    inbin = np.where((chisq[bests]>=edge[best_bin]) & (chisq[bests]<=edge[best_bin+1]))
    best = np.where(chisq == chisq[bests][inbin].min())[0][0]
    bestfit, bestpar = goodfitp[best], goodfitp[best]['par']

    # print out results
    if not quiet:
        print()
        print('Best fit results:')
        print('Status code: %1d' % bestfit['status'])
        print('  >>> %s' % bestfit['errmsg'])
        print('Total number of iterations: %d' % bestfit['niter'])
        print('Best-fit Chi-squared: %9.5f' % bestfit['chisq'])
        print('Root mean squared (RMS): %8.4f, %6.2f%%' % (bestfit['rms'], bestfit['rms']/iof.mean()*100))
        print()
        print('Final parameters:')
        print('  name    value     perror    serror')
        print(' ------ --------- --------- ---------')
        for p in bestpar:
            if hasattr(bestpar[p], '__iter__'):
                for v, perr, serr in zip(bestpar[p], bestfit['perror'][p], bestfit['serror'][p]):
                    print('%6s %9.4f %9.4f %9.4f' % (p, v, perr, serr))
            else:
                print('%6s %9.4f %9.4f %9.4f' % (p, bestpar[p], bestfit['perror'][p], bestfit['serror'][p]))
        print()
        print('Geometric albedo: %7.4f' % geoalb(bestpar))
        print('Bond albedo: %7.4f' % bondalb(bestpar))
        print()

    # generate graphics to show the statistics of model fit
    if plot is not False:
        import matplotlib.pyplot as plt

        # prepare figures
        if plot is True:
            figs = [plt.figure(dpi=120,figsize=(6,8.5)), plt.figure(dpi=120,figsize=(6,8.5)), plt.figure(dpi=120,figsize=(6,6))]
        else:
            figs = plot
        for i in range(3-len(figs),0,-1):
            f = plt.figure(dpi=120)
            if i == 1:
                f.set_size_inches(6,6,forward=True)
            else:
                f.set_size_inches(6,8.5,forward=True)
            figs.append(f)

        # plot trial statistics
        npar = len(parlist)
        if 'shoe' in parlist:
            npar += 1
        if 'cboe' in parlist:
            npar += 1
        for parname in parlist:
            if parname[0] == 'g':
                if len(parname) > 1:
                    npar += int(parname[1])-1
        f, ax = plt.subplots(npar, 1, num=figs[0].number)
        i = 0
        for parname in par0:
            if not hasattr(par0[parname],'__iter__'):
                v = [x['par'][parname] for x in goodfitp]
                v0 = [x[parname] for x in goodpar0]
                ax[i].hist(v0, bins=20, facecolor='b')
                ax[i].hist(v, bins=20, facecolor='g')
                ax[i].vlines(bestfit['par'][parname], 0, ntrial/2,linewidth=3)
                pplot(ax[i],xlabel=parname)
                i += 1
            else:
                for j in range(len(par0[parname])):
                    v = [x['par'][parname][j] for x in goodfitp]
                    v0 = [x[parname][j] for x in goodpar0]
                    ax[i].hist(v0, bins=20, facecolor='b')
                    ax[i].hist(v, bins=20, facecolor='g')
                    ax[i].vlines(bestfit['par'][parname][j], 0, ntrial/2,linewidth=3)
                    pplot(ax[i],xlabel=parname+repr(j+1))
                    i += 1

        # plot measure/model ratio
        model = RADF(sca, bestpar, sppf=sppf, mimsa=mimsa)
        ratio = iof/model
        xlbl = ['Incidence Angle [deg]', 'Emission Angle [deg]', 'Phase Angle [deg]']
        f, ax = plt.subplots(3, 1, num=figs[1].number)
        for i in range(0,3):
            ax[i].plot(sca[i], ratio, '.')
            ax[i].hlines(1,sca[i].min(),sca[i].max())
            pplot(ax[i],xlabel=xlbl[i], ylabel='Measure/Model')

        # plot model vs. measure
        ax = figs[2].add_subplot(111)
        vmin = min([model.min(),iof.min()])
        vmax = max([model.max(),iof.max()])
        ax.plot(iof,model,'.',[vmin,vmax],[vmin,vmax],'-k')
        pplot(ax,xlabel='Measure I/F', ylabel='Modeled I/F')
        plt.show()

    if not return_all:
        if plot:
            return bestfit, figs
        else:
            return bestfit
    else:
        if plot:
            return bestfit, tr_fitp, tr_par0, figs
        else:
            return bestfit, tr_fitp, tr_par0




class DiskInt5(Fittable1DModel):
    '''
 Hapke disk-integrated phase function model of a spherical shape with
 5 parameters (w, g, theta, B0, h)

 v1.0.0 : JYL @PSI, February 26, 2013

 phase angle is in degrees
 `theta` is in degrees
    '''

    w = Parameter(min=0., max=1.)
    g = Parameter(min=-1., max=1.)
    theta = Parameter(min=0., max=60.)
    B0 = Parameter(min=0.,max=5.)
    h = Parameter(min=0.)


    @staticmethod
    def evaluate(pha, w, g, theta, B0, h):
        return phasefunc(pha, {'w': w, 'g': g, 'theta': theta, 'shoe': (B0, h)}, normalize=None)

    @staticmethod
    def fit_deriv(pha, w, g, theta, B0, h):
        '''
        Derivatives of disk-integrated Hapke model
        '''

        phi = DiskInt5.evaluate(pha, w, g, theta, B0, h)
        ap = geoalb({'w':w,'g':g,'theta':theta,'shoe':(B0,h)})
        ap0 = geoalb({'w':w,'g':g,'shoe':(B0,h)})

        gamma = np.sqrt(1-w)
        r00 = r0(w)
        dr0dw = r0(w,deriv=True)
        b = SHOE_approx(B0, h)(np.deg2rad(pha))
        p = HG1(g)(np.deg2rad(pha))
        p0 = HG1(g)(0)
        anglefactor = np.ones_like(pha)
        z = pha != 0.
        anglefactor[z] = 1+np.sin(np.deg2rad(pha[z])/2)*np.tan(np.deg2rad(pha[z])/2)*np.log(np.tan(np.deg2rad(pha[z])/4))

        # global roughness correction factor k
        k = HapkeK(np.deg2rad(theta))(np.deg2rad(pha))
        dkdtheta = HapkeK.fit_deriv(np.deg2rad(pha), np.deg2rad(theta)).reshape(-1)

        # C correction term
        c = HapkeU(w, np.deg2rad(theta))(0.)
        dcdw, dcdtheta = HapkeU.fit_deriv(0., w, np.deg2rad(theta))

        # partials of ravg0
        w8 = 0.125*w
        # d(ravg0)/dcdw
        dravg0dw = (0.125*((1+b)*p-1)+0.5*(1-2*r00)*dr0dw)*anglefactor+4*r00/(3.*np.pi)*(np.sin(np.deg2rad(pha))+(np.pi-np.deg2rad(pha))*np.cos(np.deg2rad(pha)))*dr0dw
        # d(ravg0)/db_0
        dbdb0, dbdh = SHOE_approx.fit_deriv(np.deg2rad(pha),B0,h)
        dravg0db0 = w8*p*anglefactor*dbdb0
        # d(ravg0)/dh
        dravg0dh = w8*p*anglefactor*dbdh
        # d(ravg0)/dg
        dravg0dg = w8*(1+b)*anglefactor*HG1.fit_deriv(np.deg2rad(pha),g)

        # Partials of A_p0
        # dap0/dw
        dap0dw = 0.125*((1+B0)*p0-1)+0.5*(1+2.*r00/3)*dr0dw
        # dap0/db0
        dap0db0 = w8*p0
        # dap0/dg
        dap0dg = w8*(1+B0)*HG1.fit_deriv(np.deg2rad(pha), g)

        # Partials of A_p
        # dap/dw
        dapdw = 0.125*((1+B0)*p0-1)+0.5*((1+2.*r00/3)*c*dr0dw+r00*(1+r00/3.)*dcdw)
        # dap/db0
        dapdb0 = dap0db0
        # dap/dg
        dapdg = dap0dg
        # dap/dtheta
        dapdtheta = 0.5*r00*(1+r00/3)*dcdtheta

        ravg0 = phi*ap0/k

        # Partials of ravg
        apap0 = ap/ap0
        ravg0ap0 = ravg0/ap0
        ravg0apap02 = ravg0*ap/(ap0*ap0)
        # d(ravg)/dw
        dravgdw = (dravg0dw*apap0+dapdw*ravg0ap0-dap0dw*ravg0apap02)*k
        #d(ravg)/db0
        dravgdb0 = (dravg0db0*apap0+dapdb0*ravg0ap0-dap0db0*ravg0apap02)*k
        # d(ravg)/dh
        dravgdh = dravg0dh*apap0*k
        # d(ravg)/dg
        dravgdg = (dravg0dg*apap0+dapdg*ravg0ap0-dap0dg*ravg0apap02)*k
        # d(ravg)/d(theta)
        dravgdtheta = ravg0*(dkdtheta*ap+dapdtheta*k)/ap0

        return [dravgdw, dravgdg, dravgdtheta, dravgdb0, dravgdh]


class DiskInt5Log(DiskInt5):
    @staticmethod
    def evaluate(*args):
        return np.log10(DiskInt5.evaluate(*args))

    @staticmethod
    def fit_deriv(*args):
        return DiskInt5.fit_deriv(*args)/DiskInt5.evaluate(*args)


class DiskInt6(Fittable1DModel):
    '''
 Hapke disk-integrated phase function model of a spherical shape with
 6 parameters (w, b, c, theta, B0, h)

 v1.0.0 : JYL @PSI, February 26, 2013

 phase angle is in degrees
 `theta` is in degrees
    '''

    w = Parameter(min=0., max=1.)
    b = Parameter(min=0., max=1.)
    c = Parameter(min=-1., max=1.)
    theta = Parameter(min=0., max=60.)
    B0 = Parameter(min=0.,max=5.)
    h = Parameter(min=0.)


    @staticmethod
    def evaluate(pha, w, b, c, theta, B0, h):
        return phasefunc(pha, {'w': w, 'g': (b, c), 'theta': theta, 'shoe': (B0, h)}, normalize=None)


class DiskInt6Log(DiskInt6):
    @staticmethod
    def evaluate(*args):
        return np.log10(DiskInt6.evaluate(*args))


class Hapke5P(PhotometricModel):

    w = Parameter(default=0.1, min=0., max=1.)
    g = Parameter(default=-0.3, min=-1., max=1.)
    theta = Parameter(default=20., min=0., max=60.)
    B0 = Parameter(default=0.1, min=0.)
    h = Parameter(default=0.01, min=0.)

    @staticmethod
    def evaluate(i, e, a, w, g, theta, B0, h):
        return bdr((i, e, a), {'w':w, 'g': g, 'shoe': (B0, h), 'theta':theta})


class Hapke6P(PhotometricModel):
    w = Parameter(default=0.1, min=0., max=1.)
    b = Parameter(default=-0.3, min=-1., max=1.)
    c = Parameter(default=0.2, min=-1., max=1.)
    theta = Parameter(default=20., min=0., max=60.)
    B0 = Parameter(default=0.1, min=0.)
    h = Parameter(default=0.01, min=0.)

    @staticmethod
    def evaluate(i, e, a, w, b, c, theta, B0, h):
        return bdr((i, e, a), {'w': w, 'g': (b, c), 'shoe': (B0, h), 'theta': theta})


def fitDiskInt5(alpha, measure, error=None, w=0.2, g=-0.3, theta=20., B0=1.0, h=0.01, covar=False, maxiter=1000, log=False, **kwarg):
    '''Fit 5-parameter disk-integrated Hapke model

 Parameters
 ----------
 alpha, measure : array-like, number
   Phase angle [deg] and measured phase function
 error : array-like, number
   Measurement error
 w, g, theta, B0, h : numbers
   Initial model parameters
 covar : bool, optional
   If True, then program returns a tuple, where the second element is
   the covariance matrix of fitted parameters

 Returns
 -------
 The best fit Hapke.DiskInt5 model class

 v1.0.0 : JYL @PSI, October 27, 2014
    '''

    from astropy.modeling.fitting import LevMarLSQFitter
    from ..core import MPFitter

    alpha = np.asarray(alpha).flatten().copy()
    measure = np.asarray(measure).flatten().copy()
    if error is not None:
        error = np.asarray(error).flatten().copy()
        weights = 1/error
    else:
        weights = None
    if alpha.shape[0] != measure.shape[0]:
        raise RuntimeError('Input parameters must have the same number of elements')
    if error is not None and error.shape[0] != alpha.shape[0]:
        raise RuntimeError('Error array must have the same number of element as data')

    parms = {'w': w, 'g': g, 'theta': theta, 'B0': B0, 'h': h}
    fixed = kwarg.pop('fixed', None)
    if fixed is not None:
        parms['fixed'] = fixed
    if log:
        model_class = DiskInt5Log
        measure = np.log10(measure)
    else:
        model_class = DiskInt5
    model0 = model_class(**parms)

    f = MPFitter()
    #f = MPFitter()
    model = f(model0, alpha, measure, weights=weights, maxiter=maxiter, **kwarg)

    if covar:
        if weights is None:
            chisq = np.sum((model(alpha)-measure)**2)/(len(alpha)-5)
        else:
            chisq = np.sum(((model(alpha)-measure)/error)**2)/(len(alpha)-5)
        fit_info = f.fit_info.copy()
        fit_info['red_chisq'] = chisq
        if fit_info['param_cov'] is None:
            print(fit_info['ierr'], fit_info['message'])
        return model, fit_info

    return model


def fitDiskInt6(alpha, measure, error=None, w=0.2, b=0.3, c=0.4, theta=20., B0=1.0, h=0.01, covar=False, maxiter=1000, log=False, **kwarg):
    '''Fit 6-parameter disk-integrated Hapke model

 Parameters
 ----------
 alpha, measure : array-like, number
   Phase angle [deg] and measured phase function
 error : array-like, number
   Measurement error
 w, g, theta, B0, h : numbers
   Initial model parameters
 covar : bool, optional
   If True, then program returns a tuple, where the second element is
   the covariance matrix of fitted parameters

 Returns
 -------
 The best fit Hapke.DiskInt5 model class

 v1.0.0 : JYL @PSI, October 27, 2014
    '''

    from astropy.modeling.fitting import LevMarLSQFitter
    from ..core import MPFitter

    alpha = np.asarray(alpha).flatten().copy()
    measure = np.asarray(measure).flatten().copy()
    if error is not None:
        error = np.asarray(error).flatten().copy()
        weights = 1/error
    else:
        weights = None
    if alpha.shape[0] != measure.shape[0]:
        raise RuntimeError('Input parameters must have the same number of elements')
    if error is not None and error.shape[0] != alpha.shape[0]:
        raise RuntimeError('Error array must have the same number of element as data')

    parms = {'w': w, 'b': b, 'c': c, 'theta': theta, 'B0': B0, 'h': h}
    fixed = kwarg.pop('fixed', None)
    if fixed is not None:
        parms['fixed'] = fixed
    if log:
        model_class = DiskInt6Log
        measure = np.log10(measure)
    else:
        model_class = DiskInt6
    model0 = model_class(**parms)

    #f = LevMarLSQFitter()
    f = MPFitter()
    model = f(model0, alpha, measure, weights=weights, maxiter=maxiter, **kwarg)

    if covar:
        if weights is None:
            chisq = np.sum((model(alpha)-measure)**2)/(len(alpha)-5)
        else:
            chisq = np.sum(((model(alpha)-measure)/error)**2)/(len(alpha)-5)
        fit_info = f.fit_info.copy()
        fit_info['red_chisq'] = chisq
        if fit_info['param_cov'] is None:
            print(fit_info['ierr'], fit_info['message'])
        return model, fit_info

    return model


def create_pvl(par, outfile, **kwarg):
    '''Generate PVL file

 par : dict
   Hapke parameters
 outfile : str, file
   Output file
 **kwarg : ISIS photomet parameters, case insensitive

 v1.0.0 : 11/3/2015, JYL @PSI
    '''

    incref = kwarg.pop('incref', 30.0)
    incmat = kwarg.pop('incmat', 89.999)
    thresh = kwarg.pop('thresh', 1000.)
    emaref = kwarg.pop('emaref', 0.0)
    pharef = kwarg.pop('pharef', 30.0)
    albedo = kwarg.pop('albedo', 1.)
    zerob0standard = kwarg.pop('zerob0standard', 'False')
    comment = kwarg.pop('comment', None)

    if hasattr(par['g'], '__iter__'):
        if len(par['g']) > 2:
            raise ValueError('invalid phase function parameters')
        g1, g2 = par['g']
        g2 = (g2+1.)/2.
    else:
        g1, g2 = par['g'], 0

    f = open(outfile,'w')
    f.write('# Created by create_pvl at {0}\n'.format(Time.now().isot))
    if comment is not None:
        if (not isinstance(comment, (str,bytes))) and hasattr(comment, '__iter__'):
            for c in comment:
                f.write('# {0}\n'.format(c))
        else:
            f.write('# {0}\n'.format(comment))
    f.write('\n')
    f.write('Object = NormalizationModel\n')
    f.write('  Group = Algorithm\n')
    f.write('    Name = Albedo\n')
    f.write('    PhotoModel = HapkeHen\n')
    f.write('    Incref = {0}\n'.format(incref))
    f.write('    Incmat = {0}\n'.format(incmat))
    f.write('    Thresh = {0}\n'.format(thresh))
    f.write('    Emaref = {0}\n'.format(emaref))
    f.write('    Pharef = {0}\n'.format(pharef))
    f.write('    Albedo = {0}\n'.format(albedo))
    f.write('  EndGroup\n')
    f.write('EndObject\n')
    f.write('Object = PhotometricModel\n')
    f.write('  Group = Algorithm\n')
    f.write('    Name = HapkeHen\n')
    f.write('    ZeroB0Standard = {0}\n'.format(zerob0standard))
    for k, v in list(kwarg.items()):
        if isinstance(v, str):
            line = '    {0} = "{1}"\n'
        else:
            line = '    {0} = {1}\n'
        f.write(line.format(k,v))
    f.write('    Theta = {0}\n'.format(par['theta']))
    f.write('    WH = {0}\n'.format(par['w']))
    f.write('    B0 = {0}\n'.format(par['shoe'][0]))
    f.write('    HH = {0}\n'.format(par['shoe'][1]))
    f.write('    HG1 = {0}\n'.format(g1))
    f.write('    HG2 = {0}\n'.format(g2))
    f.write('  EndGroup\n')
    f.write('EndObject\n')
    f.close()



class DiskInt():
    """Disk-integrated phase function class for a sphere
    """
    def __init__(self, par, obs_dist=np.inf, light_dist=np.inf):
        """
        par: dict
            Hapke model parameters
        obs_dist, light_dist: number, optional
            The observer distance and light source distance to body center
        """
        self.par = copy.deepcopy(par)
        self.obs_dist = obs_dist
        self.light_dist = light_dist

    def __call__(self, obj, pha, normalized=False, return_all=False, **kwarg):
        """Calculate disk-integrated phase function of a sphere numerically
        obj: object mesh class
            The object for which the phase function is calculated.  It must
            have a method `.backplanes` that returns a tuple of
            (inc_map, emi_map, pha_map, mask)
        pha: number or array_like of number
            Phase angles in rad
        normalized: bool, optional
            Returned phase function normalized or not
        **kwarg: optional keywords
            see `mesh.Sphere` for **kwarg

        Return: ndarray of number
            Phase function at `pha`
        """
        if hasattr(pha, '__iter__'):
            out = np.zeros_like(pha)
            if return_all:
                imaps = []
                emaps = []
                amaps = []
                masks = []
                imgs = []
            for i,p in enumerate(pha):
                tmp = copy.copy(self)(obj, p, return_all=return_all, **kwarg)
                if return_all:
                    out[i] = tmp[0]
                    imaps.append(tmp[1])
                    emaps.append(tmp[2])
                    amaps.append(tmp[3])
                    masks.append(tmp[4])
                    imgs.append(tmp[5])
                else:
                   out[i] = tmp
            if return_all:
                return out, imaps, emaps, amaps, masks, imgs
            else:
                return out
        else:
            isz = kwarg.pop('isz', 300)
            imap, emap, amap, mask = obj.backplanes(pha, obs_dist=self.obs_dist, light_dist=self.light_dist, isz=isz)
            img = np.zeros_like(imap)
            vis_ill = mask == 2
            sca = np.rad2deg(imap[vis_ill]), np.rad2deg(emap[vis_ill]), np.rad2deg(amap[vis_ill])
            img[vis_ill] = bdr(sca, self.par, **kwarg)
            xsec = len(np.where(mask != 0)[0])
            out = img.sum()/xsec
            if return_all:
                return out, imap, emap, amap, mask, img
            else:
                return out
