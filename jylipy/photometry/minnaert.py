'''Implimentation of Minnaert model

Module dependency

History
10/21/213, started by JYL @PSI
'''

import numpy as np
from .core import DiskFunction, PhotometricModel, LinMagnitude
from astropy.modeling import FittableModel, Parameter, Fittable1DModel, Fittable2DModel
from astropy.modeling.functional_models import Linear1D


class MinnaertModel(DiskFunction):
    '''Minnaert disk-function model
      r = A * mu0**k * mu**(k-1)

     v1.0.0 : JYL @PSI, Feb 24, 2015
    '''

    A = Parameter(default=1, min=0.)
    k = Parameter(default=0.5, min=0., max=1.)

    @staticmethod
    def evaluate(i, e, A, k):
        return A*np.cos(i)**k*np.cos(e)**(k-1)

    @staticmethod
    def fit_deriv(i, e, A, k):
        mu0 = np.cos(i)
        mu = np.cos(e)
        ddA = mu0**k*mu**(k-1)
        ddk = A*ddA*(np.log(mu0)+np.log(mu))
        return [ddA, ddk]


class kLinear(Fittable1DModel):
    '''Linear model for Minnaert k'''

    k0 = Parameter(default=0.5, min=0.)
    b = Parameter(default=0.)
    linear = True

    @staticmethod
    def evaluate(x, k0, b):
        if hasattr(x, 'unit'):
            x = x.to('deg').value
        return k0+x*b

    @staticmethod
    def fit_deriv(x, k0, b):
        if hasattr(x, 'unit'):
            x = x.to('deg').value
        d_b = x
        d_k0 = np.ones_like(x)
        return [d_b, d_k0]


class MinnLinMag(PhotometricModel):
    '''Minnaert model with a linear magnitude phase function

 Model equations:

   r(i, e, a) = 1/pi * A(a) * mu0**k(a) * mu**(k(a)-1) * f(a)
   A(a) = A0 * 10 ** (0.4 * beta * a)
   k(a) = k0 + b * a

   mu0 = cos(i)
   mu = cos(e)

 Model parameters:

   A0 : Minnaert albedo at 0 phase angle
   beta : Phase slope (mag/deg)
   k0 : Minnaert k at 0 phase angle
   b : Slope of Minnaert k w/r to phase angle (1/deg)

 v1.0.0 : 12/8/2015, JYL @PSI
    '''

    A0 = Parameter(default=1., min=0.)
    beta = Parameter(default=0.04, min=0.)
    k0 = Parameter(default=0.5, min=0., max=1.)
    b = Parameter(default=0.)

    @staticmethod
    def evaluate(i, e, a, A0, beta, k0, b):
        A = LinMag(A0, beta)(a)
        k = kLinear(k0, b)(a)
        return MinnaertModel(A, k)(i, e)/np.pi

    def geoalb(self):
        '''Geometric albedo'''
        return self.A0.value*2/(2*self.k0.value+1)


class MinnPoly3(PhotometricModel):
    '''Minnaert model + 3rd order polynomial magnitude phase function'''

    A = Parameter(default=0.2, min=0.)
    beta = Parameter(default=0.02,min=0.)
    gamma = Parameter(default=1e-4)
    delta = Parameter(default=1e-8)
    k0 = Parameter(default=0.5, min=0., max=1.)
    b = Parameter(default=0.004)

    @staticmethod
    def evaluate(i, e, a, A, beta, gamma, delta, k0, b):
        k = k0+b*a
        return Minnaert(A,k)(i, e)*Poly3Mag(beta, gamma, delta)(a)/np.pi


def minnaert(p, inc, emi, pha):
    mu0 = np.cos(np.deg2rad(inc))
    mu = np.cos(np.deg2rad(emi))
    f = 10**(-0.4*(p[1]*pha+p[2]*pha**2+p[3]*pha**3))
    k = p[4]+p[5]*pha
    d = mu0**k*mu**(k-1)
    return p[0]*f*d


class MinnaertModel0(Fittable2DModel):
    '''
 Minnaert model class
    '''

    k = Parameter('k')
    A = Parameter('A')
    linear = False

    def __init__(self, k, A, **constraints):
        '''
     Parameters
     ----------
     k, A : array-like, floating points
       Minnaert k and A parameter.
       reflectance = A * cos(i)^k * cos(e)^(k-1)
        '''
        super(Minnaert, self).__init__(k=k, A=A, **constraints)


    def ref(i, e):
        '''
     Calculate Minnaert reflectance

     Input
     -----
     i, e : array-like, floating point
       Incidence angle and Emission angle [deg]

     Output
     ------
     numpy array, floating point

     Notes
     -----
     1. If k and A are floating point numbers, then each element in the
       returned reflectance array corresponds to each element of the input
       pair (i, e)
     2. If i, e, k, and A are all array-like, they need to have the same
       length.  In this case, each element in the returned reflectance
       corresponds to each group of (i, e, k, A)

     History
     -------
     10/21/2013 created by JYL @PSI
        '''
        i1, e1, k1, A1 = ut.nparr(i, e, self.k.value, self.A.value)
        return A1*np.cos(i1*np.pi/180.)**k1 * np.cos(e1*np.pi/180)**(k1-1)


    def radf(i, e):
        return ref(i, e)*np.pi


    def reff(i, e):
        i1, e1, k1, A1 = ut.nparr(i, e, self.k.value, self.A.value)
        return A1*(np.cos(i1*np.pi/180.)*np.cos(e1*np.pi/180))**(k1-1)*np.pi


    def brdf(i, e):
        i1, e1, k1, A1 = ut.nparr(i, e, self.k.value, self.A.value)
        return A1*(np.cos(i1*np.pi/180.)*np.cos(e1*np.pi/180))**(k1-1)


    @staticmethod
    def evaluate(i, e, k, A):
        '''Minnaert model function'''
        return A*np.cos(i*np.pi/180)**k*np.cos(e*np.pi/180)**(k-1)*np.pi


    @staticmethod
    def fit_deriv(i, e, k, A):
        '''Minnaert model function derivative'''
        mu0, mu = np.cos(i*np.pi/180), np.cos(e*np.pi/180)
        dA = mu0**k * mu**(k-1)
        dk = A*dA*(np.log(mu0)+np.log(mu))
        return [dk, dA]


def ref(i, e, k, A):
    '''
 Calculate Minnaert reflectance

 Parameters
 ----------
 i, e : array-like, floating point
   Incidence angle and Emission angle [deg]
 k, A : array-like, floating points
   Minnaert k and A parameter.
   reflectance = A * cos(i)^k * cos(e)^(k-1)

 Returns
 -------
 numpy array, floating point

 Notes
 -----
 1. If k and A are floating point numbers, then each element in the
   returned reflectance array corresponds to each element of the input
   pair (i, e)
 2. If i, e, k, and A are all array-like, they need to have the same
   length.  In this case, each element in the returned reflectance
   corresponds to each group of (i, e, k, A)

 History
 -------
 10/21/2013 created by JYL @PSI
    '''
    i1, e1, k1, A1 = ut.nparr(i, e, k, A)
    return A1*np.cos(i1*np.pi/180.)**k1 * np.cos(e1*np.pi/180)**(k1-1)


def hillier_phase(pha, par):
    '''
 Calculate the Hillier empirical phase function model values

 Input
 -----
 pha : array-like, floating point
   Phase angles [deg]
 par : array-like, floating point
   Hillier phase function model parameters, in the order of
   (C0, C1, A0, A1, A2, A3, A4)

 Output
 ------
 array-like, floating point

 Notes
 -----
 Hillier empirical phase function model:
  f(pha) = C0 * exp(-C1 * pha) + A0 + A1*pha + A1*pha**2 +
           A3*pha**3 + A4*pha**4
 See Hillier, Buratti, and Hill (1999).  Icarus 141, 205-225.
    '''

    c0, c1, a0, a1, a2, a3, a4 = ut.nparr(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7])
    pha2 = pha*pha
    return c0*np.exp(-c1*pha)+a0+a1*pha+a2*pha2+a3*pha*pha2+a4*pha2*pha2


def maglin_phase(pha, beta, f0=1.):
    '''
 Calculate the magnitude-linear phase function value

 Input
 -----
 pha : array-like, floating point
   Phase angles [deg]
 beta : array-like, floating point
   Phase coefficient [mag/deg]
 f0 : array-like, floating point, optional
   Phase function value at zero phase angle

 Output
 ------
 array-like, floating point

 Notes
 -----
 If both pha and beta are array-like, then they have to have the same
 number of elements, and the results will correspond to each pair of
 (pha, beta).

 History
 -------
 10/21/2013, created by JYL @PSI
    '''

    pha, beta = ut.nparr(pha, beta)
    return f0*10**(-beta*pha*0.4)


def bdr(sca, par, phasefunc='maglin', funcname=None):
    '''
 Calculate the bidirectional reflectance with Minnaert model.

 Input
 -----
 sca : tuple, floating point
   Scattering geometry tuple, (inc, emi, pha) in deg
 par : array-like, floating point
   For phasefunc='maglin': four elements, each being array-like
   floating point type.  (k0, b, A0, beta)
     k0 : Minnaert k at zero phase angle
     b : Linear slope of k [1/deg]
     A0 : Minnaert albedo at zero phase angle
     beta : Minnaert albedo phase coefficient [mag/deg]
   For phasefunc='hillier': nine elements, each being array-like
   floating point type.  (k0, b, C0, C1, A0, A1, A2, A3, A4).
 phasefunc : string, optional
   Specify the name of surface phase function.  Can be:
   'maglin' : magnitude linear, where f(pha) = A0*10**(beta*pha*0.4)
   'hillier' : Hillier empirical phase function
   'user' : User specified phase function, as specified by keyword
     funcname
 funcname : function object, optional
   User supplied phase function.  It has to have a definition like
     def funcname(pha, funcpar)
   where pha is phase angle [deg] and funcpar is an array-like
   variable containing parameters.  par[2:] will be passed as funcpar

 Output
 ------
 array-like, floating point
   Returns the Minnaert bidirectional reflectance corresponding to
   each group of scattering geometry (i, e, a)

 Notes
 -----
 The form of Minnaert model is:

   bdr = A(phase) * cos(i)**k(phase) * cos(e)**(k(phase)-1)
   A(phase) = A0 * 10**(beta * phase)
   k(phase) = k0 + b*phase

 See Li et al. (2009).

 History
 -------
 10/21/2013, created by JYL @PSI
    '''

    inc, emi, pha = ut.nparr(sca)
    k0, b = ut.nparr(par[0], par[1])
    k = k0+b*pha
    if phasefunc == 'maglin':
        a = maglin_phase(pha, par[3], f0=par[2])
    elif phasefunc == 'hillier':
        a = hillier_phase(pha, par[2:])
    elif phasefunc == 'user':
        a = funcname(pha, par[2:])

    return ref(inc, emi, k, a)


def radf(sca, par, **kwarg):
    '''
 Calculate radiance factor (RADF).
 See bdr().

 History
 -------
 10/21/2013, created by JYL @PSI
    '''
    return bdr(sca, par, **kwarg)*np.pi


def reff(sca, par, **kwarg):
    '''
 Calculate reflectance factor (REFF).
 See bdr().

 History
 -------
 10/21/2013, created by JYL @PSI
    '''
    mu0 = np.cos(ut.nparr(sca[0])*np.pi/180)
    return bdr(sca, par, **kwarg)*np.pi/mu0


def brdf(sca, par, **kwarg):
    '''
 Calculate bidirectional reflectance distribution function (BRDF).
 See bdr().

 History
 -------
 10/21/2013, created by JYL @PSI
    '''
    mu0 = np.cos(ut.nparr(sca[0])*np.pi/180)
    return bdr(sca, par, **kwarg)/mu0


def fitdisk(inc, emi, iof, err=None):
    '''
 Fit I/F measurement over a disk at a constant phase angle

 Input
 -----
 inc, emi, iof: array-like, float
   Incidence angle, emission angle, both in deg, and the measured
   I/F values.
 err: array-like, float, optional
   Measurement error of I/F

 Output
 ------
 Tuple: (par, sig, chisq)
   par and sig are two 2-element array, containing the best-fit values
     and error bars of (A, k)
   chisq is the reduced chi-square = [(model-measure)/error]**2/(n-2)

 History
 -------
 10/22/2013, created by JYL @PSI
    '''

    inc1, emi1, iof1 = ut.nparr(inc, emi, iof)
    n = len(iof1)
    if err is None:
        err1 = np.repeat(1., n)
    else:
        err1 = ut.nparr(err).astype(float).flatten()

    mu0, mu = np.cos(inc1*np.pi/180), np.cos(emi1*np.pi/180)
    p, s, c = ut.powfit(mu*mu0, mu*iof1, mu*err1, fast=False)
    p[0], s[0] = p[0]/np.pi, s[0]/np.pi
    return p, s, c


def fitiof(sca, iof, err=None, phasefunc='maglin', phasebin=5., plot=False):
    '''
 Fit I/F measurement with Minnaert model

 Input
 -----
 sca : tuple of array-like, float
   Scattering geometry (inc, emi, pha) [deg]
 iof : array-like, float
   Measured I/F
 err : array-like, float, optional
   Measurement error of I/F
 phasefunc : string, optional
   Specify the phase function model
   'maglin' : magnitude linear, where f(pha) = A0*10**(beta*pha*0.4)
   'hillier' : Hillier empirical phase function
 phasebin : float, optional
   The bin size of phase angle.
 plot : bool, or a list of figure class
   Plot the measurement and fit result

 Output
 ------
 dictionary containing the best-fit result
 'par', 'err': tuples of float contain the parameters and error values
   of (k0, b, ...), where k0 and b determines the model for k as
     k = k0+b*pha
   and ... are the parameters for phase function, depending on the
   model adopted.
   For 'maglin', ... = (A0, beta).
   For 'hillier', ... = (C0, C1, A0, A1, A2, A3, A4)
 'chisq': float, reduced chisq defined as
     (((model-measure)/err)**2).sum()/(n-4)
 'plots': list of figure object, if `plot`=True

 History
 -------
 10/21/2013, created by JYL @PSI
    '''

    # prepare parameters
    if type(plot) is not bool:
        import matplotlib
        if not hasattr(plot,'__iter__'):
            plot = [plot]
        for f in plot:
            if type(f) is not matplotlib.figure.Figure:
                raise TypeError('Please pass only matplotlib.figure.Figure class.')
    sca1 = ut.nparr(sca)
    inc1, emi1, pha1 = sca1
    iof1 = ut.nparr(iof)
    n = len(iof1)
    if err is None:
        err1 = np.repeat(1., n)
    else:
        err1 = ut.nparr(err).astype(float).flatten()

    # fit parameters for each phase angle bins
    ph, ks, As, ke, Ae = [], [], [], [], []
    for pl in np.arange(pha1.min(), pha1.max(), phasebin):
        ww = (pha1 >= pl) & (pha1 < pl+phasebin)
        if ww.any():
            p, s, c = fitdisk(inc1[ww], emi1[ww], iof1[ww], err1[ww])
            ph.append(pha1[ww].mean())
            As.append(p[0])
            ks.append(p[1])
            Ae.append(s[0])
            ke.append(s[1])

    # fit phase angle dependence
    As, ks, Ae, ke = np.array(As), np.array(ks), np.array(Ae), np.array(ke)
    apar, asig, c = ut.linfit(ph, np.log10(As), Ae/As)
    apar[0], asig[0] = 10**apar[0], np.log(10)*asig[0]
    apar[1], asig[1] = 2.5*apar[1], 2.5*asig[1]
    kpar, ksig, c = ut.linfit(ph, ks, ke)
    model = radf(sca, np.array([kpar, apar]).flatten())
    res = (model-iof1)/err1
    chisq = (res*res).sum()/(n-4)
    rms = np.sqrt(chisq/(n-4))

    # make plot
    if plot is not False:
        import matplotlib.pyplot as plt

        # prepare figures
        if plot is True:
            figs = [plt.figure(dpi=120,figsize=(6,8.5)), plt.figure(dpi=120,figsize=(6,8.5)), plt.figure(dpi=120,figsize=(6,6))]
        else:
            figs = plot
        for i in range(3-len(figs),0,-1):
            f = plt.figure(dpi=120)
            if i is 1:
                f.set_size_inches(6.5,6.5,forward=True)
            else:
                f.set_size_inches(6.5,9,forward=True)
            figs.append(f)

        # plot phase angle dependence of parameters
        ax1 = figs[0].add_subplot(211)
        ax1.errorbar(ph, As, Ae, fmt='o',ecolor='r')
        ax1.plot(ph, apar[0]*10**(apar[1]*np.array(ph)*0.4),'-k')
        ax1.set_xlabel('Phase Angle [deg]')
        ax1.set_ylabel('Minnaert Albedo')
        ax2 = figs[0].add_subplot(212)
        ax2.errorbar(ph, ks, ke, fmt='o',ecolor='r')
        ax2.plot(ph, kpar[0]+kpar[1]*np.array(ph),'-k')
        ax2.set_xlabel('Phase Angle [deg]')
        ax2.set_ylabel('Minnaert k')

        # plot measure/model ratio
        ratio = iof1/model
        xlbl = ['Incidence Angle [deg]', 'Emission Angle [deg]', 'Phase Angle [deg]']
        for i in range(0,3):
            ax = figs[1].add_subplot(3,1,i+1)
            ax.plot(sca1[i], ratio,'.')
            ax.hlines(1.,sca1[i].min(),sca1[i].max())
            ax.set_xlabel(xlbl[i])
            ax.set_ylabel('Measure/Model')

        # plot model vs. measure
        ax = figs[2].add_subplot(111)
        vmin = min([model.min(),iof1.min()])
        vmax = max([model.max(),iof1.max()])
        ax.plot(iof1,model,'.',[vmin,vmax],[vmin,vmax],'-k')
        ax.set_xlabel('Measured I/F')
        ax.set_ylabel('Modeled I/F')
        plt.show()

    # return results
    fit_results = {'par':np.array([kpar, apar]).flatten(), 'err':np.array([ksig, asig]).flatten(), 'chisq':chisq}
    if plot:
        fit_results['figures'] = figs
    return fit_results


def fitfile(datafile, inclim=None, emilim=None, phalim=None, weighted=False, quiet=False, **kwargs):
    '''
 Fit Minnaert model for the I/F data in datafile

 Input
 -----
 datafile: string
   Name of the FITS file that contains the I/F data table, which needs
   to contain at least four columns, with names "Inc", "Emi", "Pha",
   "IoF".  It can contain "IoFErr" column as the measurement errors
   of I/F data.
 inclim, emilim, phalim: 2-element array-like, optional
   The minimum and maximum values of inc, emi, and pha used to fit.
 weighted: bool, optional
   If True, then use the supplied errors to perform weighted fit.
   Otherwise unweighted fit.
 quiet: bool, optional
   If True, all screen print are supressed.
 keyword parameters taken by `fit()` are all accepted

 Output
 ------
 Same as `fit()`.

 History
 -------
 10/22/2013, created by JYL @PSI
    '''

    # load i/f data
    from astropy.io import fits
    datafile = fits.open(datafile)
    inc = datafile[1].data.field('Inc')
    emi = datafile[1].data.field('Emi')
    pha = datafile[1].data.field('Pha')
    iof = datafile[1].data.field('IoF')
    good = np.ones(len(inc),dtype=bool)
    if inclim != None:
        good = good & (inc>inclim[0]) & (inc<inclim[1])
    if emilim != None:
        good = good & (emi>emilim[0]) & (emi<emilim[1])
    if phalim != None:
        good = good & (pha>phalim[0]) & (pha<phalim[1])
    sca = (inc[good], emi[good], pha[good])
    iof = iof[good]

    if weighted and 'IoFErr' in datafile[1].data.columns.names:
        err = (datafile[1].data.field('IoFErr')[good])[val]
    else:
        err = None
    datafile.close()

    return fitiof(sca, iof, err, **kwargs)

