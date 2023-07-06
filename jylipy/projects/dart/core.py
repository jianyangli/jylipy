import os, numpy as np, astropy.units as u, astropy.constants as const
from warnings import warn
from scipy.signal import savgol_filter
from astropy.time import Time
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter, FittingWithOutlierRemoval
from astropy.io import ascii, fits
from astropy.table import Table, vstack
from astropy.stats import sigma_clip
from scipy.optimize import brentq
from sbpy.bib import cite
import spiceypy as spice
import matplotlib.pyplot as plt
from ...geometry import load_generic_kernels
from ...core import syncsynd, rebin
from ...hst.wfc3 import uvis_pix

# crater size model scaling clases

class Impactor():
    """Impactor class stores the properties of impactor
    """

    @u.quantity_input(m=u.kg, a=u.m, U=u.m/u.s)
    def __init__(self, m, a, U):
        """
        m : Quantity of mass
          Impactor mass
        a : Quatity of length
          Impactor radius
        U : Quantity of velocity
          Impactor velocity
        """
        self.m = m
        self.a = a
        self.U = U

    @property
    def V(self):
        """Impactor volume, assuming a spherical shape"""
        return 4/3*np.pi*self.a*self.a*self.a

    @property
    def delta(self):
        """Impactor density, assuming a spherical shape"""
        return self.m/self.V


class Target():
    """Target class to store the impact target properties
    """

    @u.quantity_input(rho=u.kg/u.m**3, g=u.m/u.s**2, Y=u.Pa)
    def __init__(self, rho, g=None, Y=None):
        """
        rho : Quantity of density
          Density of impact target
        g : Quantity of acceleration
          Surface gravity of target
        Y : Quantity of strength (unit of pressure)
          Strength of target
        """
        self.rho = rho
        self.g = g
        self.Y = Y


class ScalingLaw():
    """Impact scaling law base class
    """

    @cite({'method': '2011Icar..211..856H'})
    def __init__(self, mu, nu, H1=None, H2=None, impactor=None, target=None, \
                               p=None, n1=None, n2=None, C1=None, k=None, \
                               regime=None):
        """
        H, mu, nu : number
            scaling law parameters
        impactor : Impactor class object
        target : Target class object
        p, n1, n2, C1, k : number
            Other scaling constants based on the definition in Housen &
            Holsapple (2011).
        """
        self.mu = mu
        self.nu = nu
        self.H1 = H1
        self.H2 = H2
        self.impactor = impactor
        self.target = target
        self.p = p
        self.n1 = n1
        self.n2 = n2
        self.C1 = C1
        self.k = k
        self._regime = regime

    @property
    def rho_over_m(self):
        if (self.impactor is None) or (self.target is None):
            return np.nan
        return (self.target.rho/self.impactor.m)**(1/3)

    @property
    def density_ratio(self):
        if (self.impactor is None) or (self.target is None):
            return np.nan
        return self.target.rho/self.impactor.delta

    @property
    def gravity_parameter(self):
        if (self.impactor is None) or (self.target is None):
            return np.nan
        return self.target.g*self.impactor.a/(self.impactor.U*self.impactor.U)

    @property
    def strength_parameter(self):
        if (self.impactor is None) or (self.target is None):
            return np.nan
        return self.target.Y/(self.target.rho*self.impactor.U*self.impactor.U)

    @property
    def regime(self):
        """Cratering regime, gravity or strength"""
        if self._regime is None:
            if (self.impactor is None) or (self.target is None) \
                    or (self.H1 is None) or (self.H2 is None) \
                    or (self.mu is None) or (self.nu is None):
                return 'unknown'
            grav = self.target.g * self.impactor.a \
                                        / (self.impactor.U * self.impactor.U)
            strg = (self.H1 / self.H2)**((2+self.mu)/self.mu) \
                    * self.density_ratio**self.nu \
                    * (self.target.Y / (self.target.rho * self.impactor.U \
                        * self.impactor.U))**((2+self.mu)/2)
            if grav > strg:
                return 'gravity'
            else:
                return 'strength'
        else:
            return self._regime

    @u.quantity_input(x=u.m)
    def v(self, x):
        """Velocity of ejecta at radial distance x from the central point of
        impact"""
        if (self.p is None) or (self.n2 is None) or (self.C1 is None):
            return np.nan
        w = x/self.impactor.a
        return (self.C1 * (w * self.density_ratio**self.nu)**(-1/self.mu) \
            * (1 - x/(self.n2*self.R))**self.p).decompose() * self.impactor.U

    @property
    def R_gravity(self):
        """Gravity dominated impact crater radius"""
        e1 = (2 + self.mu - 6*self.nu) / (3 * (2 + self.mu))
        e2 = -self.mu / (2 + self.mu)
        return (self.H1 * self.density_ratio**e1 * self.gravity_parameter**e2 \
                / self.rho_over_m).decompose()

    @property
    def R_strength(self):
        """Strength dominated impact crater radius"""
        e1 = (1 - 3*self.nu)/3
        e2 = -self.mu/2
        return (self.H2 * self.density_ratio**e1 * self.strength_parameter**e2 \
                / self.rho_over_m).decompose()

    @property
    def R(self):
        if self.regime == 'unknown':
            return -1.
        elif self.regime == 'gravity':
            return self.R_gravity
        elif self.regime == 'strength':
            return self.R_strength

    @u.quantity_input(x=u.m)
    def M(self, x):
        """Mass ejected from within x"""
        if (self.k is None) or (self.n1 is None):
            return np.nan
        return (3 * self.k / (4 * np.pi) * self.density_ratio \
            * ((x/self.impactor.a)**3 - self.n1**3) \
            * self.impactor.m).decompose()

    @property
    def M_total(self):
        """Total ejected mass"""
        return 3 * self.k * self.impactor.m / (4*np.pi) * self.density_ratio \
            *((self.n2*self.R/self.impactor.a)**3-self.n1**3)


class SFDModel():
    """Size frequency distribution (SFD) model class

    Model assumes an exponental SFD: n(r) = N * (r/r0)**(-alpha).
    Parameters of the model:
        alpha : exponent, dimensionless
        N : normalization constant, number density for dr at radius `r0`,
            in unit 1/u.m
        r0 : characteristic radius, in unit u.m
    """

    @u.quantity_input(N = 1/u.m, r0=u.m, rho=u.kg/u.m**3)
    def __init__(self, alpha, N=1/u.m, r0=1*u.m, rho=None):
        """
        rho : density of particles, in unit kg/m**3
        """
        self.alpha = alpha
        self.N = N
        self.r0 = r0
        self.rho = rho
        self.cumulative = lambda r: self.N * self.r0 / -self.cumulative_alpha \
            * (r/self.r0)**(-self.cumulative_alpha)
        self.cumulative_mass = lambda r: self.N * self.r0 * self.m0 \
            / -self.mass_cumulative_alpha \
            * (r/self.r0)**(-self.mass_cumulative_alpha)
        self.cumulative_area = lambda r: self.N * self.r0 * self.a0 \
            / -self.area_cumulative_alpha \
            * (r/self.r0)**(-self.area_cumulative_alpha)

    @u.quantity_input(r=u.m)
    def __call__(self, r):
        """SFD for radius `r`"""
        return self.N * (r/self.r0)**(-self.alpha)

    @property
    def cumulative_alpha(self):
        """Exponent parameter for cumulative SFD"""
        return self.alpha-1

    @u.quantity_input(r1=u.m, r2=u.m)
    def N_integrated(self, r1, r2):
        """Integrated number of particles between `r1` and `r2`"""
        return self.cumulative(r2) - self.cumulative(r1)

    @property
    def m0(self):
        """Characteristic mass (for radius r0)"""
        if self.rho is None:
            return np.nan
        return 4/3 * np.pi * self.r0**3 * self.rho

    @property
    def mass_alpha(self):
        """Exponent for mass distribution function"""
        return self.alpha-3

    @u.quantity_input(r=u.m)
    def mass(self, r):
        """Mass distribution function"""
        if self.rho is None:
            return np.nan
        return self.N * self.m0 * (r/self.r0)**(-self.mass_alpha)

    @property
    def mass_cumulative_alpha(self):
        """Exponent for cumulative mass distribution function"""
        return self.mass_alpha-1

    @u.quantity_input(r1=u.m, r2=u.m)
    def mass_integrated(self, r1, r2):
        """Integrated mass between `r1` and `r2`"""
        if self.rho is None:
            return np.nan
        return self.cumulative_mass(r2) - self.cumulative_mass(r1)

    @property
    def a0(self):
        """Characteristic cross-sectional area (for radius r0)"""
        return np.pi * self.r0**2

    @property
    def area_alpha(self):
        """Exponent for cross-sectional area distribution function"""
        return self.alpha-2

    @u.quantity_input(r=u.m)
    def area(self, r):
        """Cross-sectional area distribution function"""
        return self.N * self.a0 * (r/self.r0)**(-self.area_alpha)

    @property
    def area_cumulative_alpha(self):
        """Exponent for cumulative cross-sectional area distribution function"""
        return self.area_alpha-1

    @u.quantity_input(r1=u.m, r2=u.m)
    def area_integrated(self, r1, r2):
        return self.cumulative_area(r2) - self.cumulative_area(r1)

    @u.quantity_input(r1=u.m, r2=u.m)
    def area_mass_ratio(self, r1, r2):
        return 0.75 * self.mass_cumulative_alpha \
                * (r1**(-self.area_cumulative_alpha) \
                                    - r2**(-self.area_cumulative_alpha)) \
                / (self.rho * self.area_cumulative_alpha \
                            * (r1**(-self.mass_cumulative_alpha) \
                                    - r2**(-self.mass_cumulative_alpha)))


class Didymos():
    """Didymos properties from DRA V2.22"""
    Dp = 780 * u.m
    Ds = 164 * u.m
    rho = 2170 * u.kg/u.m**3
    H = 18.16 * u.mag   # absolute magnitude
    G = 0.20   # IAU HG model G parameter
    Ageo = 0.15   # geometric albedo
    dist = 1.19 * u.km  # distance between the center of primary and secondary

    @property
    def A(self):
        """Total cross-sectional area"""
        return np.pi*(self.Dp/2)**2 + np.pi*(self.Ds/2)**2

    @property
    def Vp(self):
        return 4/3*np.pi*(self.Dp/2)**3

    @property
    def Vs(self):
        return 4/3*np.pi*(self.Ds/2)**3

    @property
    def Mp(self):
        return self.Vp*self.rho

    @property
    def Ms(self):
        return self.Vs*self.rho


didy = Didymos()


@u.quantity_input(r=u.m)
def ejecta_g(r):
    """
    Gravity field at distance `r` from Dimorphos's surface in the direction
    along the line connecting both objects and away from Dimorphos.
    """
    gp = (const.G * didy.Mp / (r+didy.dist+didy.Ds/2)**2).decompose()
    gs = (const.G * didy.Ms / (r+didy.Ds/2)**2).decompose()
    return -(gp+gs).decompose()

@u.quantity_input(r=u.m)
def ejecta_vesc(r):
    """
    Escape velocity at distance `r` from Dimorphos's surface in the direction
    along the line connecting both objects away from Dimorphos.
    """
    Ep = const.G * didy.Mp / (r+didy.dist+didy.Ds/2)
    Es = const.G * didy.Ms / (r+didy.Ds/2)
    return np.sqrt(2*(Ep+Es)).decompose()


# Solver class for 1D motion equation in gravity field
class MotionInGravity1DSolver():
    """Solve motion equation in gravity based on initial conditinon

    1D case implemented for now
    """

    @u.quantity_input(r0=u.m, v0=u.m/u.s)
    def __init__(self, g, r0, v0):
        """
        g : function
            The gravitational acceleration function.  g(r) returnes
            gravitational acceleration.  g(r) needs to take astropy Quantity
            as input and returns the acceleration as a Quantity
        r0, v0 : Quantity
            Initial position r0 and velocity v0
        """
        self.g = g
        self.r0 = [r0, v0]
        self._rmax = None
        self._tmax = None

    @staticmethod
    def _model(x, t, g):
        """
        Motion equation evaluation
        """
        r = x[0]
        v = x[1]
        xdot = [[], []]
        xdot[0] = v
        xdot[1] = g(r*u.m).to('m/s2').value
        return xdot

    @u.quantity_input(t=u.s)
    def solve(self, t, **kwargs):
        from scipy.integrate import odeint
        _ = kwargs.pop('args', None)
        _ = kwargs.pop('tfirst', None)
        r0 = [self.r0[0].to('m').value, self.r0[1].to('m/s').value]
        st = odeint(MotionInGravity1DSolver._model, r0, t.to('s').value,
                    args=(self.g,), **kwargs)
        r = st[:,0]*u.m
        v = st[:,1]*u.m/u.s
        return [r, v]


class DARTEjectaMotion(MotionInGravity1DSolver, Didymos):
    """DART ejecta motion solver"""

    @u.quantity_input(r0=u.m, v0=u.m/u.s)
    def __init__(self, r0, v0):
        MotionInGravity1DSolver.__init__(self, ejecta_g, r0, v0)

    @property
    def rmax(self):
        """Maximum distance particle will move"""
        if self._rmax is None:
            if self.r0[1] >= ejecta_vesc(self.r0[0]):
                self._rmax = np.inf*u.m
            else:
                v0 = self.r0[1]
                import astropy.constants as c
                Ds2 = self.Ds/2
                V = v0*v0/(2*const.G)
                E0 = self.Mp/(self.dist+Ds2)+self.Ms/Ds2
                a = V-E0
                b = (V-E0)*(self.dist+2*Ds2) + self.Mp + self.Ms
                c = (V-E0)*Ds2*(self.dist+Ds2) + self.Mp*Ds2 + \
                        self.Ms*(self.dist+Ds2)
                self._rmax = ((-np.sqrt(b**2-4*a*c) - b) / (2 * a)).decompose()
        return self._rmax

    @property
    def tmax(self):
        """Maximum time a particle will move before fall back to the original
        distance"""
        if self._tmax is None:
            rmax = self.rmax
            if rmax == np.inf*u.m:
                self._tmax = np.inf*u.s
            else:
                v0 = self.r0[1]
                t = rmax/v0
                ts = np.concatenate([[0], np.linspace(t, t*2, 1000)])
                rs = self.solve(ts)[0]
                while (rs.max()<rmax*(1-0.00001)):
                    ts = np.concatenate([[0], np.linspace(ts[1:].max(),
                                         ts[1:].max()*2, 1000)])
                    rs = self.solve(ts)[0]
                from scipy.interpolate import interp1d
                tr = interp1d(rs.to('m').value, ts.to('s').value,
                              fill_value='extrapolate')
                self._tmax = tr(rmax)*2*u.s
        return self._tmax

    @u.quantity_input(t=u.s)
    def robust_solve(self, t, **kwargs):
        """Solve the motion equation and set returned value to NAN after
        particle returns to initial position"""
        r = np.zeros_like(t.value)*u.m*np.nan
        v = np.zeros_like(t.value)*u.m/u.s*np.nan
        ww = t<self.tmax
        r[ww], v[ww] = self.solve(t[ww])
        return [r, v]


didy_spk = os.path.join(os.path.sep, 'Users', 'jyli', 'Work', 'Dart',
                        'spice', 'spk',
                        'didymos_19500101-20501231_20220926_horizons.bsp')


class Beta(u.SpecificTypeQuantity):
    """Solar radiation pressure beta.
    """

    _equivalent_unit = u.dimensionless_unscaled
    _include_easy_conversion_members = False

    def __new__(cls, value, unit=None, **kwargs):
        unit = kwargs.pop('unit', u.dimensionless_unscaled)
        return super().__new__(cls, value, unit=unit, **kwargs)

    @classmethod
    @cite({'method': '1979Icar...40....1B'})
    @u.quantity_input(r=u.m, rho=u.kg/u.m**3)
    def from_radius(cls, r, rho=1000*u.kg/u.m**3, Qpr=1):
        """Initialize from particle radius

        r : `astropy.units.Quantity`
            Particle radius
        rho : `astropy.units.Quantity`
            Particle density
        Qpr : float
            Solar radiation pressure coefficient,
                Qpr = Q_abs + Q_sca * (1 - <cos(alpha)>)
        """
        return cls(5.7e-5 * Qpr / (rho.to_value(u.g/u.cm**3) *
                    r.to_value('cm')), u.dimensionless_unscaled)

    @cite({'method': '1979Icar...40....1B'})
    @u.quantity_input(rho=u.kg/u.m**3)
    def radius(self, rho=1000*u.kg/u.m**3, Qpr=1):
        """Radius of spherical particle corresponding to the beta

        rho : `astropy.units.Quantity`
            Density of particles
        Qpr : float
            Solar radiation pressure coefficient,
                Qpr = Q_abs + Q_sca * (1 - <cos(alpha)>)
        """
        return u.Quantity(5.7e-5 * Qpr / (rho.to_value(u.g/u.cm**3)
                    * self.value), u.cm)

    @staticmethod
    @u.quantity_input()
    def acc_solar(rh: u.au) -> u.m / u.s**2:
        """Solar gravity acceleration

        Positive means acceleration towards the Sun.

        rh : `astropy.units.Quantity`
            Heliocentric distance
        """
        return (const.G * const.M_sun / rh**2)

    @u.quantity_input()
    def acc_srp(self, rh: u.au) -> u.m / u.s**2:
        """Solar radiation pressure acceleration of particles

        Positive value means acceleration towards the Sun.

        rh : `astropy.units.Quantity`
            Heliocentric distance
        """
        return -self.acc_solar(rh) * self

    @u.quantity_input()
    def acc(self, rh: u.au) -> u.m / u.s**2:
        """Total acceleration of particle

        Positive value means acceleration towards the Sun.

        rh : `astropy.units.Quantity`
            Heiocentric distance
        """
        return self.acc_solar(rh) * (1 - self)


class DidySynchroneSyndyne():
    """Synchron syndyne model for DART ejecta"""

    target = '2065803'   # Didymos
    impact_utc = '2022-09-26T23:14:24.183'  # impact time
    spk = didy_spk  # Didymos SPK

    def __init__(self):
        self.impact_time = Time(self.impact_utc)

    def _load_spice_kernels(self):
        load_generic_kernels()
        spice.furnsh(self.spk)

    def _unload_spice_kernels(self):
        spice.kclear()

    def __call__(self, obs_utc, beta, nt=100, **kwargs):
        """Calculate synchron and syndynes

        obs_utc : str
            Observation time in UTC
        beta : float array
            Beta of particles to be calculated
        nt : float, optinal
            Number of time steps in the calculation
        kwargs : optional parameters for `jylipy.syncsynd`
        """
        self.obs_time = Time(obs_utc)
        self.beta = beta
        self.dt = (self.obs_time - self.impact_time).to_value('d') * \
                    np.linspace(0, 1, nt)
        self._load_spice_kernels()
        self.syncsynd, self.target_pos = syncsynd(self.target,
                                    self.obs_time.isot, beta, self.dt, **kwargs)
        self._unload_spice_kernels()
        return self.syncsynd, self.target_pos

    @property
    def relative_syncsynd(self):
        """Synchrones and syndynes coordinates relative to target"""
        return self.syncsynd - self.target_pos

    def plot(self, beta_sample=None, time_sample=None, ax=None, axis_scale=1,
             axis_offset=0, **kwargs):
        """Plot syncrhones and syndynes

        beta_sample : int, slice, list, optional
            The index of beta samples to be plotted.  Default is to plot all.
        time_sample : int, slice, list, optional
            The index of time samples to be plotted.  Default is to plot all.
        ax : `matplotlib.pyplot.axis`
            Axis to plot
        axis_scale : float or [x, y]
            Scaling factor(s) for coordinates.  Can be used to convert the
            default unit of coordinate in RA, Dec to other units, such as
            arcsec or km
        axis_offset : float or [x, y]
            Offset(s) of coordinates.  Can be used to shift the coordinate of
            synchrones and syndynes
        **kwargs : keyword parameters for `matplotlib.pyplot.plot`
        """
        if beta_sample is None and time_sample is None:
            return None

        s = self.relative_syncsynd * axis_scale + axis_offset

        if beta_sample == 'all':
            beta_sample = range(len(self.beta))
        if time_sample == 'all':
            time_sample = range(len(self.dt))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        if beta_sample is not None:
            for i in beta_sample:
                ax.plot(s[i, :, 0], s[i, :, 1], **kwargs)
        if time_sample is not None:
            for i in time_sample:
                ax.plot(s[:, i, 0], s[:, i, 1], **kwargs)

        return ax

    def plot_syndyne(self, beta_sample='all', **kwargs):
        """Plot syncrhones and syndynes

        beta_samples : int, slice, list, optional
            The index of beta samples to be plotted.  Default is to plot all.
        ax : `matplotlib.pyplot.axis`
            Axis to plot
        axis_scale : float
            Scaling factor for coordinate.  Can be used to convert the
            default unit of coordinate in RA, Dec to other units, such as
            arcsec or km
        axis_offset : float
            Offset of coordinate.  Can be used to shift the coordinate of
            synchrones and syndynes
        **kwargs : keyword parameters for `matplotlib.pyplot.plot`
        """
        return self.plot(beta_sample=beta_sample, **kwargs)

    def plot_synchrone(self, time_sample='all', **kwargs):
        """Plot syncrhones and syndynes

        time_samples : int, slice, list, optional
            The index of time samples to be plotted.  Default is to plot all.
        ax : `matplotlib.pyplot.axis`
            Axis to plot
        axis_scale : float
            Scaling factor for coordinate.  Can be used to convert the
            default unit of coordinate in RA, Dec to other units, such as
            arcsec or km
        axis_offset : float
            Offset of coordinate.  Can be used to shift the coordinate of
            synchrones and syndynes
        **kwargs : keyword parameters for `matplotlib.pyplot.plot`
        """
        return self.plot(time_sample=time_sample, **kwargs)


class DistanceModel(Fittable2DModel):
    """Particle motion model under SRP using linear approximation.

    In close distance to the parent body, and within a small duration of
    time compared to the parent body orbital period, the distance that
    a particle moves under SRP for a specific beta and at a specific time
    is approximately linear to beta, and linear to time squared.

        Distance = D0 * beta ** beta_exponential * time ** time_exponential

    Under the approximation of constant acceleration motion along a straight
    line, the exponential of beta is 1, and the exponential of time is to 2.

    Model Parameters
    ----------------
    d0 : Unit distance, the distance of a particle with beta=1 traveling
         in 1 time unit
    beta_exp : beta exponential, should be close to 1
    time_exp : time exponential, should be close to 2

    The model is dimensionless as defined.  When use it, the unit of d0,
    beta, and time should be consisent with what are used to fit the model.
    """

    d0 = Parameter(name='Unit Distance', default=4)
    beta_exp = Parameter(name='Beta Slope', default=1)
    time_exp = Parameter(name='Time Slope', default=2)

    @staticmethod
    def evaluate(beta, time, d0, beta_exp, time_exp):
        return d0 * beta**beta_exp * time**time_exp

    @staticmethod
    def fit_deriv(beta, time, d0, beta_exp, time_exp):
        v = DistanceModel.evaluate(beta, time, d0, beta_exp, time_exp)
        dd0 = v / d0
        dbe = v * np.log(beta)
        dte = v * np.log(time)
        return [dd0, dbe, dte]


class BetaModel(DistanceModel):
    """Inverse of `DistanceModel`.

    This model takes time and distance as input, and calculates beta.

    NOTE: fit_deriv is not set up.  This model CANNOT be used for modeling
    fitting.

    """

    @staticmethod
    def evaluate(time, dist, d0, beta_exp, time_exp):
        return (dist / (d0 * time**time_exp))**(1 / beta_exp)


def mag2xsec(dmag, magerr=None):
    """Calcluate total cross-section from delta-magnitude

    Assume the dust has the same albedo and phase function as Didymos,
    the calculation scales from the cross-sectional area of
    Didymos-Dimorphos system.
    """
    didy = Didymos()
    area_didy = 0.25 * np.pi * (didy.Dp**2 + didy.Ds**2)

    area_dust = (10**(-0.4 * dmag) - 1) * area_didy
    if magerr is None:
        return area_dust
    else:
        area_dust_err = (10**(-0.4 * magerr) - 1) * area_dust
        return area_dust, area_dust_err


def show_stacked(ds9, info):
    """Display all long exposure stacked images

    ds9 : DS9
        DS9 instance to display images
    info : str
        Info file for long exposure stacks
    """
    info = ascii.read(info)
    ds9.set('frame delete all')
    ds9.imdisp([f.replace('flux', 'flux_clean') for f in info['file']])
    ds9.set('lock scalelimits')
    ds9.set('lock scale')
    ds9.set('lock colorbar')
    ds9.set('lock frame image')
    ds9.set('scale log')
    ds9.set('scale limits 0 0.01')
    ds9.set('cmap cool')


def show_stacked_long(ds9,
    info='/Users/jyli/Work/DART/HSTGO16674/morphology/stacking/info_long.csv'):
    show_stacked(ds9, info)


def show_stacked_short(ds9,
    info='/Users/jyli/Work/DART/HSTGO16674/morphology/stacking/info_short.csv'):
    show_stacked(ds9, info)


def mean_angle(angles, axis=None):
    """Calculate the mean of angles"""
    cos = np.cos(u.Quantity(angles, u.deg))
    sin = np.sin(u.Quantity(angles, u.deg))
    out = np.arctan2(sin.mean(axis=axis), cos.mean(axis=axis))
    if isinstance(angles, u.Quantity):
        return ((out + 360 * u.deg) % (360 * u.deg)).to(angles.unit)
    else:
        return (out.to_value('deg') + 360) % 360


@u.quantity_input(delta=u.km)
def angular_distance(delta=None):
    """Equivalencies to convert between UVIS pixels, arcsec, and physical
    distances.

    delta : u.Quantity, optional
        Distance of target.  If not supplied, then only equivalency between
        UVIS pixels and angular size is provided.
    """
    equiv = uvis_pix()
    if delta is not None:
        equiv.append((u.arcsec,
                 u.km,
                 lambda angle: angle / 206265 * delta.to_value('km'),
                 lambda dist: dist / delta.to_value('km') * 206265))
        equiv.append((u.pix,
                  u.km,
                  lambda pix: pix * 0.04 / 206265 * delta.to_value('km'),
                  lambda dist: dist / delta.to_value('km') * 206265 / 0.04))
    return equiv


class BrightnessProfile():
    """Class to process tail brightness profile"""

    def __init__(self, prof, x=None, info=None):
        """
        prof : 1d u.Quantity array
            Tail brightness profile.
        x : 1d array of numbers or u.Quantity, optional
            The x-axis value of the tail profile.  Default is a sequence
            of the same length of `prof`, starting from 0 with an incremental
            of 1, with a unit of pixels
        info : dict, optional
            Information dictionary.
        """
        self._data = prof
        if x is None:
            x = np.arange(len(self._data))
        self._x = x
        self.info = info

    @property
    def data(self):
        return self._data

    @property
    def x(self):
        return self._x

    def x_in(self, unit):
        """Return x in the specified unit"""
        if not isinstance(self.x, u.Quantity):
            warn('x is not a `u.Quantity`, default to unit of pixels.')
            x = self.x * u.pix
        else:
            x = self.x
        if self.info is not None:
            return x.to(unit, angular_distance(self.info['range'] * u.au))
        else:
            return x.to(unit, angular_distance())

    def smooth(self, size, x=None, method='savgol', log=True, polyorder=3,
                **kwargs):
        """Return smoothed profile

        Parameters
        ----------
        size : number
            Smooth size
        x : reference
            Returns the x corresponding to the smoothed profile.  Useful
            when x changes during the smooth for, e.g., binning method.
        method : ['moving', 'savgol', 'binning'], optional
            The method to smooth the data.
            'moving' : Moving average
            'binning' : Simple binning
            'savgol' : Savitzky-Golay filter
        log : bool, optional
            Smooth in log scale.
        polyorder : number, optional
            Polynomial order for Savitzky-Golay filter.  No effect for
            other methods.
        kwargs : dict
            Other keyword parameters to pass to the smooth algorithm
        """
        # strip out unit
        y = getattr(self.data, 'value', self.data)
        unit = getattr(self.data, 'unit', 1)
        # take logarithm if needed
        if log:
            y = np.log10(np.clip(y, 1e-20, None))
        # smooth
        if method == 'moving':
            box = np.ones(size) / size
            y = np.convolve(y, box, mode='same')
        elif method == 'savgol':
            y = savgol_filter(y, size, polyorder=polyorder, **kwargs)
        elif method == 'binning':
            x = self.x.value
            xu = self.x.unit
            x = rebin(x, size, mean=True) * xu
            y = rebin(y, size, mean=True)
        else:
            raise ValueError('Method {} not recognized.'.format(method))
        # apply exponential if needed
        if log:
            y = 10**y
        # apply unit
        y *= unit
        # return
        if method == 'binning':
            return y, x
        else:
            return y

    def to_hdu(self, save_x=False, primary=False):
        """Convert data to a fits HDU and return"""
        if save_x:
            data = np.stack([self.data.value, self.x.value])
        else:
            data = self.data.value
        if primary:
            hdu = fits.PrimaryHDU(data)
        else:
            hdu = fits.ImageHDU(data)
        if self.info is not None:
            excluded_keys = set(hdu.header.keys()).union({'EXTEND'})
            for k in self.info.keys():
                if k not in excluded_keys:
                    hdu.header[k] = self.info[k]
        hdu.header['bunit'] = str(self.data.unit)
        if save_x:
            hdu.header['xunit'] = str(self.x.unit)
        return hdu

    def plot(self, ax=None, **kwargs):
        """Plot profile.

        ax : matplotlib.axes.Axes
            Axis to plot.
        kwargs : Keyword argument to pass to matplotlib.pyplot.plot()
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.x, self.data, **kwargs)

    def peak(self, x0, width, model='moffat', background='none',
                par=None, background_par=None, order=3,
                fitter=LevMarLSQFitter, tol=0.5, maxiter=5,
                sigma=None):
        """Find the peak position in the profile

        x0 : number, u.Quantity
            Initial value of the peak.
        width : number, u.Quantity
            Wdith within which the peak is searched.
        model : str, int, optional
            Model to be used to fit peak.  Values can be
                1, 'gaussian' : Gaussian function
                2, 'lorenz' : Lorenz function
                3, 'moffat' : Moffat function
                4, 'voigt' : Voigt function
                5, 'poly' : polynomial model
        background : str, optional
            Specify how background is handled.  Values can be
                'none' : No background handling
                'const' : Constant background.
                'linear' : Linear background.
                'data' : Estimate a constant background from data
            For the case of 'const' and 'linear', the background model
            can either be specified via the `background_par` parameter
            or be fitted and stored in `.par` attribute.
        par : sequence of float, optional
            The initial parameters of model.  If `None`, then the parameters
            will be automatically determined from `x0` and `width` parameters.
        background_par : float or sequence of float, optional
            The background parameters to be held fixed.  If `None`, then
            the background model will be fitted.  The treatment of this
            parameter depends on the value of `background` parameter:
                `none` : Ignore this parameter.
                `const` : Use the value or the first value if sequence
                    as the constant background.
                `linear` : Use the first two values as (slope, intercept)
                    parameters of the linear background.
        order : int, optional
            Order of polynomial fit.  Only useful when polynomial model
            is used.
        fitter : astropy-type fitter class
            The fitter to be used.  Only useful when polynomial model is
            not used.
        tol : number, u.Quantity, optional
            Tolerance of peak position.  In whatever unit the x-axis is in.
        maxiter : int, optional
            Maximum number of iteration.
        sigma : int or None, optional
            If not `None`, then specify the outlier removal threshold.
            In this case, attribute `.fit` dict contains a boolean array
            'outlier' to identify the outliers (True).

        Returns
        -------
        float or Quantity : The peak position.

        Attributes
        ----------
        This method will add the following attributes to the class
        `.par` : dict
            The parameters of the profile peak.  Depending on the model
            used, this dict could contain the following keys:
            'peak' : the peak position
            'amplitude' : the amplitude of the profile
            'fwhm' : the full-width-half-max of the profile
        `.model` : astropy model class object or numpy array
            The best-fit model or model parameters for the case of polynomial.
        `.fit` : dict
            'x_data', 'y_data' : the data used to fit the peak
            'x', 'y' : more continuous data and the best-fit model
            'outlier' : a boolean array masking out the outliers, if `sigma`
                is not `None`.
        """
        # preprocess and check parameters
        if isinstance(self.x, u.Quantity):
            x_value = self.x.value
            x_unit = self.x.unit
            if not (isinstance(x0, u.Quantity) and
                    isinstance(width, u.Quantity)):
                raise ValueError('u.Quantity required for `x0` and `width`.')
            if not (x0.unit.is_equivalent(x_unit) and
                    width.unit.is_equivalent(x_unit)):
                raise u.UnitsError(
                    'Incompatible units between `x0`, `width` and x-axis.')
            if isinstance(tol, u.Quantity):
                if not tol.unit.is_equivalent(x_unit):
                    raise u.UnitsError('Incompatible unit provided for `tol`.')
                tol = tol.to_value(x_unit)
            x0 = x0.to_value(x_unit)
            width = width.to_value(x_unit)
        else:
            x_value = self.x
            if isinstance(x0, u.Quantity) or isinstance(width, u.Quantity):
                raise ValueError('`x0` or `width` cannot be u.Quantity.')

        data_value = getattr(self.data, 'value', self.data)

        # initialize model
        if par is None:
            x1 = x0 - width / 2
            x2 = x0 + width / 2
            s1 = np.abs(x_value - x1).argmin()
            s2 = np.abs(x_value - x2).argmin()
            amp = data_value[s1:s2+1].max()
            if model in [1, 'gaussian', 2, 'lorentz', 3, 'moffat']:
                par = [amp, x0, width / 4]
            if model in [3, 'moffat']:
                par.append(1)
            if model in [4, 'voigt']:
                par = [x0, amp, width / 4, width / 4]
        if model in [1, 'gaussian']:
            m0 = models.Gaussian1D(*par)
        elif model in [2, 'lorentz']:
            m0 = models.Lorentz1D(*par)
        elif model in [3, 'moffat']:
            m0 = Moffat1D(*par)
        elif model in [4, 'voigt']:
            m0 = Voigt1D(*par)
        elif model in [5, 'poly']:
            pass
        else:
            raise ValueError('unrecognized model {}'.format(model))

        # prepare background model
        if background in ['none']:
            pass
        elif background in ['const']:
            if background_par is None:
                bg = models.Const1D(0.)
            else:
                const = background_par[0] if (hasattr(background_par,
                        '__iter__') and len(background_par.shape) > 0) \
                        else background_par
                bg = models.Const1D(const, fixed={'amplitude': True})
        elif background in ['linear']:
            if background_par is None:
                bg = models.Linear1D(0., 0.)
            else:
                bg = models.Linear1D(*background_par[:2],
                                     fixed={'slope': True, 'intercept': True})
        elif background in ['data']:
            const = np.nanmedian(data_value)
            bg = models.Const1D(const, fixed={'amplitude': True})
        else:
            raise ValueError('unrecognized background model')

        if model not in [5, 'poly']:
            # initialize fitter
            if sigma is None:
                fit = fitter()
            else:
                fit = FittingWithOutlierRemoval(fitter(), sigma_clip,
                            sigma=sigma)
            # add background
            if background != 'none':
                m0 = m0 + bg

        # iteration
        dx = 100
        n = 0
        x00 = x0
        while dx > tol and n < maxiter:
            x1 = x0 - width / 2
            x2 = x0 + width / 2
            # if solution out of range, then put it back to the initial value
            if (x0 < x00 - width) or (x0 > x00 + width):
                x0 = x00
                x_peak = x0
                m = m0
                continue
            s1 = np.abs(x_value - x1).argmin()
            s2 = np.abs(x_value - x2).argmin()
            x = x_value[s1:s2+1]
            y = data_value[s1:s2+1]

            # fit model
            xx = np.linspace(x.min(), x.max(), 1000)
            if model in [5, 'poly']:
                pp = np.polyfit(x, y, order)
                m = np.poly1d(pp)(xx)
                x_peak = xx[m.argmax()]
            else:
                m = fit(m0, x, y)
                if sigma is not None:
                    m, mask = m
                if model in [1, 'gaussian']:
                    x_peak = getattr(m, 'mean') if background == 'none' \
                                else getattr(m[0], 'mean')
                else:
                    x_peak = getattr(m, 'x_0') if background == 'none' \
                                else getattr(m[0], 'x_0')
            dx = abs(x_peak - x0)
            x0 = x_peak
            n += 1

        # save best-fit results
        self.fit = {'x': xx, 'x_data': x, 'y_data': y}
        if model in [5, 'poly']:
            self.model = pp
            self.fit['y'] = m
        else:
            self.model = m
            self.fit['y']= m(xx)
        if sigma is not None:
            self.fit['outlier'] = mask

        # save best-fit parameters
        xunit = getattr(self.x, 'unit', 1.)
        dunit = getattr(self.data, 'unit', 1.)
        self.par = {'peak': x_peak * xunit}
        if model not in [5, 'poly']:
            amp = self.model.amplitude \
                    if background == 'none' else self.model[0].amplitude
            self.par['amplitude'] = getattr(amp, 'value', amp)
            fwhm = self.model.fwhm \
                     if background == 'none' else self.model[0].fwhm
            self.par['fwhm'] = getattr(fwhm, 'value', fwhm)
            # apply units if needed
            self.par['amplitude'] *= dunit
            self.par['fwhm'] *= xunit

        return x_peak * getattr(self.x, 'unit', 1.)


class BrightnessProfileSet(list):
    """Class to process a set of tail brightness profile"""

    def __init__(self, *args, **kwargs):
        """
        meta : dict
            Meta data.
        """
        meta = kwargs.pop('meta', {})
        super().__init__(*args, **kwargs)
        for i, m in enumerate(self):
            if not isinstance(m, BrightnessProfile):
                if i == 1:
                    suffix = 'st'
                elif i == 2:
                    suffix = 'nd'
                elif i == 3:
                    suffix = 'rd'
                else:
                    suffix = 'th'
                raise ValueError('Incompatible type {} in the {}{} '
                    'elements.'.format(type(m), i, suffix))
        self.meta = meta

    def __getitem__(self, k):
        out = super().__getitem__(k)
        if hasattr(out, '__iter__'):
            return self.__class__(out, meta=self.meta)
        else:
            return out

    @classmethod
    def read(cls, infile):
        """Read from input file"""
        obj = []
        with fits.open(infile) as f_:
            meta = {}
            if 'meta' in f_[0].header:
                for i in range(f_[0].header['meta']):
                    k = f_[0].header['meta_{}'.format(i)]
                    meta[k] = f_[0].header[k]
            for hdu in f_[1:]:
                if hdu.data.ndim == 1:
                    data = hdu.data * u.Unit(hdu.header['bunit']) \
                            if 'bunit' in hdu.header else 1.
                    obj.append(BrightnessProfile(data, info=hdu.header))
                else:
                    data = hdu.data[0] * u.Unit(hdu.header['bunit']) \
                            if 'bunit' in hdu.header else 1.
                    x = hdu.data[1] * u.Unit(hdu.header['xunit']) \
                            if 'xunit' in hdu.header else u.pix
                    obj.append(BrightnessProfile(data, x, info=hdu.header))
        return cls(obj, meta=meta)

    def to_hdulist(self, save_x=False):
        """Convert to an HDUList class object

        save_x : bool, optional
            Save x-axis.  If not, then the default x-axis will start from
            0 with 1 pixel increment.
        """
        hdulist = [p.to_hdu(save_x=save_x) for p in self]
        hdu0 = fits.PrimaryHDU()
        hdu0.header['meta'] = len(self.meta), 'number of meta data keys'
        if hdu0.header['meta'] > 0:
            for i, k in enumerate(self.meta.keys()):
                hdu0.header['meta_{}'.format(i)] = k
            for k, v in self.meta.items():
                hdu0.header[k] = v
        hdulist.insert(0, hdu0)
        return fits.HDUList(hdulist)

    def write(self, outfile, overwrite=False, **kwargs):
        """Write to output file.

        Parameters
        ----------
        outfile : str
            Output file name.
        overwrite : bool, optional
            Overwrite existing file.
        """
        self.to_hdulist(**kwargs).writeto(outfile, overwrite=overwrite)

    @property
    def info(self):
        """Information table"""
        tables = []
        for i, p in enumerate(self):
            if p.info is not None:
                row = {'index': [i]}
                for k, v in p.info.items():
                    row[k.lower()] = [v]
                #print(row)
                tables.append(Table(row))
            else:
                tables.append(Table({'index': [i]}))
        return vstack(tables)
        table = tables[0]
        for t in tables[1:]:
            table = join(table, t, join_type='outer')
        return table

    def sort(self, key='utc-mid', reverse=False):
        """Sort the profile set in place

        key : str, optional
            Sorting key
        reverse : bool, optional
            Reverse sorting
        """
        def get_key(x):
            return getattr(x, 'info').__getitem__(key)
        super().sort(key=get_key, reverse=reverse)


class Moffat1D(models.Moffat1D):
    """Enhanced Moffat1D model

    .fwhm returns quantity when appropriate
    """
    @property
    def fwhm(self):
        w = super().fwhm
        if hasattr(self.gamma, 'unit'):
            w = u.Quantity(w, self.gamma.unit)
        return w


class Voigt1D(models.Voigt1D):
    """Enhanced Voigt1D model.

    Add attribute `.fwhm` and `.amplitude`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.fwhm_L.bounds = [0, None]
        self.fwhm_G.bounds = [0, None]

    @property
    def fwhm(self):
        dx = np.max([self.fwhm_G.value, self.fwhm_L.value])
        if hasattr(dx, 'unit'):
            dx = dx.value
        half_max = self.amplitude / 2
        if hasattr(half_max, 'unit'):
            half_max = half_max.value

        # Find the points where y crosses half_max
        x0 = self.x_0.value if hasattr(self.x_0, 'unit') else self.x_0
        fwhm_points = brentq(lambda x: self.evaluate(x, *self.parameters)
                                        - half_max,
                             x0 + dx * 0, x0 + dx * 3,
                             full_output=True)
        if fwhm_points[1].converged:
            fwhm = (fwhm_points[0] - x0) * 2
            return fwhm if not hasattr(self.x_0, 'unit') \
                        else u.Quantity(fwhm, self.x_0.unit)
        else:
            return None

    @property
    def amplitude(self):
        dx = np.max([self.fwhm_G.value, self.fwhm_L.value])
        if hasattr(self.x_0, 'unit'):
            dx = u.Quantity(dx, self.x_0.unit)
        x = np.linspace(self.x_0 - dx, self.x_0 + dx, 1000)
        return self(x).max()


class AzimuthalProfile(BrightnessProfile):
    """Azimuthal profile"""

    def peak(self, *args, **kwargs):

        # fix angle wrapping, TBD
        #if s1 < 0:
        #    seg = np.concatenate([scan[s1:], scan[:s2]])
        #    w = np.concatenate([error[s1:], error[:s2]])
        #elif s2 >= len(scan):
        #    seg = np.concatenate([scan[s1:], scan[:s2-len(scan)]])
        #    w = np.concatenate([error[s1:], scan[:s2-len(scan)]])
        #else:
        #    seg = scan[s1:s2]
        #    w = error[s1:s2]
        #ang = np.linspace(s1, s2, len(seg)) * ddeg

        return super().peak(*args, **kwargs)
