import numpy as np
import astropy.units as u
import astropy.constants as c


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

    def __init__(self, mu, nu, H1=None, H2=None, impactor=None, target=None, \
                               p=None, n1=None, n2=None, C1=None, k=None, regime=None):
        """
        H, mu, nu : number
          scaling law parameters
        impactor : Impactor class object
        target : Target class object
        p, n1, n2, C1, k : number
          Other scaling constants based on the definition in Housen & Holsapple (2011).
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
            if (self.impactor is None) or (self.target is None) or (self.H1 is None) \
                or (self.H2 is None) or (self.mu is None) or (self.nu is None):
                return 'unknown'
            grav = self.target.g * self.impactor.a / (self.impactor.U * self.impactor.U)
            strg = (self.H1 / self.H2)**((2+self.mu)/self.mu) * \
                   self.density_ratio**self.nu * \
                   (self.target.Y / (self.target.rho * self.impactor.U * self.impactor.U))**((2+self.mu)/2)
            if grav > strg:
                return 'gravity'
            else:
                return 'strength'
        else:
            return self._regime

    @u.quantity_input(x=u.m)
    def v(self, x):
        if (self.p is None) or (self.n2 is None) or (self.C1 is None):
            return np.nan
        w = x/self.impactor.a
        return (self.C1 * (w * self.density_ratio**self.nu)**(-1/self.mu) * (1 - x/(self.n2*self.R))**self.p).decompose() * self.impactor.U

    @property
    def R_gravity(self):
        """Gravity dominated impact crater radius"""
        e1 = (2 + self.mu - 6*self.nu) / (3 * (2 + self.mu))
        e2 = -self.mu / (2 + self.mu)
        return (self.H1 * self.density_ratio**e1 * self.gravity_parameter**e2 / self.rho_over_m).decompose()

    @property
    def R_strength(self):
        """Strength dominated impact crater radius"""
        e1 = (1 - 3*self.nu)/3
        e2 = -self.mu/2
        return (self.H2 * self.density_ratio**e1 * self.strength_parameter**e2 / self.rho_over_m).decompose()

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
        return (3 * self.k / (4 * np.pi) * self.density_ratio * ((x/self.impactor.a)**3 - self.n1**3) * self.impactor.m).decompose()

    @property
    def M_total(self):
        """Total ejected mass"""
        return 3*self.k*self.impactor.m/(4*np.pi)*self.density_ratio*((self.n2*self.R/self.impactor.a)**3-self.n1**3)

class SFDModel():
    """Size frequency distribution (SFD) model class

    Model assumes an exponental SFD: n(r) = N * (r/r0)**(-alpha).
    Parameters of the model:
        alpha : exponent, dimensionless
        N : normalization constant, number density for dr at radius `r0`, in unit 1/u.m
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
        self.cumulative = lambda r: self.N * self.r0 / -self.cumulative_alpha * (r/self.r0)**(-self.cumulative_alpha)
        self.cumulative_mass = lambda r: self.N * self.r0 * self.m0 / -self.mass_cumulative_alpha * (r/self.r0)**(-self.mass_cumulative_alpha)
        self.cumulative_area = lambda r: self.N * self.r0 * self.a0 / -self.area_cumulative_alpha * (r/self.r0)**(-self.area_cumulative_alpha)

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
        #return self.a0*self.r0/self.m0*self.mass_cumulative_alpha/self.area_cumulative_alpha * \
        #        (r1**(-self.area_cumulative_alpha) - r2**(-self.area_cumulative_alpha)) / (r1**(-self.mass_cumulative_alpha) - r2**(-self.mass_cumulative_alpha))
        return 3/4/self.rho*self.mass_cumulative_alpha/self.area_cumulative_alpha * \
                (r1**(-self.area_cumulative_alpha) - r2**(-self.area_cumulative_alpha)) / (r1**(-self.mass_cumulative_alpha) - r2**(-self.mass_cumulative_alpha))



class Didymos():
    """Didymos properties from DRA V2.22"""
    Dp = 780 * u.m
    Ds = 164 * u.m
    rho = 2170 * u.kg/u.m**3
    H = 18.16 * u.mag   # absolute magnitude
    G = 0.20   # IAU HG model G parameter
    Ageo = 0.15   # geometric albedo

    @property
    def A(self):
        """Total cross-sectional area"""
        return np.pi*(self.Dp/2)**2 + np.pi*(self.Ds/2)**2
