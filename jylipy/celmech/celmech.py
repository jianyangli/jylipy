'''
Package for basic celestial mechanics calculations




Package dependences
-------------------
numpy, quantities.units, quantities.constants.G, astropy.time.TimeDelta,
PyAstronomy.pyasl.KeplerEllipse, PyAstronomy.constants, coord.vectang

History
-------
8/4/2013, started by JYL @PSI
'''

import numpy
from numpy import sin, cos, sqrt, pi, cross, arccos, arctan, tan, array
from numpy.linalg import norm
import quantities.units as u
from quantities.constants import G
from astropy.time import TimeDelta
from PyAstronomy.pyasl import KeplerEllipse
from PyAstronomy import constants
from ..coord import vecsep

constants.load('/Volumes/apophis_raid/work/JYLPy/constants',True)
MSun = constants.inUnitsOf('MSun', u.kg) # Mass of the Sun
kG = constants.inUnitsOf('kG', 1/u.day)  # Gaussian gravitational constant


class KeplerSolver(object):
    '''
 Solver for Keper equation E = M+e*sin(M), required by class
 PyAstronomy.pyasl.KeplerEllipse.  It only contains one single
 function, getE(M, e), to solve for eccentric anomaly for given
 mean anomaly and eccentricity.
    '''
    def getE(self, M, e, tol=1e-14, E0=0.):

        diff = 1.
        E_old = E0
        while diff>tol:
            E_new = E_old-(E_old-e*sin(E_old)-M)/(1-e*cos(E_old))
            diff = abs(E_new-E_old)
            E_old = E_new
        return E_old


def vec2orb(r, v, t, mu=G*MSun, beta=0.):
    '''
 Calculate orbital elements from the position and velocity vectors.

 Usage::
 a, e, i, node, peri, tp = vec2orb(x, v, t[, mu=G*MSun][, beta=0.])

 Input
 -----
 r, v  : numpy array of quantities
     The position vector and velocity vector
 t  : astropy Time instance
     The time corresponding to the position and velocity vectors
 mu : astropy quantity
     Gravitational parameter GM
 beta : floating point
     Solar radiation pressure parameter

 Output
 ------
 Returns a tuple (a, e, i, node, peri, tp), where
 a  : astropy quantity
     Semi-major axis
 e  : floating point
     Eccentricity
 i  : floating point
     Inclination [deg]
 node : floating point
     Longitude of ascending node [deg]
 peri : floating point
     Argument of periapsis [deg]
     tp : astropy Time instance
     Time of periapsis

 History
 -------
 8/9/2013, created by JYL@PSI
    '''

    rr = norm(r)*r.units
    vr = norm(v)*v.units
    theta = vectang(r.magnitude, v.magnitude, False)
    L = cross(r,v)*r.units*v.units  # angular momentum
    Lr = norm(L)*L.units

    # semi-major axis
    a = mu*rr/(2*mu-rr*vr*vr)
    print(a)
    # eccentricity
    e = sqrt(1-(Lr**2/(mu*a)).simplified.magnitude)

    # mean motion n
    n = kG*sqrt((1.-beta)/(a/u.au).simplified.magnitude**3)
    if e == 1:
        n /= sqrt(2)
    if e > 1:
        n *= (e-1)**(3/2.)

    # inclination, angle between L and z-axis
    i = arccos(L[2].magnitude/norm(L))*180/pi

    # true anomaly, eccentric anomaly, and mean motion
    if e < 1:
        cosE = (1. - (rr/a).simplified.magnitude) / e
        E = arccos(cosE)
        M = E - e*sqrt(1. - cosE*cosE)
        nu = 2*arctan(sqrt((1+e)/(1-e))*tan(E/2))
    if e > 1:
        coshF = ((rr/a).simplified.magnitude + 1.) / e
        F = arccosh(coshF)
        M = e*sqrt(coshF*coshF - 1.) - F
        nu = 2*arctan(sqrt((e+1)/(e-1))*tanh(F/2))
    # e == 1 case to be implemented
    nu *= 180/pi
    if theta > 90.:
        nu = 360.-nu
        M = 2.*pi-M

    # longitude of ascending node, and argument of perihelion
    # * longitude of ascending node is the angle from x-axis to node vector.
    #   node vector is the cross product of z-axis and L
    # * if i==0 or i==180, then node and peri are both 0 by by definition.
    if i == 0. or i == 180.:
        node = peri = 0.
    else:
        vnode = cross(array([0.,0,1]), L.magnitude)
        node = vectang(array([1.,0,0]), vnode)
        peri = vectang(vnode, r.magnitude) - nu

    # time of perihelion
    dt = M/n
    tp = t - TimeDelta((dt/u.s).simplified.magnitude, format='sec')

    return a, e, i, node, peri, tp


def orb2vec(a, e, i, node, peri, tp, t, mu=G*MSun, beta=0.):
    '''
 Calculate the position and velocity vectors from orbital elements

 Usage::
 r, v = (a, e, i, node, peri, tp, t[, mu=G*MSun][, beta=0.])

 Input
 -----
 a : astropy quantity
     Semi-major axis (e<1), transverse semi-axis (e>1), or periapsis
     (e==1) distance
 e  : floating point
     Eccentricity
 i  : floating point
     Inclination [deg]
 node : floating point
     Longitude of ascending node [deg]
 peri : floating point
     Argument of perihelion [deg]
 tp : astropy Time instance
     Time of perihelion
 t  : astropy Time instance
     The time for which the position and velocity vectors are to be
     calculated.
 mu : astropy quantity
     Gravitational parameter GM
 beta : floating point
     Solar radiation pressure parameter

 Output
 ------
 Returns the position vector(s) and velocity vector(s) for the specified
 orbital elements and the times.  Both are numpy arrays of quantity type.

 History
 -------
 8/7/2013, created by JYL @PSI
    '''

    # mean motion n in radiance/day
    n = kG*sqrt((1.-beta)/(a/u.au).simplified.magnitude**3)
    if e == 1:
        n /= sqrt(2)
    if e > 1:
        n *= (e-1)**(3/2.)

    # period or pseudo-period for open orbits
    period = 2*pi/n

    # set up an orbit
    a1 = a.magnitude
    p1 = period.magnitude
    tau = tp.mjd
    ks = KeplerSolver
    orbit = KeplerEllipse(a1, p1, e=e, tau=tau, Omega=node, w=peri, i=i, ks=ks)

    # calculate r and v
    r = orbit.xyzPos(t.mjd)*a.units
    v = (orbit.xyzVel(t.mjd)*a.units/period.units)

    return r, v


def hohmann_dv(r1, r2, mu=G*MSun):
    '''
 Calculate the delta-v for Hohmann transfer orbits

 Input
 -----
 r1  : astropy quantity
     Radius of inner orbit
 r2  : astropy quantity
     Radius of outer orbit
 mu  : astropy quantity
     The gravity parameter G*(M+m).  Default is G*MSun

 Output
 ------
 Returns the total delta-v corresponding to a Hohmann transfer.  astropy
 quantity

 Notes
 -----
 Based on wikipedia: http://en.wikipedia.org/wiki/Hohmann_transfer_orbit

 History
 -------
 8/8/2013, created by JYL @PSI
    '''

    dv1 = sqrt(mu/r1)*(sqrt(2*r2/(r1+r2))-1)
    dv2 = sqrt(mu/r2)*(1-sqrt(2*r1/(r1+r2)))
    dv = dv1+dv2
    dv.units = u.km/u.s
    return dv


