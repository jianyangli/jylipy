""" astronomical units conversion tools.
"""

## Copyright (C) 2008 APC CNRS Universite Paris Diderot <betoule@apc.univ-paris7.fr>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see http://www.gnu.org/licenses/gpl.html


from pylab import *
import numpy
import re
import pyfits

Jansky2SI=1.0e-26
SI2Jansky=1.0e+26
speedOfLight=2.99792458e8
kBoltzmann=1.380658e-23
sigmaStefanBoltzmann=5.67051e-8
hPlanck=6.6260755e-34
astronomicalUnit=1.49597893e11
solarConstant=1368.0
tropicalYear=3.15569259747e7
tcmb = 2.726
prefixes = {'n':1e-9,'u':1e-6,'m':1e-3,'k':1e3,'M':1e6,'G':1e9}

def Jy(freq):
    """ Return conversion factor from Jansky to SI at a given frequency.

    Parameters
    ----------
    freq : scalar. Frequency value.

    Returns
    -------
    scalar, conversion factor value.
    """
    return Jansky2SI


def K_CMB(freq):
    """ Return conversion factor from Kelvin CMB to SI at a given frequency.

    Parameters
    ----------
    freq : scalar. Frequency value.

    Returns
    -------
    scalar, conversion factor value.
    """
    x = (hPlanck *freq)/ (kBoltzmann * tcmb)
    ex = exp(x)
    den = ex-1
    den *= den
    den = 1/den
    fc = freq /speedOfLight
    return 2*kBoltzmann * fc *fc * x * x * ex * den

def K_RJ(freq):
    """ Return conversion factor from Kelvin Raleigh Jeans to SI at a given frequency.

    Parameters
    ----------
    freq : scalar. Frequency value.

    Returns
    -------
    scalar, conversion factor value.
    """
    return 2*kBoltzmann*freq*freq/(speedOfLight*speedOfLight)

def K_KCMB(freq):
    """ Return conversion factor from K/ KCMB to SI at a given frequency.

    Parameters
    ----------
    freq : scalar. Frequency value.

    Returns
    -------
    scalar, conversion factor value.
    """

    return tcmb*K_CMB(freq)

def y_sz(freq):
    """ Return conversion factor from y SZ to SI at a given frequency.

    Parameters
    ----------
    freq : scalar. Frequency value.

    Returns
    -------
    scalar, conversion factor value.
    """
    x = (hPlanck *freq)/ (kBoltzmann * tcmb)
    ex = exp(x)
    den = ex-1
    fc = freq /speedOfLight
    den = 1/den
    return x*ex*den*(x*(ex+1)*den - 4)*2*hPlanck*freq*den*fc*fc


def si(freq):
    """ Return conversion factor to SI at a given frequency.

    Parameters
    ----------
    freq : scalar. Frequency value.

    Returns
    -------
    scalar, conversion factor value.
    """
    return 1.0

def taubeta2(freq):
    """ Return conversion factor to taubeta2 at a given frequency.

    Parameters
    ----------
    freq : scalar. Frequency value.

    Returns
    -------
    scalar, conversion factor value.
    """

    return 1.0



def convfact(freq=1.0e10, fr=r'mK_CMB',to=r'mK_CMB'):
    """ Compute the conversion factor between two units at a given frequency in GHz.

    Parameters
    ----------
    freq : scalar. Frequency value.
    fr : string as prefixunit. unit can be either 'Jy/sr', 'K_CMB', 'K_RJ', 'K/KCMB', 'y_sz', 'si', 'taubeta'
    with optionnal prefix 'n', 'u', 'm', 'k', 'M', 'G'
    to : string as prefixunit.

    Returns
    -------
    scalar, conversion factor value.
    """

    frpre, fru =  re.match(r'(n|u|m|k|M|G)?(Jy/sr|K_CMB|K_RJ|K/KCMB|y_sz|si|taubeta2)', fr).groups()
    topre, tou = re.match(r'(n|u|m|k|M|G)?(Jy/sr|K_CMB|K_RJ|K/KCMB|y_sz|si|taubeta2)', to).groups()

    if fru == 'Jy/sr':
        fru = 'Jy'
    if tou == 'Jy/sr':
        tou = 'Jy'
    if fru == tou:
        fac = 1.0
    else:
        fac = eval(fru+'(freq)/'+tou+'(freq)')

    if not frpre is None:
        fac *= prefixes[frpre]
    if not topre is None:
        fac /= prefixes[topre]

    return fac

def glon2phi(glon):
    deg2rad = pi/180.0
    return glon*deg2rad

def phi2glon(phi):
    rad2deg = 180.0/pi
    return phi*rad2deg

def glat2theta(glat):
    deg2rad = pi/180.0
    return (90-glat)*deg2rad

def theta2glat(theta):
    rad2deg = 180.0/pi
    return 90- (theta*rad2deg)

def deg2rad(x):
    return x*pi/180.0

def rad2deg(x):
    return x*180.0/pi
