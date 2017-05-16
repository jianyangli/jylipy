import numpy
from numpy import pi, abs, sqrt, cos, sin, arccos, arctan, tan
from PyAstronomy.pyasl import MarkleyKESolver

class KeplerOrbit(object):
  '''
    Calculate a Kepler orbit.
    
    The definitions and most of the formulae used in this class
    derive from the book "Orbital Motion" by A.E. Roy.
    
    The orientation of the ellipse:
        For zero inclination the ellipse is located in the x-y plane.
        If the eccentricity is increased, the periastron will lie
        in +x direction. If the inclination is increased, the ellipse
        will be rotating around the x-axis, so that +y is rotated
        toward +z. In increase in Omega corresponds to a rotation
        around the z-axis so that +x is rotated toward +y.
        Changing `w`, i.e., the longitude of the periastron, will
        not change the plane of the orbit, but rather represent a
        rotation of the orbit in the plane.
    
    Parameters
    ----------
    a : float
        Semi-major axis
    per : float
        Orbital period
    e : float, optional
        Orbital eccentricity.
    tau : float, optional
        Time of perihelion passage.
    Omega : float, optional
        Longitude of the ascending node [deg].
    w : float, optional
        Argument of perihelion [deg]
    i : float, optional
        Orbit inclination [deg].
    ks : Class object, optional
        The solver for Kepler's Equation. Default is the
        `MarkleyKESolver`. Each solver must have a `getE`
        method, which takes either a float or array of float
        of mean anomalies and the eccentricity, and returns
        the associated eccentric anomalies.
    
    Attributes
    ----------
    a : float
        Semi-major axis
    q : float
        Periapsis distance
    e : float
        Eccentricity
    i : float
        Orbit inclination [deg].
    Omega : float
        Longitude of the ascending node [deg]
    w : float
        Argument of perihelion [deg]
    per : float
        Orbital period
    tau : float
        Time of perihelion passage
    ks : Class object
        Solver for Kepler's equation
    n : float
        Mean motion (circular frequency).
  '''
  
  def meanAnomaly(self, t):
    '''
      Calculate the mean anomaly.
      
      Parameters
      ----------
      t : float or array
          The time axis.
      
      Returns
      -------
      Mean anomaly : float or array
          The mean anomaly (whether float or array
          depends on input).
    '''
    return self._n*(t-self.tau)

  def _getEccentricAnomaly(self, t):
    '''
      Calculate eccentric anomaly.
      
      Parameters
      ----------
      t : array of float
          The times at which to calculate the eccentric anomaly, E.
      
      Returns
      -------
      E : Array of float
          The eccentric anomaly.
    '''
    M = self.meanAnomaly(t)
    if not hasattr(t, "__iter__"):
      if self.e < 1:
      	E = self.ks.getE(M, self.e)
      if self.e > 1:
      	E = 0.   #### TO BE IMPLEMENTED
      if self.e == 1:
      	E = 0.5*(3.*M + np.sqrt(9.*M*M + 4.))  # Eq. 6.2.29 in Collins.  Don't know whether this is indeed eccentric anomaly for parabola
    else:
      E = numpy.zeros(len(t))
      if self.e < 1:
        for i in range(len(t)):
          E[i] = self.ks.getE(M[i], self.e)
      if self.e > 1:
      	E[:] = 0.   #### TO BE IMPLEMENTED
      if self.e == 1:
      	E = 0.5*(3.*M + np.sqrt(9.*M*M + 4.))
    return E
  
  def radius(self, t, E=None):
    '''
      Calculate the orbit radius.
      
      Parameters
      ----------
      t : float or array
          The time axis.
      E : float or array, optional
          If known, the eccentric anomaly corresponding
          to the time points. If not given, the numbers
          will be calculated.
      
      Returns
      -------
      Radius : float or array
          The orbit radius at the given points in time.
          Type depends on input type.
    '''
    if E is None:
      E = self._getEccentricAnomaly(t)
    if self.e < 1:
      r = self.a * (1. - self.e*np.cos(E))
    if self.e > 1:
      r = self.a * (self.e*np.cosh(E) - 1.)
    if self.e == 1:
      M = self.meanAnomaly(t)
      x = 0.5*(3.*M + np.sqrt(9.*M*M + 4.))
      tan_halfnu = x**0.3333333333333333 - x**-0.3333333333333333
      r = self.q*(1. + tan_halfnu*tan_halfnu)

    return r
  
  def xyzPos(self, t, getTA=False):
    '''
      Calculate orbit position.
      
      Parameters
      ----------
      t : float or array
          The time axis.
      getTA : boolean, optional
          If True, returns the "true anomaly" as a function
          of time (default is False).
      
      Returns
      -------
      Position : array
          The x, y, and z coordinates of the body at the given time.
          If the input was an array, the output will be an array of
          shape (input-length, 3), holding the positions at the given
          times.
      True anomaly : float or array
          Is returned only if `getTA` is set to True. The true anomaly
          at the specified times.
    '''
    # E = self._getEccentricAnomaly(t)
    r = self.radius(t) # , E=E)
    if self.e < 1:
      f = np.arctan( np.sqrt((1.+self.e)/(1.-self.e)) * np.tan(E/2.) ) * 2.0
    if self.e > 1:
      f = np.arctan( np.sqrt((self.e+1.)/(self.e-1.)) * np.tanh(E/2.) ) * 2.0
    if self.e == 1:
      f = np.arctan( E**0.3333333333333333 - E**-0.3333333333333333 ) * 2.0
    wf = self._w + f
    if not hasattr(wf, "__iter__"):
      xyz = numpy.array([cos(self._Omega)*cos(wf) - sin(self._Omega)*sin(wf)*cos(self._i),
                         sin(self._Omega)*cos(wf) + cos(self._Omega)*sin(wf)*cos(self._i),
                         sin(wf)*sin(self._i)
                         ]) * r
    else:
      # Assume it is an array
      xyz = numpy.zeros( (len(t), 3) )
      for i in range(len(t)):
        xyz[i,::] = numpy.array([cos(self._Omega)*cos(wf[i]) - sin(self._Omega)*sin(wf[i])*cos(self._i),
                         sin(self._Omega)*cos(wf[i]) + cos(self._Omega)*sin(wf[i])*cos(self._i),
                         sin(wf[i])*sin(self._i)
                         ]) * r[i]
    if not getTA:
      return xyz
    else:
      return xyz, f
  
  def xyzVel(self, t):
    '''
      Calculate orbit velocity.
      
      Parameters
      ----------
      t : float or array
          The time axis.
      
      Returns
      -------
      Velocity : array
          The x, y, and z components of the body's velocity at the
          given time. If the input was an array, the output will be
          an array of shape (input-length, 3), holding the velocity
          components at the given times. The unit is that of the
          semi-major axis divided by that of the period.
    '''
    # From AE ROY "Orbital motion" p. 102
    E = self._getEccentricAnomaly(t)
    l1 = cos(self._Omega)*cos(self._w) - sin(self._Omega)*sin(self._w)*cos(self._i)
    l2 = -cos(self._Omega)*sin(self._w) - sin(self._Omega)*cos(self._w)*cos(self._i)
    m1 = sin(self._Omega)*cos(self._w) + cos(self._Omega)*sin(self._w)*cos(self._i)
    m2 = -sin(self._Omega)*sin(self._w) + cos(self._Omega)*cos(self._w)*cos(self._i)
    n1 = sin(self._w)*sin(self._i)
    n2 = cos(self._w)*sin(self._i)
    b = self.a * sqrt(1. - self.e**2)
    r = self.radius(t, E)
    nar = self._n * self.a / r
    if not hasattr(t, "__iter__"):
      vel = nar * numpy.array([b*l2*cos(E) - self.a*l1*sin(E),
                               b*m2*cos(E) - self.a*m1*sin(E),
                               b*n2*cos(E) - self.a*n1*sin(E)])
    else:
      # Assume it is an array
      vel = numpy.zeros( (len(t), 3) )
      for i in range(len(t)):
        vel[i,::] = nar[i] * numpy.array([b*l2*cos(E[i]) - self.a*l1*sin(E[i]),
                               b*m2*cos(E[i]) - self.a*m1*sin(E[i]),
                               b*n2*cos(E[i]) - self.a*n1*sin(E[i])])
    return vel
  
  def xyzPeriastron(self):
    '''
      The position of the periastron.
      
      Returns
      -------
      Periastron : array of float
          The x, y, and z coordinates of the periastron. 
    '''
    return self.xyzPos(self.tau)
  
  def xyzApastron(self):
    '''
      The position of the apastron.
      
      The apastron is the point of greatest distance.
      
      Returns
      -------
      Apastron : array of float
          The x, y, and z coordinates of the apastron
    '''
    return self.xyzPos(self.tau + 0.5*self.per)

  def xyzCenter(self):
    '''
      Center of the ellipse
      
      Returns
      -------
      Center : array of float
          x, y, and z coordinates of the center of the Ellipse.
    '''
    return (self.xyzPeriastron() + self.xyzApastron())/2.0

  def xyzFoci(self):
    '''
      Calculate the foci of the ellipse
      
      Returns
      -------
      Foci : Tuple of array of float
          A tuple containing two arrays, which hold the x, y, and z
          coordinates of the foci.
    '''
    peri = self.xyzPeriastron()
    apas = self.xyzApastron()
    center = (peri + apas) / 2.0
    direc = (peri - apas) / numpy.sqrt( ((peri - apas)**2).sum() )
    ae = self.a * self.e
    return (center + ae*direc, center - ae*direc)
    
  def xyzNodes(self):
    '''
      Calculate the nodes of the orbit.
      
      The nodes of the orbit are the points at which
      the orbit cuts the observing plane. In this case,
      these are the points at which the z-coordinate
      vanishes, i.e., the x-y plane is regarded the plane
      of observation.
      
      Returns
      -------
      Nodes : Tuple of two coordinate arrays
          Returns the xyz coordinates of both nodes. 
    '''
    # f = -w
    E = arctan( tan(-self._w/2.0) * sqrt((1.-self.e)/(1.+self.e)) ) * 2.
    M = E - self.e * sin(E)
    t = M/self._n + self.tau
    node1 = self.xyzPos(t)
    # f = -w + pi
    E = arctan( tan((-self._w + pi)/2.0) * sqrt((1.-self.e)/(1.+self.e)) ) * 2.
    M = E - self.e * sin(E)
    t = M/self._n + self.tau
    node2 = self.xyzPos(t)
    return (node1, node2)
  
  def projPlaStDist(self, t):
    '''
      Calculate the sky-projected planet-star separation.
      
      Parameters
      ----------
      t : float or array
          The time axis.
      
      Returns
      -------
      Position : array
          The sky-projected planet-star separation at the given time.
          If the input was an array, the output will be an array, 
          holding the separations at the given times.
    '''
    p = self.a * (1.- self.e**2)
    E = self._getEccentricAnomaly(t)
    f = arctan( sqrt((1.+self.e)/(1.-self.e)) * tan(E/2.) ) * 2.0
    wf = self._w + f
    if not hasattr(wf, "__iter__"):
      psdist = p/(1.+self.e*cos(f))*sqrt(1.-sin(self._i)**2*sin(wf)**2)

    else:
      # Assume it is an array
      psdist = numpy.zeros(len(t))
      for i in range(len(t)):
        psdist[i] = p/(1.+self.e*cos(f[i]))*sqrt(1.-sin(self._i)**2*sin(wf[i])**2)

    return psdist
  
  def yzCrossingTime(self):
    '''
      Calculate times of crossing the yz-plane.
      
      This method calculates the times at which
      the yz-plane is crossed by the orbit. This
      is equivalent to finding the times where
      x=0.
      
      Returns
      -------
      Time 1 : float
          First crossing time defined as having POSITIVE
          y position.
      Time 2 : float
          Second crossing time defined as having NEGATIVE
          y position.
    '''
    if abs(self._Omega) < 1e-16:
      f = -self._w + pi/2.0
    else:
      f = -self._w + arctan(1.0/(tan(self._Omega)*cos(self._i)))
    E = 2.*arctan(sqrt((1-self.e)/(1.+self.e)) * tan(f/2.))
    t1 = (E - self.e*sin(E))/self._n + self.tau
    p1 = self.xyzPos(t1)
    f += pi
    E = 2.*arctan(sqrt((1-self.e)/(1.+self.e)) * tan(f/2.))
    t2 = (E - self.e*sin(E))/self._n + self.tau
    
    t1 -= self._per * numpy.floor(t1/self._per)
    t2 -= self._per * numpy.floor(t2/self._per)
    
    if p1[1] >= 0.0:
      # y position of p1 is > 0
      return (t1, t2)
    else:
      return (t2, t1)
 
  def xzCrossingTime(self):
    '''
      Calculate times of crossing the xz-plane.
      
      This method calculates the times at which
      the xz-plane is crossed by the orbit. This
      is equivalent to finding the times where
      y=0.
      
      Returns
      -------
      Time 1 : float
          First crossing time defined as having POSITIVE
          x position.
      Time 2 : float
          Second crossing time defined as having NEGATIVE
          x position.
    '''
    f = -self._w - arctan(tan(self._Omega)/cos(self._i))
    E = 2.*arctan(sqrt((1-self.e)/(1.+self.e)) * tan(f/2.))
    t1 = (E - self.e*sin(E))/self._n + self.tau
    p1 = self.xyzPos(t1)
    f += pi
    E = 2.*arctan(sqrt((1-self.e)/(1.+self.e)) * tan(f/2.))
    t2 = (E - self.e*sin(E))/self._n + self.tau
    
    t1 -= self._per * numpy.floor(t1/self._per)
    t2 -= self._per * numpy.floor(t2/self._per)
    
    if p1[0] >= 0.0:
      # y position of p1 is > 0
      return (t1, t2)
    else:
      return (t2, t1)
        
  def _setPer(self, per):
    self._per = per
    self._n = 2.0*pi/self._per
  
  def _seti(self, i):
    self._i = i/180.*pi
  
  def _setw(self, w):
    self._w = w/180.*pi
  
  def _setOmega(self, omega):
    self._Omega = omega/180.*pi
  
  per = property(lambda self: self._per, _setPer, doc="The orbital period.")
  i = property(lambda self: self._i/pi*180, _seti)
  w = property(lambda self: self._w/pi*180, _setw)
  Omega = property(lambda self: self._Omega/pi*180, _setOmega)
  
  def __init__(self, a, per, e=0, tau=0, Omega=0, w=0, i=0, ks=MarkleyKESolver):
    # i, w, Omega are properties so that the numbers can be given in
    # deg always. The underscored attributes are in rad.
    self.i = i
    self.w = w
    self.Omega = Omega
    self.e = e
    self.a = a
    self.per = per
    self.tau = tau
    self.ks = ks()



def phaseAngle(pos, los='-z'):
  '''
    Calculate the phase angle.
    
    The phase angle is the angle between the star and the Earth (or Sun)
    as seen from the planet (e.g., Seager et al. 1999, ApJ, 504).
    The range of the phase angle is 0 - 180 degrees. In the calculations,
    it is assumed to be located at the center of the coordinate system.
    
    Parameters
    ----------
    pos : array
        Either a one-dimensional array with xyz coordinate or a
        [3,N] array containing N xyz positions.
    los : LineOfSight object
        A `LineOfSight` object from the pyasl giving the line of
        sight.
    
    Returns
    -------
    Phase angle : The phase angle in degrees. Depending on the input,
        it returns a single value or an array. 
  '''
  from PyAstronomy.pyasl import LineOfSight
  l = LineOfSight(los)
  if pos.shape == (3,):
    # It is a single value
    return numpy.arccos( numpy.sum(-pos * (-l.los)) / \
            numpy.sqrt(numpy.sum(pos**2)))/numpy.pi*180. 
  else:
    # It is an array of positions
    N = len(pos[::,0])
    result = numpy.zeros(N)
    for i in range(N):
      print(i, numpy.sum((-pos[i,::]) * (-l.los)), pos[i,::])
      result[i] = numpy.arccos( numpy.sum((-pos[i,::]) * (-l.los)) / \
                  numpy.sqrt(numpy.sum(pos[i,::]**2)) )
    return result/numpy.pi*180.
