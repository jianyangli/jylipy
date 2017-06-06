'''Package containing the calculation of observation geometries for
solar system objects.
'''
import spiceypy as spice
import numpy as np
from .core import *
from .apext import Table, Column
from numpy.linalg import norm
from .vector import *

__all__ = ['load_generic_kernels', 'subcoord', 'obsgeom']

def load_generic_kernels(kernel_dir=None):
	'''
 Load generic SPICE kernels
	'''
	if kernel_dir is None:
		kerdir = '/Users/jyli/work/naif/generic_kernels/'
	else:
		kerdir = kernel_dir
	spice.furnsh(kerdir+'lsk/naif0011.tls')
	spice.furnsh(kerdir+'spk/planets/de430.bsp')
	spice.furnsh(kerdir+'pck/pck00010.tpc')
	spice.furnsh(kerdir+'names.ker')


def subcoord(time, target, observer='earth', bodyframe=None, saveto=None, planetographic=False):
	'''
 Calculate the sub-observer and sub-solar points.

 Parameters
 ----------
 time : str, array-like
   Time(s) to be calculated.  Must be in a format that can be accepted
   by SPICE.
 target : str
   The name of target that SPICE accepts.
 observer : str, optional
   The name of observer that SPICE accepts.
 bodyframe : str, optional
   The name of the body-fixed coordinate system, if not standard (in
   the form of target+'_fixed')
 saveto : str, file, optional
   Output file
 planetographic : bool, optional
   If `True`, then the planetographic coordinates are returned as
   opposed to planetocentric.

 Return
 ------
 Astropy Table:
   'sslat' : sub-solar latitude
   'sslon' : sub-solar longitude
   'solat' : sub-observer latitude
   'solon' : sub-observer latitude
   'rh' : heliocentric distance
   'range' : observer distance
   'phase' : phase angle
   'polepa' : position angle of pole
   'poleinc' : inclination of pole
   'sunpa' : position angle of the Sun
   'suninc' : inclination angle of the Sun
 Position angles are measured from celestial N towards E.  inclination
 angles are measured from sky plane (0 deg) towards observer (+90).
 All angles are in deg, all distances are in AU

 v1.0.0 : JYL @PSI, May 23, 2014
 v1.0.1 : JYL @PSI, July 8, 2014
   Modified the call to spice.gdpool to accomodate a behavior that is
   different from what I remember
   Modified the call to spice.str2et
 v1.0.2 : JYL @PSI, July 15, 2014
   Changed the calculation of pole orientation to accomodate the case
   where the frame is expressed in the kernel pool differently than
   for small bodies.
 v1.0.3 : JYL @PSI, October 8, 2014
   Fixed the bug when input time is a scalor
   Change return to an astropy Table
 v1.0.4 : JYL @PSI, November 19, 2014
   Small bug fix for the case when input time is a scalor
   Small bug fix for the output table unit and format
 v1.0.5 : JYL @PSI, October 14, 2015
   Add keyword `planetographic`
   Add target RA and Dec in the return table
   Increased robustness for the case when no body frame is defined
   Change the table headers to start with capital letters
   Improve the program structure
	'''

	# Determine whether kernel pool for body frame exists
	if bodyframe is None:
		bodyframe = target+'_fixed'
	try:
		kp = spice.gipool('FRAME_'+bodyframe.upper(),0,1)
	except spice.utils.support_types.SpiceyError:
		kp = None
	if kp is not None:
		code = kp[0]
		polera = spice.bodvrd(target, 'POLE_RA', 3)[1][0]
		poledec = spice.bodvrd(target, 'POLE_DEC', 3)[1][0]
		r_a, r_b, r_c = spice.bodvrd(target, 'RADII', 3)[1]
		r_e = (r_a+r_b)/2
		flt = (r_e-r_c)/r_e

	# Process input time
	if isinstance(time,str):
		et = [spice.str2et(time)]
		time = [time]
	elif hasattr(time, '__iter__'):
		et = [spice.str2et(x) for x in time]
	else:
		raise TypeError('str or list of str expected, {0} received'.format(type(time)))

	# Prepare for iteration
	sslat, sslon, solat, solon, rh, delta, phase, polepa, poleinc, sunpa, suninc, tgtra, tgtdec = [], [], [], [], [], [], [], [], [], [], [], [], []
	workframe = condition(kp is None, 'J2000', bodyframe)

	# Iterate over time
	for t in et:

		# Target position (r, RA, Dec)
		pos1, lt1 = spice.spkpos(target, t, 'J2000', 'lt+s', observer)
		pos1 = np.array(pos1)
		rr, ra, dec = spice.recrad(pos1)
		delta.append(rr*units.km.to(units.au))
		tgtdec.append(np.rad2deg(dec))
		tgtra.append(np.rad2deg(ra))

		# Heliocentric distance
		pos2, lt2 = spice.spkpos('sun', t-lt1, 'J2000', 'lt+s', target)
		rh.append(norm(pos2)*units.km.to('au'))

		# Phase angle
		phase.append(vecsep(-pos1, pos2, directional=False))

		# Sun angle
		m = np.array(spice.twovec(-pos1, 3, [0,0,1.], 1))
		rr, lon, lat = spice.recrad(m.dot(pos2))
		sunpa.append(np.rad2deg(lon))
		suninc.append(np.rad2deg(lat))

		if kp is not None:

			# Sub-observer point
			pos1, lt1 = spice.spkpos(target, t, bodyframe, 'lt+s', observer)
			pos1 = np.array(pos1)
			if planetographic:
				lon, lat, alt = spice.recpgr(target, -pos1, r_e, flt)
			else:
				rr, lon, lat = spice.recrad(-pos1)
			solat.append(np.rad2deg(lat))
			solon.append(np.rad2deg(lon))

			# Sub-solar point
			pos2, lt2 = spice.spkpos('sun', t-lt1, bodyframe, 'lt+s', target)
			lon, lat, alt = spice.recpgr(target, pos2, r_e, 0.9)
			#print np.rad2deg(lon), np.rad2deg(lat)
			rr, lon, lat = spice.recrad(pos2)
			#print np.rad2deg(lon), np.rad2deg(lat)
			if planetographic:
				lon, lat, alt = spice.recpgr(target, pos2, r_e, flt)
			else:
				rr, lon, lat = spice.recrad(pos2)
			sslon.append(np.rad2deg(lon))
			sslat.append(np.rad2deg(lat))

			# North pole angle
			pole = [polera, poledec]
			rr, lon, lat = spice.recrad(m.dot(sph2xyz(pole)))
			polepa.append(np.rad2deg(lon))
			poleinc.append(np.rad2deg(lat))

	if kp is None:
		tbl = Table((time, rh, delta, phase, tgtra, tgtdec), names='Time rh Range Phase RA Dec'.split())
	else:
		tbl = Table((time, rh, delta, phase, tgtra, tgtdec, solat, solon, sslat, sslon, polepa, poleinc, sunpa, suninc), names='Time rh Range Phase RA Dec SOLat SOLon SSLat SSLon PolePA PoleInc SunPA SunInc'.split())

	for c in tbl.colnames:
		tbl[c].format='%.2f'
		tbl[c].unit = units.deg
	tbl['Time'].format='%s'
	tbl['Time'].unit=None
	tbl['rh'].format = '%.4f'
	tbl['rh'].unit = units.au
	tbl['Range'].format = '%.4f'
	tbl['Range'].unit = units.au
	if saveto is not None:
		tbl.write(saveto)

	return tbl


def obsgeom(time, target, observer='earth', frame=None, saveto=None):
	'''Calculate observing geometry

	time : instance of astropy.time.Time, str, number, or iterables of them
	  Time, UTC string, or ET
	target : str
	  NAIF SPICE code of target
	observer : str, optional
	  NAIF SPICE code of observer
	frame : str, optional
	  NAIF SPICE code of image coordinate.  If specified, then the PA of north
	  is calculated.
	saveto : str, file, optional
	  Save table to file

	Returns an astropy.table.Table instance

	v1.0.0 : Feb 2015, JYL @PSI
	v1.0.1 : 4/7/2016, JYL @PSI
	  Added solar elongation
	  Added 'lt+s' correction for all position calculation
	'''

	if isinstance(time, Time):
		et = time.et
		tstr = time.utc.isot
	elif hasattr(time, '__iter__'):
		time = np.asanyarray(time)
		if time.dtype.kind in ['S','U']:
			tmp = Time(time)
			et = tmp.et
			tstr = tmp.utc.isot
		elif (time.dtype.kind == 'f') or (time.dtype.kind == 'i'):
			et = time
			tstr = Time(time,format='et').utc.isot
		else:
			raise TypeError('`time` must have string type or number type elements, {0} received'.format(time.dtype))
	else:
		raise TypeError('`time` must be either a Time instance, or an iterable type, {0} received'.format(type(time)))

	ra, dec, rh, delta, phase, sunpa, suninc, velpa, velinc, selon = np.zeros((10, len(et)))
	if frame is not None:
		norpa = []
	for i in range(len(et)):
		ste, lt = spice.spkezr(target, et[i], 'j2000','lt+s',observer)
		sts, lt = spice.spkezr('sun', et[i]-lt, 'j2000','lt+s',target)
		sts = -np.asarray(sts)
		sss, lt = spice.spkezr('sun', et[i], 'j2000','lt+s',observer)
		delta[i], ra[i], dec[i] = xyz2sph(ste[:3])
		rh[i] = norm(sts[:3])
		#delta[i] = np.linalg.norm(ste[:3])
		phase[i] = vecsep(ste[:3],sts[:3],directional=False)
		selon[i] = vecsep(sss[:3], ste[:3],directional=False)
		sunpa[i], suninc[i] = vecpa(ste[:3], -np.array(sts[:3]))
		velpa[i], velinc[i] = vecpa(ste[:3], sts[3:])
		if frame is not None:
			m = np.array(spice.sxform('j2000',frame,et[i]))[:3,:3]
			norpa.append(xyz2sph(m.dot([0.,0.,1.]))[1])

	from astropy import units
	rh *= units.km.to('au')
	delta *= units.km.to('au')

	tbl = Table([tstr, ra, dec, rh, delta, phase, selon, sunpa, suninc, velpa, velinc], names='utc ra dec rh range phase selong sunpa sunalt velpa velalt'.split())
	if frame is not None:
		tbl.add_column(Column(norpa, name='norpa'))

	if saveto is not None:
		tbl.write(saveto)

	return tbl


def obliquity(i, o, w, ra, dec, orbframe=True):
	'''Calculate obliquity of an object given its orbital elements
	(i, o) and pole orientation (ra, dec)

	i : inclination (deg)
	o : longitude of ascending node (deg)
	w : argument of periapsis (deg)
	ra, dec: pole orientation (deg)
	orbframe : set return type

	If `orbframe'=True, then returns (obliquity, TA), where TA is the true
	anomaly of solstice for the positive pole hemisphere.
	Otherwise returns vector(s) whose norm(s) is/are the obliquity values and
	the directions pointing to the solstice in (RA, Dec) for the positive pole
	hemisphere.
	np.broadcasting rules apply.

	v1.0.0 : 12/14/2016, JYL @PSI
	'''
	orbp = orbpole(i, o)
	b = np.broadcast(ra, dec)
	bshape = b.shape
	rotp = Vector([Vector(1.,r,d,deg=True,type='geo') for r,d in b]).reshape(bshape)
	b = np.broadcast(i, o, rotp.x, rotp.y, rotp.z)
	ob = orbp.vsep(rotp, directional=False, deg=True)
	line = orbp.cross(orbp.cross(rotp))
	vob = Vector(ob, line.lon, line.lat, type='geo')
	if orbframe:
		o = np.deg2rad(o)
		i = np.deg2rad(i)
		w = np.deg2rad(w)
		b = np.broadcast(vob.x, vob.y, vob.z, o, i, w)
		bshape = b.shape
		if bshape != ():
			vob = Vector([VectRot(eularm(a,b,c))*Vector(x,y,z) for x,y,z,a,b,c in b]).reshape(bshape)
		else:
			vob = VectRot(eularm(o,i,w))*vob
		return vob.norm(), np.rad2deg(vob.lon)
	else:
		return vob


def orb2pos(a, e, i, o, w, m, anomaly='mean'):
	'''Calculate position from orbital elements

	a : semi-major axis
	e : eccentricity
	i : inclination (deg)
	o : longitude of ascending node (deg)
	w : argument of periapsis (deg)
	m : anomaly angle (deg)
	anomaly : specifies what kind of anomaly, 'mean', 'true', or 'eccentric'

	Returns the position vector(s) in type Vector.  np.broadcasting rules apply.

	v1.0.0 : 12/14/2016, JYL @PSI
	'''
	m = np.deg2rad(m)
	if anomaly == 'mean':
		b = np.broadcast(m, e)
		bshape = b.shape
		if bshape != ():
			nu = np.array([spiceypy1.kepleq(x, 0, y) for x, y in b]).reshape(bshape)
		else:
			nu = spiceypy1.kepleq(m, 0, e)
	else:
		nu = m
	b = np.broadcast(a,e,nu)
	rr = np.empty(b.shape)
	rr.flat = [x*(1-y*y)/(1+y*np.cos(z)) for x,y,z in b]
	xx = rr*np.cos(nu)
	yy = rr*np.sin(nu)
	zz = np.zeros_like(xx)
	vv = Vector(xx,yy,zz)
	b = np.broadcast(vv.x, vv.y, vv.z, o, i, w)
	bshape = b.shape
	if bshape != ():
		vv2 = Vector([Vector(x,y,z).eular(a,b,c) for x,y,z,a,b,c in b]).reshape(bshape)
	else:
		vv2 = vv.eular(o, i, w)
	return vv2


def orbpole(i, o):
	'''Calculate orbital pole from orbital elements

	i : inclination (deg)
	o : longitude of ascending node (deg)

	Returns the pole vector.  np.broadcasting rules apply.

	v1.0.0 : 12/14/2016, JYL @PSI
	'''
	b = np.broadcast(o, i)
	bshape = b.shape
	if bshape != ():
		return Vector([Vector(0,0,1.).eular(x, y, 0) for x, y in b]).reshape(bshape)
	else:
		return Vector(0,0,1.).eular(o, i, 0)
