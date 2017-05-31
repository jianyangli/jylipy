'''Planetary constants
'''

from astropy.constants.constant import Constant

# Gaussian gravitational constant
kG = Constant('kG', 'Gaussian gravitational constant', 1.9909836747685184e-07, '1/s', 0., 'http://en.wikipedia.org/wiki/Gaussian_gravitational_constant','si')

# Sun
rho_sun = Constant('D_sun', 'Solar mean density', 1.408, 'g/cm3', 0., 'JPL Horizons 2013-Jul-31','si')
g_sun = Constant('g_sun', 'Solar surface gravity', 274.0, 'm/s2', 0., 'JPL Horizons 2013-Jul-31','si')
P_sun = Constant('P_sun', 'Solar sidereal period', 25.38, 'day', 0., 'JPL Horizons 2013-Jul-31','si')
F_sun = Constant('F_sun', 'Solar constant (1 AU)', 1367.6, 'W/m2', 0., 'JPL Horizons 2013-Jul-31','si')
Ve_sun = Constant('Ve_sun', 'Solar surface escape velocity', 617.7, 'km/s', 0., 'JPL Horizons 2013-Jul-31','si')
T_sun = Constant('T_sun', 'Solar effective temperature', 5778, 'K', 0., 'JPL Horizons 2013-Jul-31','si')
Tpb_sun = Constant('Tpb_sun', 'Solar photosphere bottom temperature', 6600, 'K', 0., 'JPL Horizons 2013-Jul-31','si')
Tpt_sun = Constant('Tpt_sun', 'Solar photosphere top temperature', 4400, 'K', 0., 'JPL Horizons 2013-Jul-31','si')

# Mars
R_mar = Constant('R_mar', 'Mars mean radius', 3389.92, 'km', 0.04, 'JPL Horizons 2012-Sep-28','si')
M_mar = Constant('M_mar', 'Mars mass', 6.4185e23, 'kg', 0., 'JPL Horizons 2012-Sep-28','si')
P_mar = Constant('P_mar', 'Mars sidereal period', 24.622962, 'h', 0., 'JPL Horizons 2012-Sep-28','si')

# Ceres
M_ceres = Constant('M_ceres', 'Ceres mass', 9.4e20, 'kg', 0.1e20, 'Viateau, B. & Rapport, N. (2001) AA 370, 602-609.  Michalak, G. (2001), AA 360, 363-374.','si')
R_ceres = Constant('R_ceres', 'Ceres mean radius', 476.2, 'km', 1.7, 'Thomas, P.C., et al. (2005) Nature 437, 224-226','si')
rho_ceres = Constant('D_ceres', 'Ceres density', 2077., 'kg/m3', 36., 'Thomas, P.C., et al. (2005) Nature 437, 224-226','si')
P_ceres = Constant('P_ceres', 'Ceres sidereal period', 9.074170, 'h', 0.000002, 'Chamberlain, M.A., Sykes, M.V., Esquerdo, G.A. (2007) Icarus 188, 451-456.','si')

# Vesta
M_vesta = Constant('M_vesta', 'Vesta mass', 2.59076e20, 'kg', 0.00001e20, 'Russell, C.T., et al. (2012) Science 336, 684-686.','si')
R_vesta = Constant('R_vesta', 'Vesta mean radius', 262.7, 'km', 0.1, 'Russell, C.T., et al. (2012) Science 336, 684-686.','si')
D_vesta = Constant('D_vesta', 'Vesta density', 3456., 'kg/m3', 35., 'Russell, C.T., et al. (2012) Science 336, 684-686.','si')
P_vesta = Constant('P_vesta', 'Vesta sidereal period', 5.3421277, 'h', 0.0000001, 'Russell, C.T., et al. (2012) Science 336, 684-686.','si')
