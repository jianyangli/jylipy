# test area to mass ratio calculation

sfd = SFDModel(5, rho=1000*u.kg/u.m**3)
r1 = 0.001 * u.m
r2 = 1 * u.m

m = sfd.mass_integrated(r1, r2)
a = sfd.area_integrated(r1, r2)
ratio1 = a/m
ratio2 = sfd.area_mass_ratio(0.001*u.m, 1*u.m)
print('mass = {:.5g}, area = {:.5g}'.format(m, a))
print('ratio 1 = {:.5g}, ratio 2 = {:.5g}, difference {:.5g}'.format(ratio1, ratio2, ratio1-ratio2))


# expected output:
# mass = 4.1846e+06 kg, area = 1.5708e+06 m2
# ratio 1 = 0.37537 m2 / kg, ratio 2 = 0.37537 m2 / kg, difference 0 m2 / kg



# test SFDModel
%matplotlib widget

sfd = SFDModel(5, rho=1000*u.kg/u.m**3)
r = np.logspace(-2, 5, 100) * u.m
n = sfd(r)
plt.figure(figsize=(6,4))
plt.plot(r, sfd(r))
plt.plot(r, sfd.N_integrated(r, np.inf*u.m))
plt.plot(r, sfd.mass_integrated(r, np.inf*u.m))
plt.plot(r, sfd.area_integrated(r, np.inf*u.m))
jp.pplot(xscl='log',yscl='log')
plt.vlines(1, 1e-8,1e4, ls='--')
plt.legend(['SDF', 'Total N', 'Total mass', 'total area'])

# expected output in figure ScalingLaw_test.png



# test ejecta model

%matplotlib widget

a = 1.6 * u.mm
delta = 2700 * u.kg/u.m**3
V = 4/3 * np.pi * a**3
m = delta * V
U = 6200 * u.m/u.s
impactor = Impactor(m, a, U)

rho = 3000 * u.kg/u.m**3
g = 9.8 * u.m/u.s**2
Y = 30 * u.MPa
target = Target(rho, g, Y)

H2 = 1.1
mu = 0.55
nu = 0.4
regime='strength'
scl = ScalingLaw(mu, nu, H2=H2, impactor=impactor, target=target, regime=regime)
print(scl.R.to('cm'))

scl.C1 = 1.5
scl.p = 0.5
scl.n2 = 1

x = np.linspace(1.2*scl.impactor.a, scl.R, 100)
v = u.Quantity([scl.v(i) for i in x])
plt.figure(figsize=(6,4))
plt.plot(x, v)
jp.pplot(xscl='log',yscl='log',xlabel='x ('+str(x.unit)+')', ylabel='Ejecta Speed ('+str(v.unit)+')')

scl.k = 0.3
scl.n1 = 1.2
M = u.Quantity([scl.M(i) for i in x])
plt.figure(figsize=(6,4))
plt.plot(x, M)
jp.pplot(xscl='log',yscl='log',xlabel='x ('+str(x.unit)+')', ylabel='Ejecta Mass ('+str(M.unit)+')')

# expected output in figure ScalingLaw_test2.png and ScalingLaw_test3.png
# expected printout:
# 2.6326187529336997 cm


