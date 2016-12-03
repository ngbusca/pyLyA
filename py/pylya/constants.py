import scipy as sp
from scipy import interpolate

lya=1215.67 ## angstrom

deg = sp.pi/180.

nside = 256
nest=False


z0 = 2.25
alpha=3.8

nbins=10000
zmax=10.
dz = zmax/nbins
z=sp.array(range(nbins))*dz

Om = 0.315
H0=100.
hubble = H0*sp.sqrt(Om*(1+z)**3+1-Om)
c = 299792.4583

chi=sp.zeros(nbins)
for i in range(1,nbins):
    chi[i]=chi[i-1]+c*(1./hubble[i-1]+1/hubble[i])/2.*dz

r_comoving = interpolate.interp1d(z,chi)
chi = interpolate.interp1d(z,chi)
hubble = interpolate.interp1d(z,hubble)

cross_rpar_min = -200.
cross_rpar_max = 200.
cross_npar = 100

rper_max = 200.
nper = 50
