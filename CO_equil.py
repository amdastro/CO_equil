import numpy as np
import pylab as plt
from scipy.optimize import newton

plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2

#----- Physical constants ------#
mp = 1.67e-24
kB = 1.38e-16
h = 6.626e-27
mu = 1.e-23 # reduced mass of C and O
B_CO = 1.78e-11
Msun = 1.9e33
gamma = 1.3

#-------- Arrays ----------#
t = np.linspace(1e4,1e7, 1e4)
t_exp = np.zeros_like(t)
T_cs = np.zeros_like(t)
n = np.zeros_like(t)
c_s = np.zeros_like(t)
R_cs = np.zeros_like(t)
delta = np.zeros_like(t)
n_CO = np.zeros_like(t,dtype='float64')
Y_clayton = np.zeros_like(t)
n_clayton = np.zeros_like(t)
n_C_free = np.zeros_like(t)
n_O_free = np.zeros_like(t)
Y_CO = np.zeros_like(t)
Y_C_free = np.zeros_like(t)
Y_O_free = np.zeros_like(t)
K_ra = np.zeros_like(t,dtype='float64')
K_rd = np.zeros_like(t,dtype='float64')
K_th = np.zeros_like(t,dtype='float64')
K_nth = np.zeros_like(t,dtype='float64')

#------ Initial Parameters ------#
# The total mass of the cold shell and the ejecta velocity
M_cs = 2e-4*Msun 
v_ej = 1e8 

# The initial values for the cold shell Temp, radius, thickness, and density:
T_cs[0] = 1e4
c_s[0] = np.sqrt(kB*T_cs[0]/(mp))
R_cs[0] = 1e14
delta[0] = R_cs[0]/4000.
n[0] = M_cs/(4.*np.pi*mp*R_cs[0]**2*delta[0])

# The total number fraction of C and O atoms is fixed.
Y_C_tot = 0.4
Y_O_tot = 0.6
# Initially all C and O is free: 
Y_C_free[0] = Y_C_tot
Y_O_free[0] = Y_O_tot


# Defining functions for reaction rates: 
def K_rad(T):
	# from Lazzati:
	K = 4.467e-17/((T/4467.)**(-2.08) + (T/4467.)**(-0.22))**0.5
	# from T&F: 
	#K = (1.25)*1e-16
	return K

def K_therm(T):
	K = 4.467e-17/((T/4467.)**(-2.08) + (T/4467.)**(-0.22))**0.5\
		*(h**2 / (2.*np.pi*mu*kB*T))**(-1.5) * np.exp(-B_CO/(kB*T))
	return K

# In the future K_nonthermal will depend on time, but 
# for now it is constant.
# observed gamma ray luminosity:
L_gamma = 3e35 
epsilon = 1.
# energy per particle:
W_d = 2.4e-10
K_nontherm = epsilon*L_gamma*mp/M_cs/W_d

# Equilibrium density of CO from Clayton 2013 (to compare): 
def n_clay(n_C, n_O, T) :
	nCO = n_C*n_O * (h**2 / (2.*np.pi*mu*kB*T))**(1.5) * np.exp(B_CO/(kB*T))
	return nCO

# Solution to quadratic eq for equilibrium number fraction of CO:
# Y_C and Y_O are the *total* number fractions.
# Although Y_C + Y_O = 1 here, I keep them separate since 
# this will not always be true. 
def Y_CO_equil(Y_C, Y_O, n, Kra, Krd):
	YCO = 0.5*(Y_C + Y_O + Krd/Kra/n) - \
		0.5*((Y_C + Y_O + Krd/Kra/n)**2 - 4.*Y_C*Y_O)**0.5
	return YCO
 

K_ra[0] = K_rad(T_cs[0])
K_th[0] = K_therm(T_cs[0])
K_nth[0] = K_nontherm
# K_rd is the sum of both destruction rates.
K_rd[0] = K_th[0] + K_nth[0]

Y_CO[0] = Y_CO_equil(Y_C_tot,Y_O_tot,n[0],K_ra[0],K_rd[0])
Y_C_free[0] = Y_C_tot - Y_CO[0]
Y_O_free[0] = Y_O_tot - Y_CO[0]


# Loop over time (with constant dt for now): 
for i in range(1,len(t)):

	# Expand the shell: 
	R_cs[i] = R_cs[0] + v_ej*t[i]
	delta[i] = delta[0] + c_s[i-1]*t[i]
	if t[i] < (R_cs[0]**2 * delta[i]/v_ej**3)**(1./3.): 
		n[i] = M_cs/(4.*np.pi*mp*R_cs[0]**2 * delta[i])
		t_exp[i] = t[i]/2.
	else:
		n[i] = M_cs/(4.*np.pi*mp*(v_ej*t[i])**3)
		t_exp[i] = t[i]/3.
	# Temperature and sound speed decrease: 
	T_cs[i] = T_cs[0]*(n[i]/n[0])**(gamma-1.)
	c_s[i] = np.sqrt(kB*T_cs[i]/(mp))

	# Expand the shell: 
	#R_cs[i] = R_cs[0] + v_ej*t[i]
	#delta[i] = delta[0] + c_s[i-1]*t[i]
	#n[i] = M_cs/(4.*np.pi*mp*R_cs[i]**2 * delta[i])
	# Temperature and sound speed decrease: 
	#T_cs[i] = T_cs[0]*(n[i]/n[0])**(gamma-1.)
	# expansion timescale - T/Tdot
	#c_s[i] = np.sqrt(kB*T_cs[i]/(mp))

	# Save the rates: 
	K_ra[i] = K_rad(T_cs[i])
	K_th[i] = K_therm(T_cs[i])
	K_nth[i] = K_nontherm
	K_rd[i] = K_th[i] + K_nth[i]
	# Solve for the equilibrium density: 
	Y_CO[i] = Y_CO_equil(Y_C_tot, Y_O_tot, n[i], K_ra[i], K_rd[i])
	# From number fraction to total number density of CO: 
	n_CO[i] = Y_CO[i] * n[i]
	# number fraction of free C and free O may change: 
	Y_C_free[i] = Y_C_tot - Y_CO[i]
	Y_O_free[i] = Y_O_tot - Y_CO[i]
	n_C_free[i] = Y_C_free[i]*n[i]
	n_O_free[i] = Y_O_free[i]*n[i]
	#n_clayton[i] = n_clay(n_C_free[i], n_O_free[i], T_cs[i])
	#Y_clayton[i] = n_clayton[i]/n[i]
	if Y_C_free[i] + Y_O_free[i] + 2.*Y_CO[i] != 1.: break

#--------------------------------#

# destruction timescale
t_dest = 1./(K_rd)
# formation timescale
t_form = 1./(K_ra*n)


#plt.plot(t, 1./(K_rd)/t_exp)
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel('$t$')
#plt.ylabel(r'$t_{\rm eq} /t_{\rm exp}$')
#plt.savefig('t_eq.png')

plt.figure(figsize=(14,8))

plt.subplot(231)
plt.plot(t, n,color='black')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$t$')
plt.ylabel(r'$n_{\rm tot}$')

plt.subplot(232)
plt.plot(t, T_cs,color='black')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$t$')
plt.ylabel(r'$T_{\rm cs}$')

plt.subplot(233)
plt.plot(t, t_form/t, label=r'$t_{\rm form}/t$',color='blue')
plt.plot(t, t_dest/t, label=r'$t_{\rm dest}/t$',color='red')
plt.legend(loc=2)
plt.ylabel(r'$\tau$')
plt.xlabel(r'$t$')
plt.yscale('log')
plt.xscale('log')

plt.subplot(234)
plt.plot(t,Y_C_free,label=r'$Y_{\rm C}^{\rm free}$',color='#013ADF')
plt.plot(t,Y_O_free,label=r'$Y_{\rm O}^{\rm free}$',color='#04B45F')
plt.plot(t,Y_CO,label=r'$Y_{\rm CO}$',color='red')
plt.xlabel('$t$')
plt.ylabel(r'${\rm number \, fraction}$')
plt.xscale('log')
plt.ylim([-0.1,1.1])
plt.legend(loc=2)


plt.subplot(235)
#plt.plot(t,n_C_free, label=r'$n_{\rm C}$')
#plt.plot(t,n_O_free, label=r'$n_{\rm O}$')
plt.plot(t,n_CO, label=r'$n_{\rm CO}$',color='black')
#plt.plot(t,n_clayton, label=r'$n_{\rm Clayton}$')
plt.xlabel('$t$')
plt.ylabel(r'${\rm equilibrium} \, n_{\rm CO}$')
#plt.yscale('log')
plt.xscale('log')
#plt.legend(loc=2)


plt.subplot(236)
plt.plot(t,K_th,label=r'$K_{\rm therm}$',linestyle='--',color='orange')
plt.plot(t,K_nth,label=r'$K_{\rm nontherm}$',linestyle='--',color='magenta')
plt.plot(t,(K_rd), label=r'$K_{\rm rd}$',color='red')
plt.plot(t,(K_ra*n), label=r'$K_{\rm ra} n$',color='blue')
plt.xlabel('$t$')
plt.ylabel(r'$K \rm [s^{-1}]$')
plt.yscale('log')
plt.xscale('log')
plt.legend(loc=3)
plt.ylim([1e-80,1e10])


plt.tight_layout()
plt.show()