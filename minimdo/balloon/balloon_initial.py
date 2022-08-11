from compute import Var, Par
from datastructures.api import Model, adda, addsolver
from constants_balloon import ρinterp, Pinterp, Ginterp, Tinterp
import numpy as np

model = Model(rootname='root')
m = model.root
idbyname = True
z = Var('z', 30, unit='km')
R = Par('R', 287.05, 'J/kg/K')
alpha = Par(r'\alpha', 1)

apogee = addsolver(m, name='apogee', idbyname=idbyname)
pz = adda(apogee, 'p_z', lambda z: Pinterp(z), (z,), unit='Pa') # input assumed to be SI base (i.e. meter)
Tz = adda(apogee, 'T_z', lambda z: Tinterp(z), (z,), unit='K') # is in Kelvin already
gz = adda(apogee, 'g_z', lambda z: Ginterp(z), (z,), unit='m/s^2') # is in Kelvin already
rhoz = adda(apogee, r'\rho_z', pz/(R*Tz))
k = Par('k', 1.38064852e-23, 'J/K')
mm_He = Par('M_{He}', 6.64e-27, 'kg')
mm_H2 = Par('M_{H2}', 1.66e-27, 'kg')
rho_LGz = adda(apogee, r'\rho_{LGz}', pz*(alpha*mm_He+(1-alpha)*mm_H2)/(k*Tz))
m_t = Var('m_t', unit='kg')
W_z = adda(apogee, 'W_z', gz*m_t, unit='N')
L_z = adda(apogee, 'L_z', W_z, unit='N')
V_z = adda(apogee, 'V_z', L_z/(gz*rhoz))
mrz = adda(apogee, 'm_{rz}', V_z*rho_LGz)

materials = addsolver(m, name='materials', idbyname=idbyname)
rz = adda(materials, 'r_z', (3*abs(V_z)/(4*np.pi))**(1/3), unit='m', forceunit=True)
hz = adda(materials, 'h_z', 2*(3/2)*rz)
p = Par('p', 8/5) #some parameter
S = adda(materials, 'S', 4*np.pi*(((rz**2)**p+2*abs(rz)**p*(abs(hz)/2)**p)/3)**(1/p))
t_LLDPE = Par('t_S', 25.4e-6*1, 'm')
rho_LLDPE = Par(r'\rho_S', 925, 'kg/m^3')
m_b = adda(materials, 'm_b', 2*(3/2)*S*t_LLDPE*rho_LLDPE)

m_b = Var('m_b', unit='kg')
mass = addsolver(m, name='mass', idbyname=idbyname)
m_vhc = Par('m_v', 4545, 'kg')
m_parafoil = Par('m_p', 500., 'kg')
mr0 = Var('m_{r0}', unit='kg')
#m_l = adda(mass, 'm_l', lambda m1,m2: max(m1,m2), (mrz, mr0), unit='kg')
m_l = adda(mass, 'm_l', mr0, unit='kg')
adda(mass, m_t, m_vhc+m_parafoil+m_l+m_b);

aero = addsolver(m, name='aerodynamics', idbyname=idbyname)
V_0 = Var('V_0', unit='m^3')
g = Par('g', Ginterp(0), unit='m/s^2')
rho_0 = Par(r'\rho_0', ρinterp(0), 'kg/m^3')
W_0 = adda(aero, 'W_0', g*m_t, unit='N')
L_0 = adda(aero, 'L_0', g*rho_0*V_0, unit='N')
D = adda(aero, 'D', lambda L_0, W_0: max(L_0-W_0,0), (L_0,W_0), unit='N') #adda(aero, 'D', L_0-W_0, unit='N')
C_D = Par('C_D', 0.47)
vr = Var('v', 6, 'm/s')
A_0 = adda(aero, 'A_0', 2*D/(C_D*rho_0*vr**2))

geom = addsolver(m, name='geometry', idbyname=idbyname)
r_0 = adda(geom, 'r_0', 1/np.pi*abs(A_0)**0.5, unit='m')
adda(geom, V_0, 4/3*np.pi*r_0**3, unit='m^3')
rho_He = Par(r'\rho_{He}', 0.1786, 'kg/m^3')
rho_H2 = Par(r'\rho_{H2}', 0.08988, 'kg/m^3')
rho_LG0 = adda(geom, r'\rho_{LG0}', alpha*rho_He+(1-alpha)*rho_H2)
adda(geom, mr0, V_0*rho_LG0)        