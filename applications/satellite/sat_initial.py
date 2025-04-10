from modeling.compute import Var, Par
from modeling.gen3.nesting import Model, adda, addsolver
from applications.satellite.constants import μ, R, Q, k, c, G, H_int, ρ_int
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import implemented_function

dBtoLinear = lambda db: 10**(db/10)
D_r = Par('D_r', 5.3, 'm')
L = Par('L', dBtoLinear(1+8.5+0.3+0.1)) #based on SMAD v3 page 567
T_s = Par('T_s', 135, 'K')
B = Par('B', 8, 'bit')
N = Par('N', 2e3, '')
eta = Par(r'\eta_c', 0.55)
l_v = Par('l_v', 500, 'nm')
f = Par('f', 2.2, 'GHz')
rho_T = Par(r'\rho_T', 500e3, 'kg*m')
rho_P = Par(r'\rho_P', 0.002e-3, 'kg/J')
P_l = Par('P_l', 12, 'W')
m_c = Par('m_c', 0.2, 'kg')
C_D = Par('C_D', 2.2)
I_sp = Par('Isp', 70, 's')
w_W = Par('w_W', 1000, 'rad/s')
c_W = Par('c_W', 1)
rho_M = Par(r'\rho_M', 11.4)
rho_P2 = Par(r'\rho_P2', 9/0.1)
M_B = Par('M_B', 7.96e15)
kp = Par('kp', 0.0002)

model = Model()
m = model.root
orbit = addsolver(m, name='orbit')
h = Var('h', 400, 'km') #
a = adda(orbit, 'a', h + R, unit='km')
T = adda(orbit, 'T', 2*np.pi*(a**3/μ)**0.5, unit='min')
g = adda(orbit, 'g', 1/np.pi*sp.acos(R/a), unit='')
d = adda(orbit, 'd', g+0.5)
r = adda(orbit, 'r', (h**2+2*R*h)**0.5, unit='km')

power = addsolver(m, name='power')
eta_A = Par(r'\eta_A', 0.3)
rho_A = Par(r'\rho_A', 10, 'kg/m^2') #
A = Var('A', 0.05, unit='m^2')
m_A = adda(power, 'm_A', rho_A*A, unit='kg')
P_c = adda(power, 'P_c', d*A*Q*eta_A, unit='W')
P_T = adda(power, 'P_T', P_c-P_l, unit='W') #hack
E_b = adda(power, 'E_b', P_c*T/d, unit='kJ')
rho_b = Par(r'\rho_b', 0.002, 'kg/kJ')
m_b = adda(power, 'm_b', rho_b*E_b, unit='kg')

payload = addsolver(m, name='payload')
X_r = Var('X_r', 5, 'm')
D_p = adda(payload, 'D_p', 1.22*l_v*h/X_r)
D = adda(payload, 'D', 2*np.pi*R*B*N/X_r, unit='GB')
rho_p = Par(r'\rho_p', 2, 'kg/m^1.5') 
#D_p, m_p = D_pi[payload], m_pi[payload]
m_p = adda(payload, 'm_p', rho_p*D_p**1.5, unit='kg')

comms = addsolver(m, name='comms')
b = adda(comms, 'b', D/(g*T), unit='MB/s')
λ_c = adda(comms, r'\lambda_c', c/f, unit='cm')
G_T = Par('G_T', dBtoLinear(16.5), '')
D_T = adda(comms, 'D_T', λ_c*(G_T/eta)**0.5/np.pi, unit='m')
rho_T = Par(r'\rho_T', 0.2, 'kg/m^1.5')
m_T = adda(comms, 'm_T', rho_T*D_T**1.5, unit='kg')
G_r = adda(comms, 'G_r', eta*(np.pi*D_r/λ_c)**2)
EN = adda(comms, 'EN', P_T*G_r*G_T/(L*k*T_s*b)*(λ_c/(4*np.pi*r))**2, unit='')

struct = addsolver(m, name='struct')
mt = Var('m_t', unit='kg')
eta_S = Par(r'\eta_S', 0.2)
m_s = adda(struct, 'm_s', eta_S*mt, unit='kg')

mass = addsolver(m, name='mass')
m_pr = Var('m_{pr}', 0.5, unit='kg')
adda(mass, mt, m_T+m_p+m_b+m_A+m_s+m_pr)

prop = addsolver(m, name='prop')
H = implemented_function(sp.Function('H'), H_int)
rho = implemented_function(sp.Function('rho'),  ρ_int)
L_min = Par('L_{min}', 10, 'yr')
m_pr = Var('m_{pr}', 0.5, unit='kg')
Hval = adda(prop, 'H_{val}', H(h*1e3), unit='m', forceunit=True)
rhoval= adda(prop, r'\rho_{val}', rho(h*1e3), unit='kg/m**3', forceunit=True)
Ln = adda(prop, 'L_n', Hval*mt/(2*np.pi*C_D*A*rhoval*a**2)*T, unit='yr')
Lp = adda(prop, 'L_p', m_pr*I_sp*G*a/(0.5*C_D*A*rhoval*μ), unit='yr')
Lt = adda(prop, 'L_t', Ln+Lp, unit='yr')