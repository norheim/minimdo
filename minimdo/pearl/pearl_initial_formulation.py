from compute import Var, Par
from datastructures.api import Model, adda, addsolver
from numpy import pi
from sympy import log, sqrt, cos

# Input variables
Df = Var('D_f', 2, unit='m')
Ds = Var('D_s', 1, unit='m')
Dd = Var('D_d', 1.25, unit='m')
tf = Var('t_f', 0.5, unit='m')
ts = Var('t_s', 2, unit='m')
td = Var('t_d', 0.5, unit='m')

# Global parameters
rhow = Par(r'\rho_w', 1023.6, 'kg/m^3') # density of seawater [kg/m^3]
rho = Par(r'\rho', 700, 'kg/m^3')       # arbitrarily set
rhoh = Par(r'\rho_h', 2700, 'kg/m^3')   # arbitrarily set

idbyname = True
model = Model(rootname='root')
m = model.root

# Geometry
geometry = addsolver(m, name='geometry', idbyname=idbyname)
A_s = Var('A_{solar}', unit='m^2')
alpha = Par(r'\alpha', 0.05)
adda(geometry, Df, (4*abs(A_s)/(pi*(1-alpha)))**0.5)
d = adda(geometry, 'd', alpha*Df)

# Hydrodynamics
hydro = addsolver(m, name='hydro', idbyname=idbyname)
## Local variable
hf = Var('h_f', 0.9*tf.varval, 'm')
## Local parameter
g = Par('g', 9.81, 'm/s^2')
Vd = adda(hydro, 'Vd', pi/4*(Df**2*hf+Ds**2*ts+Dd**2*td))
FB = adda(hydro, 'F_B', rhow*Vd*g, unit='N')
FW = adda(hydro, 'F_W', FB)
xd = td/2
xs = td+ts/2
xf1 = td+ts+hf/2
xf2 = td+ts+tf/2
totA1 = hf*Df+ts*Ds+td*Dd
totA2 = tf*Df+ts*Ds+td*Dd
KB = adda(hydro, 'K_B', (hf*Df*xf1+ts*Ds*xs+td*Dd*xd)/totA1) # z_B
KG = adda(hydro, 'K_G', (tf*Df*xf2+ts*Ds*xs+td*Dd*xd)/totA2) 
I = adda(hydro, 'I', pi/64*Df**4)
BM = adda(hydro, 'B_M', I/Vd)
#should be 3-5% according to http://web.mit.edu/13.012/www/handouts/Reading3.pdf
GM = adda(hydro, 'G_M', KB+BM-KG) 
mtot = Var('m_{platform}', unit='kg')

C33 = adda(hydro, 'C_{33}', pi*rhow*g/4*Df**2)
A33 = adda(hydro, 'A_{33}', 0.0525*pi*rhow*(Dd**3+Ds**3+Df**3))
omega0 = adda(hydro, r'\omega_0', (C33/(A33+mtot))**1/2, unit='rad/s', forceunit=True)

# Mass
mass = addsolver(m, name='mass', idbyname=idbyname)
adda(mass, mtot, FW/g, 'kg')
mbatt = Var('m_{batt}', unit='kg')
mprop = Par('m_{prop}', 50, 'kg') # mass of propulsion
mcomms = Par('m_{comms}', 50, 'kg') # mass of comms system
eta_solar = Par(r'\eta_{solar}', 10, 'kg/m^2')
msolar = adda(mass, 'm_{solar}', eta_solar*A_s)
mstruct = adda(mass, 'm_{struct}', mtot-mbatt-msolar-mcomms-mprop) 
adda(mass, td, (4/pi*mstruct-Df**2*tf*rho-Ds**2*ts*rho)/(Dd**2*rhoh))

# Propulsion
prop = addsolver(m, name='prop', idbyname=idbyname)
S_wd = pi*((Dd/2)**2-(Ds/2)**2+(Dd/2)**2+2*(Dd/2)*td)
S_ws = 2*pi*(Ds/2)*ts
S_wf = pi*((Df/2)**2-(Ds/2)**2+2*(Df/2)*hf)
S_w = adda(prop, 'S_w', S_wd+S_ws+S_wf, 'm**2') # wetted surface
C_d = Par('C_d', 1) # estimate drag coefficient (a square flat plate at 90 deg to the flow is 1.17)
v = Var('v', 1, unit='m/s') # [m/s]
eta_m = Par(r'\eta_m', 0.75) # estimated, need to determine from motors?
# D = rhow*C_d*S_w*v**2*0.5
P_move = adda(prop, 'P_{move}', rhow*C_d*S_w*v**3/(2*eta_m), unit='W')

# Communications
comms = addsolver(m, name='comms', idbyname=idbyname)
db2dec = lambda x: 10**(x/10)
dec2db = lambda x: 10*log(abs(x), 10)
k = Par('k', 1.38065e-23, 'J/K')
c = Par('c', 3e8, 'm/s')
f = Par('f', 2.2, 'GHz')
Lambda = adda(comms, r'\lambda', c/f, unit='m')
eta_parab = Par(r'\eta_{parab}', 0.55)
theta_t = Par(r'\theta_t', 32)
error_t = Par('e_t', 27)
#G_pt_db = adda(comms, 'G_{pt}', 44.3-dec2db(theta_t**2), forceunit=True)
L_pt_db = adda(comms, 'L_{pt}', -12*(error_t/theta_t)**2)
#G_t = adda(comms, 'G_t', db2dec(G_pt_db+L_pt_db))
G_t = adda(comms, 'G_t', eta_parab*(pi*d/Lambda)**2)
D_r = Par('D_r', 0.3, 'm')
G_r = adda(comms, 'G_r', eta_parab*(pi*D_r/Lambda)**2)
h = Par('h', 780, 'km')
Re = Par('Re', 6378, 'km')
S = adda(comms, 'S', sqrt(h*(h+2*Re)), unit='km', forceunit=True)
L_s = adda(comms, 'L_s', (Lambda/(4*pi*S))**2)
R = Var('R', 10, 'Mbit/s') # 1 per microseconds
T_s = Par('T_s', 135, 'K')
L_a = Par('L_a', db2dec(-0.3))
L_l = Par('L_l', db2dec(-1))
L_p = Par('L_p', db2dec(-0.1))
EN = Var('EN', 10)
Pcomms = adda(comms, 'P_{comms}', EN/(L_a*L_s*L_l*L_p*G_r*G_t)*(k*T_s*R), unit='W')

# Power
power = addsolver(m, name='power', idbyname=idbyname)
## Energy budget
# should t_move, t_comms, t_service and t_recharge add up to 24 hours?
t_mission = Par('t_{mission}', 24, 'hr')
t_comms = Par('t_{comms}', 1, 'hr')
t_move = Par('t_{move}', 1, 'hr')
t_service = Par('t_{service}', 12, 'hr')
t_recharge = Par('t_{recharge}', 12, 'hr')
P_hotel = Par('P_{hotel}', 50, 'W')
E_AUV = Par('E_{AUV}', 1.9, 'kW*hr') # AUV battery capacity (to be recharged), based on Bluefin-9
gamma = Par(r'\gamma', 1) # AUVs serviced per mission duration (aka in the time of t_mission)
E_move = adda(power, 'E_{move}', P_move*t_move, unit='kW*hr')
E_hotel = adda(power, 'E_{hotel}', P_hotel*t_mission, unit='kW*hr')
E_comms = adda(power, 'E_{comms}', Pcomms*t_comms, unit='W*hr')
E_service = adda(power, 'E_{service}', E_AUV * gamma, unit='kW*hr')
P_service = adda(power, 'P_{service}', E_service/t_service, unit='W')
E_required = adda(power, 'E_{required}', E_hotel+E_move+E_service+E_comms, unit='kW*hr')

## Solar panel sizing
E_recharge = adda(power, 'E_{recharge}', E_required, unit='kW*hr')
P_recharge = adda(power, 'P_{recharge}', E_recharge / t_recharge, unit='W')
eta_s = Par(r'\eta_s', 0.27)
phi_s = Par(r'\phi_s', 800, 'W/m**2')
theta_bar = Par(r'\theta', 55, 'deg')
Ideg = Par('I_{deg}', 0.9)
ddeg = Par('d_{deg}', 0.005)
Lsolar = Par('L_{solar}', 10) #lifetime in years, but units act weirdly in powers
adda(power, A_s, P_recharge / (eta_s * phi_s * cos(theta_bar) * Ideg * (1-ddeg)**Lsolar), unit='m^2', forceunit=True)

## Battery sizing
mu_batt = Par(r'\mu_{battery}', 200, 'W*hr/kg') # Lithium ion
DOD = Par('DOD', 0.7)
eta_batt = Par(r'\eta_{battery}', 0.85) # transmission efficiency
N = Par('N', 1)
C = adda(power, 'C', E_required/(DOD*N*eta_batt), unit='kW*hr')
m_batt_zero = Par('m_{batt_zero}', 5, 'kg')
adda(power, mbatt, C/mu_batt + m_batt_zero, unit='kg') # was already defined in the beginning
# Not needed parameters for now:
# V_batt = adda(power, 'V_{batt}', C/nu_batt, unit='m**3')
# nu_batt = Par(r'\nu_{battery}', 450, 'kW*hr/(m**3)')

