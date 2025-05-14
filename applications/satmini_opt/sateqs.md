# Parameters with units
$$
D_r = 5.3 m           \\ % D_r (m)
T_s = 135 K           \\ % T_s (K)
B = 8 bit             \\ % B (bit)
N = 2e3               \\ % N
\eta_c = 0.55         \\ % eta_c
l_v = 500 nm          \\ % l_v (nm)
f = 2.2 GHz           \\ % f (GHz)
\rho_T = 500e3 kg*m   \\ % rho_T (kg*m)
\rho_P = 0.002e-3 kg/J\\ % rho_P (kg/J)
P_l = 12 W            \\ % P_l (W)
m_c = 0.2 kg          \\ % m_c (kg)
C_D = 2.2             \\ % C_D
I_{sp} = 70 s         \\ % I_sp (s)
w_W = 1000 rad/s      \\ % w_W (rad/s)
c_W = 1               \\ % c_W
\rho_M = 11.4         \\ % rho_M
\rho_{P2} = 9/0.1     \\ % rho_P2
M_B = 7.96e15         \\ % M_B
k_p = 0.0002          \\ % k_p
\rho_p = 2 kg/m^1.5   \\ 
\eta_S = 0.2          \\
$$

# Empirical models
$$
h_table (x 10^3): 
[100,  150,  200,  250,  300,  350,  400,  450,  500,  550,  
 600,  650,  700,  750,  800,  850,  900,  950, 1000, 1250,
1500]

rho_table:
[4.79e-07, 1.81e-09, 2.53e-10, 6.24e-11, 1.95e-11, 6.98e-12,
 2.72e-12, 1.13e-12, 4.89e-13, 2.21e-13, 1.04e-13, 5.15e-14,
 2.72e-14, 1.55e-14, 9.63e-15, 6.47e-15, 4.66e-15, 3.54e-15,
 2.79e-15, 1.11e-15, 5.21e-16]

H_table (x 10^3):
[  5.9,  25.5,  37.5,  44.8,  50.3,  54.8,  58.2,  61.3,  64.5,  68.7,
  74.8,  84.4,  99.3, 121.0, 151.0, 188.0, 226.0, 263.0, 296.0, 408.0,
 516.0]

rho_interp(x) = interpolate(x, log(h_table), log(rho_table))
H_interp(x) = interpolate(x, log(h_table), log(H_table))
\rho(h) = rho_table[0] if h <= h_table[0] else exp(rho_interp(log(h))) if h <= 1500 * 10^3 else 5.21e-16
H(h) = H_table[0] if h <= h_table[0] else exp(H_interp(log(h))) if h <= 1500 * 10^3 else 516 * 10^3
$$

## Functions
$$
% dB to linear conversion
dBtoLinear(db) = 10^{\frac{db}{10}} \\
$$

## Equations 
$$
% SUBPROBLEM: Orbit equations
a [km] = h + R \\
T [min] = 2\pi\sqrt{\frac{a^3}{\mu}} \\ 
g = \frac{1}{\pi}\arccos{\frac{R}{a}}\\
d = g + 0.5 \\
r [km] = \sqrt{h^2 + 2Rh} \\

% SUBPROBLEM: Power equations
m_A [kg] = \rho_A \cdot A \\
P_c [W] = d \cdot A \cdot Q \cdot \eta_A \\
P_T [W] = P_c - P_l \\ 
E_b [kJ] = P_c \cdot \frac{T}{d} \\
m_b [kg] = \rho_b \cdot E_b \\

% SUBPROBLEM: Payload equations
D_p = \frac{1.22 \cdot l_v \cdot h}{X_r} \\
D [GB] = \frac{2 \cdot \pi \cdot R \cdot B \cdot N}{X_r} \\
m_p [kg] = \rho_p \cdot D_p^{1.5} \\

% SUBPROBLEM: Comms equations
b [MB/s] = \frac{D}{g \cdot T} \\
\lambda_c [cm]= \frac{c}{f} \\
G_T = dBtoLinear(16.5) \\
D_T [m] = \frac{\lambda_c \cdot (\frac{G_T}{\eta})^{0.5}}{\pi} \\
m_T [kg] = \rho_T \cdot D_T^{1.5} \\
G_r = \eta \cdot (\frac{\pi \cdot D_r}{\lambda_c})^2 \\
EN = \frac{P_T \cdot G_r \cdot G_T}{L \cdot k \cdot T_s \cdot b} \cdot \left(\frac{\lambda_c}{4 \cdot \pi \cdot r}\right)^2 \\

% SUBPROBLEM: Struct equations
m_s [kg] = \eta_S \cdot m_t \\

% SUBPROBLEM: Mass equations
m_t = m_T + m_p + m_b + m_A + m_s + m_{pr} \\

% Prop equations
H_{val} = H(h \cdot 10^3) \\
\rho_{val} = \rho(h \cdot 10^3) \\
L_n = \frac{H_{val} \cdot m_t}{2 \cdot \pi \cdot C_D} 
$$