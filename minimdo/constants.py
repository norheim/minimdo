from compute import Par, ureg
from scipy import interpolate
import numpy as np

μ = Par('mu', 3.986005e14, 'm^3/s^2')
R = Par('R', 6378, 'km')
Q = Par('Q', 1367, 'W/m^2')
k = Par('k', 1.38064852e-23, 'J/K')
c = Par('c', 2.99e8, 'm/s') #
G = Par('G', 9.81, 'm/s^2')

h_ρi = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 
                 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500])*1e3
ρi = np.array([4.79e-07, 1.81e-09, 2.53e-10, 6.24e-11, 1.95e-11, 6.98e-12, 2.72e-12, 
               1.13e-12, 4.89e-13, 2.21e-13, 1.04e-13, 5.15e-14, 2.72e-14, 1.55e-14, 
               9.63e-15, 6.47e-15, 4.66e-15, 3.54e-15, 2.79e-15, 1.11e-15, 5.21e-16])
Hi = np.array([5.9, 25.5, 37.5, 44.8, 50.3, 54.8, 58.2, 61.3, 64.5, 68.7, 74.8, 84.4, 
               99.3, 121, 151, 188, 226, 263, 296, 408, 516])*1e3
ρinterp = interpolate.interp1d(np.log(h_ρi), np.log(ρi))
Hinterp = interpolate.interp1d(np.log(h_ρi), np.log(Hi))
ρ_int = lambda h: ρi[0] if h<=h_ρi[0] else np.exp(ρinterp(np.log(h))) if h <= 1500e3 else 5.21e-16
H_int = lambda h: Hi[0] if h<=h_ρi[0] else np.exp(Hinterp(np.log(h))) if h <= 1500e3 else 516e3

# For balloon

# z_table = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40])*1e3
# T_table_celsius = np.array([15, 8.5,2,-4.49,-10.98,-17.47,-23.96,-30.45,-36.94,-43.42,-49.9,-56.5,-56.5,-51.6,-46.64,-22.8])
# T_table = ureg.Quantity(T_table_celsius, 'degC').to('K').magnitude
# G_table = np.array([9.807,9.804,9.801,9.797,9.794,9.791,9.788,9.785,9.782,9.779,9.776,9.761,9.745,9.73,9.715,9.684])
# P_table = np.array([10.13,8.988,7.95,7.012,6.166,5.405,4.722,4.111,3.565,3.08,2.65,1.211,0.5529,0.2549,0.1197,0.0287])*1e4
# ρ_table = np.array([1.225,1.112,1.007,0.9093,0.8194,0.7364,0.6601,0.59,0.5258,0.4671,0.4135,0.1948,0.08891,0.04008,0.01841,0.003996])

# Tinterp = interpolate.interp1d(z_table, T_table)
# Ginterp = interpolate.interp1d(z_table, G_table)
# Pinterp = interpolate.interp1d(z_table, P_table)
# ρinterp = interpolate.interp1d(z_table, ρ_table)