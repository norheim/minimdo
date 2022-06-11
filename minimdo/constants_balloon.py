from compute import Par, ureg
from scipy import interpolate
import numpy as np
import jax.numpy as anp

# For balloon

z_table = anp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40])*1e3
T_table_celsius = anp.array([15, 8.5,2,-4.49,-10.98,-17.47,-23.96,-30.45,-36.94,-43.42,-49.9,-56.5,-56.5,-51.6,-46.64,-22.8])
T_table = ureg.Quantity(T_table_celsius, 'degC').to('K').magnitude
G_table = anp.array([9.807,9.804,9.801,9.797,9.794,9.791,9.788,9.785,9.782,9.779,9.776,9.761,9.745,9.73,9.715,9.684])
P_table = anp.array([10.13,8.988,7.95,7.012,6.166,5.405,4.722,4.111,3.565,3.08,2.65,1.211,0.5529,0.2549,0.1197,0.0287])*1e4
ρ_table = anp.array([1.225,1.112,1.007,0.9093,0.8194,0.7364,0.6601,0.59,0.5258,0.4671,0.4135,0.1948,0.08891,0.04008,0.01841,0.003996])

Tinterp = lambda x: anp.interp(x, z_table, T_table)
Ginterp = lambda x: anp.interp(x, z_table, G_table)
Pinterp = lambda x: anp.interp(x, z_table, P_table)
ρinterp = lambda x: anp.interp(x, z_table, ρ_table)

# Tinterp = interpolate.interp1d(z_table, T_table)
# Ginterp = interpolate.interp1d(z_table, G_table)
# Pinterp = interpolate.interp1d(z_table, P_table)
# ρinterp = interpolate.interp1d(z_table, ρ_table)