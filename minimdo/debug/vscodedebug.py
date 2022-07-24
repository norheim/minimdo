from compute import Par
from datastructures.api import adda, Model, Component
# model = Model()
# m = model.root
mu_batt = Par(r'\mu_{battery}', 200, 'W*hr/kg') # Lithium ion
C = Par('C', 5.294, unit='kW*hr')
m_batt_zero = Par('m_{batt_zero}', 5, 'kg')
mbatt = Par('m_{batt}', 31.47142857, 'kg')
c = Component.fromsympy(-mu_batt*m_batt_zero+mu_batt*mbatt-C, C, ignoretovar=True, component=36)
print(c.evaldict({'m_{batt}': 31.47142857, 'm_{batt_zero}':5, '\\mu_{battery}':200, 'C':5.29428571}))