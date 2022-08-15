from compute import Par, Var
from datastructures.api import adda, Model, Component
import sympy as sp
# model = Model()
# m = model.root
V_0 = Var('V_0', unit='m^3')
r_0 = Var('r_0', unit='m')
c = Component.fromsympy(sp.Pow(V_0,1/3), tovar=r_0, component=20)
print(c)