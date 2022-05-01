from compute import (Var, Model, adda, Evaluable,
    unit_conversion_factors)
import sympy as sp

m = Model()
th = Var('th', 10, unit='')
s = sp.sin(th)
