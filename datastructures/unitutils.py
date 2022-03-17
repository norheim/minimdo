import sympy as sp
import numpy as np
from compute import Var, ureg

# The following class is a very hacky class that is used further down to recover the unit associated with a specific function. It overrides all standard operators in python
class MockFloat(float):
    def __new__(self, value):
        return float.__new__(self, value)
    def __init__(self, value):
        float.__init__(value)
    def __add__(self, other):
        return self
    def __sub__(self, other):
        return self
    def __mul__(self, other):
        return self
    def __truediv__(self, other):
        return self
    def __floordiv__(self, other):
        return self
    def __mod__(self, other):
        return self
    def __pow__(self, other):
        return self
    def __rshift__(self, other):
        return self
    def __lshift__(self, other):
        return self
    def __and__(self, other):
        return self
    def __or__(self, other):
        return self
    def __xor__(self, other):
        return self

def get_unit(expr):
    if isinstance(expr, Var):
        return expr.varunit
    else:
        free_symbols = list(expr.free_symbols)
        if free_symbols:
            fx = sp.lambdify(free_symbols, expr, np)
            args = (ureg.Quantity(MockFloat(1), free_symbol.varunit) for free_symbol in free_symbols)
            dim = fx(*args)
            # need this case rhs_unit is a float, which can happen when we have powers, e.g. 10^x
            if not isinstance(dim, ureg.Quantity):
                dim = ureg('')
            return dim
        else:
            return ureg('') #most likely we encountered a number

def get_unit_multiplier(unit):
    return unit.to_base_units().magnitude

def unit_conversion_factors(right, orig_unit, symb_order):
    unit = orig_unit if orig_unit else ureg('')
    rhs_unit = get_unit(right)
    convert = np.array([get_unit_multiplier(free_symbol.varunit) for 
            free_symbol in symb_order])
    if orig_unit:
        assert(unit.dimensionality == rhs_unit.dimensionality)
        conversion_unit = unit
    else: # unitstr was not given
        if not hasattr(rhs_unit, 'units'):
            conversion_unit = ureg('')
        else:
            conversion_unit = ureg.Quantity(1, 
                rhs_unit.to_base_units().units)

    factor = get_unit_multiplier(conversion_unit)
    return convert, factor

def evaluable_with_unit(right, symb_order, tovar=None):
    convert, factor = unit_conversion_factors(right, tovar, symb_order)
    def correction(fx):
        def rhsfx_corrected(*args):
            return fx(*(convert*np.array(args).flatten()))/factor
        return rhsfx_corrected

    return correction