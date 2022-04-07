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

def expression_conversion_unit(expr_unit, tounit=None):
    unit = tounit if tounit else ureg('')
    if tounit:
        assert(unit.dimensionality == expr_unit.dimensionality)
        conversion_unit = unit
    else: # tounit was not given
        if not hasattr(expr_unit, 'units'):
            conversion_unit = ureg('')
        else:
            # if we evaluate an expression and get some crazy unit 
            # we bring it back to it's base dimensionality
            conversion_unit = ureg.Quantity(1, 
                expr_unit.to_base_units().units)
    return conversion_unit

def get_unit_multiplier(unit):
    return unit.to_base_units().magnitude

def unit_conversion_factors(outunitpairs, inunits):
    convert = np.array([get_unit_multiplier(inp) for 
            inp in inunits])
    factors = []
    for outunit, tounit in outunitpairs:
        conversion_unit = expression_conversion_unit(outunit, tounit)
        factor = get_unit_multiplier(conversion_unit)
        factors.append(factor)
    return convert, factors

def listify(out):
    return out if isinstance(out, list) else [out]

def get_unit(fx, inputunits):
    args = (ureg.Quantity(MockFloat(1), inputunit) for inputunit in inputunits)
    dim = fx(*args)
    dims = listify(dim)
    # need this case if output is a float, which can happen when we have powers, e.g. 10^x:
    dims = [dim if isinstance(dim, ureg.Quantity) else ureg('') for dim in dims]
    return dims

def flatten_list(ls):
    return ls if np.isscalar(ls)==1 else list(ls)

def executable_with_conversion(convert, factors, fx):
    def scaled_fx(*args):
        return flatten_list(np.array(fx(*(convert*np.array(args).flatten())))/factors)
    return scaled_fx

def fx_with_units(fx, inunitsflat, outunitsflat):
    expr_units = get_unit(fx, inunitsflat)
    outunitpairs = tuple((outunit, outunitsflat[idx]) for idx, outunit in enumerate(expr_units))
    convert, factors = unit_conversion_factors(outunitpairs, inunitsflat)
    fx_scaled = executable_with_conversion(convert, factors, fx)
    return fx_scaled