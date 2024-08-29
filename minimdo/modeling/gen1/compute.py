from units import evaluable_with_unit
from autograd import grad
import autograd.numpy as anp
import numpy as np
import sympy as sp

def args_in_order(name_dict, names):
    return [name_dict[in_var] for in_var in names if in_var in name_dict]

def fill_args(args, input_names, attrcheck='varval'):
    fxargs = []
    idx = 0
    for elt in input_names:
        if getattr(elt, attrcheck):
            fxargs.append(elt.varval)
        else:
            fxargs.append(args[idx])
            idx+=1
    return fxargs

def partialfx(fx, input_names):
    def wrapper(*args, **kwargs):
        partial = kwargs.get('partial', None)
        if partial:
            return fx(*fill_args(args, input_names, partial))
        else:
            return fx(*args)
    return wrapper

# The following class emulates being a dictionary for sympys lambdify to work
# with autograd
math_functions = ['cos', 'sin', 'tan', 'arccos', 'arcsin', 'arctan', 'sqrt', 
'exp', 'log', 'log2', 'log10']

anp_math = {elt: getattr(anp, elt) for elt in math_functions}

# Should be sympy agnostic
class Evaluable():
    def __init__(self, fx, input_names=None):
        input_names = input_names if input_names else fx.__code__.co_varnames
        self.input_vars = input_names
        self.input_names = list(map(str,input_names))
        self.fx = partialfx(fx, input_names)
        wrapped_fx = fx if len(input_names) == 1 else (
                lambda x: fx(*x)) #adapt to numpy
        self.jfx = grad(wrapped_fx)
        self.njfx = lambda *args: self.jfx(np.array(args).astype(float))

    @classmethod
    def fromsympy(cls, expr, tovar=None):
        input_names = list(expr.free_symbols)
        fx = sp.lambdify(input_names, expr, anp_math)
        if tovar and not isinstance(type(expr), sp.core.function.UndefinedFunction):# and hasattr(expr, 'dimensionality'): 
            # this second conditions can be dangerous but need it to fix something
            unitify = evaluable_with_unit(expr, input_names, tovar.varunit) 
            # this is to get the right multiplier, any untis checks will have been done during creation? 
            # TODO: check this
            fx = unitify(fx)
        return cls(fx, input_names)
    
    def evaldict(self, indict):
        return self.fx(*args_in_order(indict, self.input_names))
    
    def graddict(self, indict):
        args = np.array(args_in_order(indict, self.input_names))
        return self.jfx(args)