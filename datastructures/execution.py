from graphutils import merge_edges
import numpy as np
import autograd.numpy as anp
import sympy as sp
from autograd import grad
from unitutils import evaluable_with_unit

# The following class emulates being a dictionary for sympys lambdify to work
# with autograd
math_functions = ['cos', 'sin', 'tan', 'arccos', 'arcsin', 'arctan', 'sqrt', 
'exp', 'log', 'log2', 'log10']

anp_math = {elt: getattr(anp, elt) for elt in math_functions}

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

class Component():
    def __init__(self, fx, inputs=None, outputs=None, component=None, fxdisp=None, arg_mapping=None):
        inputs = inputs if inputs else fx.__code__.co_varnames
        self.inputs = inputs
        self.outputs = outputs
        self.component = component
        self.fxdisp = fxdisp
        self.function = partialfx(fx, inputs)
        wrapped_fx = fx if len(inputs) == 1 else (
                lambda x: fx(*x)) #adapt to numpy
        self.gradient = grad(wrapped_fx)
        self.njfx = lambda *args: self.gradient(np.array(args).astype(float))
        self.mapped_names = [arg_mapping[inp] for inp in inputs] if arg_mapping else self.inputs

    @classmethod
    def fromsympy(cls, expr, tovar=None, component=None, arg_mapping=None):
        inputs = list(expr.free_symbols)
        fx = sp.lambdify(inputs, expr, anp_math)
        output_names = (None,)
        if tovar and not isinstance(type(expr), sp.core.function.UndefinedFunction):# and hasattr(expr, 'dimensionality'): 
            output_names = (tovar.varid,)
            # this second conditions can be dangerous but need it to fix something
            unitify = evaluable_with_unit(expr, inputs, tovar.varunit) 
            # this is to get the right multiplier, any untis checks will have been done during creation? 
            # TODO: check this
            fx = unitify(fx)
        input_names = tuple(inp.varid for inp in inputs)
        return cls(fx, input_names, output_names, component, expr, arg_mapping)

    def evaldict(self, indict):
        return self.function(*args_in_order(indict, self.mapped_names))
    
    def graddict(self, indict):
        args = np.array(args_in_order(indict, self.mapped_names))
        return self.gradient(args)

    def __repr__(self):
        return str((self.inputs, self.component, self.outputs, str(self.fxdisp)))


def bindFunction(function, n_reversed):
    def residual(*x):
        return np.array(x[n_reversed:])-function(*x[0:n_reversed]) 
    return residual

def generate_residuals(Ein, Rin, f):
    residuals = dict()
    merged_edges = merge_edges(Ein,Rin) # this makes sure we get the same order as used during workflow generation
    for fx,ins in Rin.items():
        merged_ins = merged_edges[fx]
        fkey = (Rin[fx], Ein[fx]) #Rin encodes the old outputs
        function = f[fkey]
        n_reversed = len(ins)
        output_size = (None,)*n_reversed
        # need to do some local binding for residual function
        residuals[(merged_ins, output_size)] = bindFunction(function, n_reversed)
    return residuals