from graphutils import merge_edges
import numpy as np
import autograd.numpy as anp
import sympy as sp
from autograd import grad, jacobian
from unitutils import fx_with_units
from compute import ureg

# The following class emulates being a dictionary for sympys lambdify to work
# with autograd
math_functions = ['cos', 'sin', 'tan', 'arccos', 'arcsin', 'arctan', 'sqrt', 
'exp', 'log', 'log2', 'log10']

anp_math = {elt: getattr(anp, elt) for elt in math_functions}

def generate_grad(fx, inputs, outputs, indims, outdims):
    f = lambda x: anp.hstack(fx(*anp.split(x, anp.cumsum(indims[:-1]))))
    g = jacobian(f)
    def getgrad(*inargs):
        ins = np.hstack(inargs).astype(float)
        jout = g(ins)
        outsplit = zip(outputs, np.split(jout, outdims[:-1], axis=0))
        insplit = lambda grads: zip(inputs, np.split(grads, np.cumsum(indims[:-1]), axis=1))
        return {(outvr, invr):np.squeeze(grad) for outvr,grads in outsplit for invr,grad in insplit(grads)}
    return getgrad

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
    def __init__(self, fx, inputs=None, outputs=None, component=None, indims=None, outdims=None, fxdisp=None, arg_mapping=None):
        inputs = inputs if inputs else fx.__code__.co_varnames
        self.inputs = inputs
        self.indims = indims if indims else tuple(1 for elt in inputs)
        self.outputs = outputs
        self.outdims = outdims if outdims else tuple(1 for elt in outputs)
        self.component = component
        self.fxdisp = fxdisp
        self.function = fx
        self.gradient = generate_grad(fx, inputs, outputs, indims, outdims)
        self.mapped_names = [arg_mapping[inp] for inp in inputs] if arg_mapping else self.inputs

    @classmethod
    def withunits(cls, fx, inputs, outputs, inunitmap, outunitmap, component=None):
        inpunitsflat = tuple(ureg(inunitmap[inpvar]) for inpvar in inputs)
        outunitsflat = tuple(ureg(outunitmap[outvar]) for outvar in outputs)
        fx_scaled = fx_with_units(fx, inpunitsflat, outunitsflat)
        return cls(fx_scaled, inputs, outputs, component)

    @classmethod
    def fromsympy(cls, expr, tovar=None, ignoretovar=False, component=None, arg_mapping=None):
        inputs = list(expr.free_symbols)
        inpunitsflat = tuple(inpvar.varunit for inpvar in inputs)
        fx = sp.lambdify(inputs, expr, anp_math)
        output_names = (tovar.varid,) if tovar and not ignoretovar else (None,) 
        if tovar and not isinstance(type(expr), sp.core.function.UndefinedFunction):# and hasattr(expr, 'dimensionality'): 
            # this second conditions can be dangerous but need it to fix something
            outunitsflat = (tovar.varunit,)
            fx = fx_with_units(fx, inpunitsflat, outunitsflat) 
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