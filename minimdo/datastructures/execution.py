from collections import defaultdict
from datastructures.graphutils import merge_edges
import numpy as np
#import autograd.numpy as anp
import jax.numpy as anp
import sympy as sp
#from autograd import jacobian
from jax import jacobian
from datastructures.unitutils import fx_with_units
from compute import ureg

import jax
jax.config.update('jax_platform_name', 'cpu')

# The following class emulates being a dictionary for sympys lambdify to work
# with autograd
math_functions = ['cos', 'sin', 'tan', 'sqrt', 'exp', 'log', 'log2', 'log10']
math_functions_dict = {'acos': 'arccos', 'asin':'arcsin', 'atan':'arctan'}

anp_math = {elt: getattr(anp, elt) for elt in math_functions}
anp_math.update({key:getattr(anp, val) for key,val in math_functions_dict.items()})

def grad_key_hide_none(outvr,invr):
    return (outvr,invr) if outvr else invr

def generate_grad(fx, inputs, outputs, indims, outdims):
    f = lambda x: anp.hstack(fx(*anp.split(x, anp.cumsum(anp.asarray(indims[:-1])))))
    g = jacobian(f)
    def getgrad(*inargs):
        ins = np.hstack(inargs).astype(float)
        jout = g(ins)
        outsplit = zip(outputs, np.split(jout, outdims[:-1], axis=0))
        insplit = lambda grads: zip(inputs, np.split(grads, np.cumsum(anp.asarray(indims[:-1])), axis=1))
        return {grad_key_hide_none(outvr, invr):np.squeeze(grad) for outvr,grads in outsplit for invr,grad in insplit(grads)}
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

def sympy_fx_inputs(expr, library=None):
    inputs = list(expr.free_symbols)
    inpunitsflat = tuple(inpvar.varunit for inpvar in inputs)
    modules = library if library is not None else anp_math
    fx = sp.lambdify(inputs, expr, modules)
    return fx, inputs, inpunitsflat

def sympy_fx_with_units():
    pass

def component_hash(algebraic_expr, outputs):
    return hash(str(algebraic_expr))+hash(outputs)

class Component():
    def __init__(self, fx, inputs=None, outputs=None, component=None, indims=None, outdims=None, fxdisp=None, arg_mapping=None):
        #assert len(set(inputs).intersection(outputs))==0
        self.inputs = inputs #Should have a test for inputs as numbers
        self.indims = indims if indims else tuple(1 for elt in inputs)
        self.outputs = outputs if outputs else (None,)
        self.outdims = outdims if outdims else tuple(1 for elt in self.outputs)
        self.component = component
        self.fxdisp = fxdisp
        self.function = fx
        # TODO: this needs to be moved over to openMDAO code
        self.mapped_names = [arg_mapping[inp] for inp in inputs] if arg_mapping and inputs else self.inputs
        self.mapped_outputs = [arg_mapping.get(out,None) for out in self.outputs] if arg_mapping else self.outputs
        self.gradient = generate_grad(fx, self.mapped_names, self.mapped_outputs, self.indims, self.outdims) if fx else None
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        if self.fxdisp:
            return component_hash(self.fxdisp, self.outputs)
        else:
            return object.__hash__(self)

    @classmethod
    def withunits(cls, fx, inputs, outputs, inunitmap, outunitmap, component=None):
        inpunitsflat = tuple(ureg(inunitmap[inpvar]) for inpvar in inputs)
        outunitsflat = tuple(ureg(outunitmap[outvar]) for outvar in outputs)
        fx_scaled = fx_with_units(fx, inpunitsflat, outunitsflat)
        return cls(fx_scaled, inputs, outputs, component)

    @classmethod
    def from_lambda(cls, lfx, **kwargs):
        inputs = inputs if kwargs.pop('inputs', False) else lfx.__code__.co_varnames
        return cls(lfx, inputs, **kwargs)

    @classmethod
    def fromsympy(cls, expr, tovar=None, ignoretovar=False, component=None, arg_mapping=None):
        fx, inputs, inpunitsflat = sympy_fx_inputs(expr)
        output_names = (tovar.varid,) if tovar and not ignoretovar else (None,) 
        if not isinstance(type(expr), sp.core.function.UndefinedFunction):# and hasattr(expr, 'dimensionality'): 
            # this second conditions can be dangerous but need it to fix something
            outunitsflat = (tovar.varunit,) if tovar else (None,)
            unitoverride = tovar.forceunit if tovar else False
            fxforunits = sp.lambdify(inputs, expr, "numpy")
            fx = fx_with_units(fx, inpunitsflat, outunitsflat, unitoverride, fxforunits) 
        input_names = tuple(inp.varid for inp in inputs)
        return cls(fx, input_names, output_names, component, fxdisp=expr, arg_mapping=arg_mapping)

    def evaldict(self, indict):
        return self.function(*args_in_order(indict, self.mapped_names))
    
    def graddict(self, indict):
        args = np.array(args_in_order(indict, self.mapped_names))
        return self.gradient(args)

    def __repr__(self):
        return str((self.inputs, self.component, self.outputs, str(self.fxdisp)))

def comp_id_lookup(comps):
    comp_ids = defaultdict(list)
    for comp in comps:
        comp_ids[comp.component].append(comp)
    return {key: var[0] if len(var)==1 else var for key,var in comp_ids.items()}

def edges_from_components(comps):
    Ein,Eout = dict(),dict()
    for comp in comps:
        Ein[comp.component] = comp.inputs
        Eout[comp.component] = comp.outputs
    return Ein, Eout, dict()

def newfx(c, *x):
    fxval = c.function(*x[:sum(c.indims)])
    outval = x[sum(c.indims):]
    out = [outval[idx]-elt for idx,elt in enumerate(fxval)]
    # IMPORTANT: flatten the output!
    return [elt for vector in out for elt in vector]

# TODO: this now lives in transformations

def residual_component(c, idx=0):
    newinputs = c.inputs + c.outputs
    newindims = c.indims + c.outdims
    fx = lambda *x: newfx(c, *x)
    return Component(fx, newinputs, (None,), idx, newindims, c.outdims)

def generate_components_and_residuals(components, edges):
    rcomps = [residual_component(c, c.component) for c in components if c.component in edges[2].keys()]
    return components+rcomps