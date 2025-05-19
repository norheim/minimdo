from engine.foundations.functionals import encode_sympy
from engine.foundations.functional_noobj import concatenate_residuals
from engine.foundations.functional_noobj import feed_forward
from engine.foundations.functional_noobj import residual_solver
from engine.foundations.functional_noobj import eliminate_vars
from modeling.gen4.arghandling import Encoding, decode, encode  
from modeling.gen4.compute import EncodedFunction  
from modeling.gen2.transformations import partial_inversion
from modeling.compute import Var
from itertools import chain
import sympy as sp
import numpy as np

def strategy_eliminate_feedfwd(residual, functional, solvevars, **kwargs):
    Relim = eliminate_vars(residual, functional)
    RS = residual_solver(Relim, Encoding(solvevars), 
                            **kwargs)
    combined = feed_forward((RS,functional))
    return combined

def feedback(projectables, solvevars=None, **kwargs):
    all_functions = ()
    residuals = ()
    invars = ()
    solvevars = solvevars if solvevars is not None else tuple()
    for projectable in projectables:
        F = projectable.functional()
        if len(set(F.decoder.order) & set(invars))==0:
            invars += tuple(elt for elt in F.encoder.order 
                        if elt not in invars)
            all_functions += (F,)
            res = () # no residual
        else:
            solvevars += tuple(elt for elt in F.decoder.order 
                                if elt not in solvevars)
            res = (projectable.residuals(),)
        residuals += res
    R = concatenate_residuals(residuals)
    F = feed_forward(all_functions)
    return F, R, solvevars

def restructure(sympy_equations, new_output):
    new_equations = ()
    for idx, (var, right) in enumerate(sympy_equations):
        new_var = new_output.get(idx, None)
        res_sympy = right
        if new_var != var:
            res_sympy = partial_inversion(right, 
                                        old_output=var,
                                        new_output=new_var,
                                        flatten_residuals=False)
        new_equations += ((new_var, res_sympy),)
    return new_equations          

class SympyProjectable():
    def __init__(self, right, var=None):
        self.right = right
        self.var = var

    def residuals(self):
        res_sympy = partial_inversion(self.right, 
                                     old_output=self.var,
                                     flatten_residuals=False)
        R = encode_sympy(res_sympy)
        return R
    
    def functional(self):
        F = encode_sympy(self.right, self.var)
        return F

class ResidualProjectable():
    def __init__(self, R, solvevar_encoder):
        self.R = R
        self.solvevar_encoder = solvevar_encoder
        self.solveparkwargs = dict()

    def solvepar(self, **kwargs):
        self.solveparkwargs.update(kwargs)

    def required_solvevars(self):
        return self.solvevar_encoder.order

    def residuals(self):
        return self.R
    
    def functional(self, **kwargs):
        merged_kwargs = {**self.solveparkwargs, **kwargs}
        assert 'x_initial' in merged_kwargs
        RS = residual_solver(self.R, self.solvevar_encoder, **merged_kwargs)
        return RS
    
def build_residual(F, mergedin_vars, mergedin_shapes, local_coupled):
    def function(*args):
        local_dict = decode(args, mergedin_vars, mergedin_shapes)
        f_all_out = F.dict_in_dict_out(local_dict)
        f = encode(f_all_out, local_coupled, flatten=True)
        x = encode(local_dict, local_coupled, flatten=True)
        return x-f
    return function
    
class ExplicitProjectable():
    def __init__(self, F, left):
        self.F = F
        self.left = left

    def residuals(self):
        return build_residual(self.F, self.F.encoder.order, self.F.encoder.shapes, 
            (self.left,))
    
    def functional(self, **kwargs):
        return self.F

def coupled_vars_and_input(all_functions):
    non_coupled_vars = set()
    coupled_vars = ()
    invars = []
    for F in all_functions:
        var_shapes_in = zip(F.encoder.order, F.encoder.shapes)
        for vr, shp in var_shapes_in:
            if vr in non_coupled_vars:
                non_coupled_vars -= {vr}
                coupled_vars += ((vr,shp),)
            if (vr,shp) not in chain(coupled_vars,invars):
                invars.append((vr,shp))
        var_shapes_out = zip(F.decoder.order, F.decoder.shapes)
        for vr, shp in var_shapes_out:
            if (vr,shp) in invars and (vr,shp) not in coupled_vars:
                coupled_vars += ((vr,shp),)
                if (vr, shp) in invars:
                    invars.remove((vr,shp))
            elif vr not in non_coupled_vars:
                non_coupled_vars.add(vr)
    final_invars = tuple(elt for elt in invars if elt not in coupled_vars)
    return coupled_vars, final_invars


def merge_with_coupling(*projectables):
    all_functions = tuple(projectable.functional() 
                          for projectable in projectables)
    coupled_vars, final_invars = coupled_vars_and_input(all_functions)
    coupled_vars, coupled_shapes = zip(*coupled_vars)
    ins_vars, ins_shapes = zip(*final_invars)
    mergedin_vars = coupled_vars+ins_vars
    mergedin_shapes = coupled_shapes+ins_shapes

    residuals = ()
    for F in all_functions:
        local_coupled = tuple(vr for vr in F.decoder.order 
                              if vr in coupled_vars)
        function = build_residual(F, mergedin_vars, mergedin_shapes, local_coupled)
        res = EncodedFunction(function, Encoding(mergedin_vars, 
                                                mergedin_shapes))
        residuals += (res,)
    R = concatenate_residuals(residuals)
    return ResidualProjectable(R, Encoding(coupled_vars, 
                                          coupled_shapes))

class ProjectableIntersection():
    def __init__(self, *projectables, **kwargs):
        self.projectables = tuple(projectables)
        self.solveparkwargs = dict()

    def solvepar(self, **kwargs):
        self.solveparkwargs.update(kwargs)
    
    def functional_parts(self):
        return feedback(self.projectables)
    
    def required_solvevars(self):
        _, _, solvevars = self.functional_parts()
        return solvevars

    def residuals(self):
        all_residuals = tuple((projectable.residuals() 
                               for projectable in self.projectables))
        return concatenate_residuals(all_residuals)

    def functional(self, **kwargs):
        # default mode
        F, R, solvevars = self.functional_parts()
        combined = F
        if len(solvevars)>0:
            merged_kwargs = {**self.solveparkwargs, **kwargs}
            combined = strategy_eliminate_feedfwd(R, F, 
                                              solvevars, 
                                              **merged_kwargs)
        return combined

class ProjectableModel(ProjectableIntersection):
    def __init__(self):
        super().__init__()

    def add_equation(self, var, right):
        self.projectables += (SympyProjectable(right, var),)

    def Var(self, name, right):
        new_var = Var(name)
        if not isinstance(right, sp.Expr):
            new_var.varval = right
            encoded_function = EncodedFunction(lambda : (right,), None, 
                                               Encoding((new_var,)))
            self.projectables += (ExplicitProjectable(encoded_function, new_var),)
        else:
            self.add_equation(new_var, right)
        return new_var
    
    def VarRaw(self, name, right):
        new_var = Var(name)
        right.decoder = Encoding((new_var,))
        self.projectables += (ExplicitProjectable(right,new_var),)
        return new_var