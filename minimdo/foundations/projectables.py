from foundations.functionals import encode_sympy
from foundations.functional_noobj import concatenate_residuals
from foundations.functional_noobj import feed_forward
from foundations.functional_noobj import residual_solver
from foundations.functional_noobj import eliminate_vars
from modeling.arghandling import Encoder
from modeling.transformations import partial_inversion
from modeling.compute import Var
import numpy as np

def strategy_eliminate_feedfwd(residual, functional, solvevars, **kwargs):
    Relim = eliminate_vars(residual, functional)
    RS = residual_solver(Relim, Encoder(solvevars), 
                            **kwargs)
    combined = feed_forward((RS,functional))
    return combined

def feedback_residuals(sympy_equations, solvevars=None):
        all_functions = ()
        residuals = ()
        invars = ()
        solvevars = solvevars if solvevars is not None else tuple()
        for var, right in sympy_equations:
            F = encode_sympy(right, var)
            res = (F,) # default case, var=None
            if var is not None:
                # we only keep functions whose output 
                # is not an input to any other functions
                if len(set(F.decoder.order) & set(invars))==0:
                    invars += tuple(elt for elt in F.encoder.order 
                                if elt not in invars)
                    all_functions += (F,)
                    res = () # no residual
                else:
                    solvevars += tuple(elt for elt in F.decoder.order 
                                if elt not in solvevars)
                    res_sympy = partial_inversion(right, 
                                     old_output=var,
                                     flatten_residuals=False)
                    res = (encode_sympy(res_sympy),)
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

class Projectable():
    def __init__(self) -> None:
        self.sympy_equations = ()

    def add_equation(self, var, right):
        self.sympy_equations += ((var, right),)

    def Var(self, name, right):
        new_var = Var(name)
        self.add_equation(new_var, right)
        return new_var
    
    def residuals(self):
        all_residuals = ()
        for var, right in self.sympy_equations:
            res_sympy = partial_inversion(right, 
                                     old_output=var,
                                     flatten_residuals=False)
            all_residuals += (encode_sympy(res_sympy),)
        return concatenate_residuals(all_residuals)

    def functional(self, solvevars=None, **kwargs):
        functional, residual, solvevars = feedback_residuals(
            self.sympy_equations, solvevars)
        combined = functional
        if len(solvevars)>0:
            combined = strategy_eliminate_feedfwd(residual, 
                                              functional, 
                                              solvevars, 
                                              **kwargs)
        return combined
