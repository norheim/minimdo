from modeling.arghandling import EncodedFunction, EncodedFunctionContainer
from modeling.arghandling import Encoder
from modeling.arghandling import decode, encode, merge_encoders
from modeling.arghandling import flatten_args
from modeling.execution import sympy_fx_inputs
from modeling.transformations import partial_inversion
from modeling.compute import Var
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from functools import lru_cache, wraps
import numpy as np

def np_cache(function):
    @lru_cache()
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        return cached_wrapper(tuple(array))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper

def feed_forward_solver(function_order, functional):
    def solver_function(*args):
        local_dict = decode(args, functional.encoder.order)
        final_output_dict = {}
        for f in function_order:
            output_dict = f.dict_in_dict_out(local_dict)
            final_output_dict.update(output_dict)
            local_dict.update(output_dict)
        out = encode(final_output_dict, functional.decoder.order)
        return out
    return solver_function

def flat_residuals_executable(function_order, functional):
    def residual_function(*args):
        local_dict = decode(args, functional.encoder.order)
        residuals_list = [f.dict_in_flat_out(local_dict) for f in function_order]
        flattened_residuals = flatten_args(residuals_list)
        return flattened_residuals
    return residual_function

def residual_solver(functional):
    projectable = functional.projectable
    def solver_function(*args, x_initial=None, random_generator=None):
        x_initial = dict() if x_initial is None else x_initial
        fixed_vars_dict = decode(args, functional.encoder.order)
        def local_residual(x):
            x_dict = decode(x, functional.decoder.order, unflatten=True)
            x_dict.update(fixed_vars_dict)
            return projectable.dict_in_flat_out(x_dict)
        x0 = encode(x_initial, functional.decoder.order,
                     missingarg=random_generator, flatten=True)
        x_root = fsolve(local_residual, x0)
        return x_root

    return solver_function

class Projectable(EncodedFunctionContainer):
    def __init__(self, encoded_functions=None):
        super().__init__(encoded_functions)
        self.f = flat_residuals_executable(self.encoded_functions, 
                                           self)

def encode_sympy(sympexpr, named_output=None, single_output=False):
    fx_multiple_outputs, inputs, _ = sympy_fx_inputs(sympexpr)
    if not single_output:
        fx = lambda *args: (fx_multiple_outputs(*args),)
    else:
        fx = fx_multiple_outputs
    output_encoder = None
    if named_output is not None:
        output_encoder = Encoder((named_output,))
    encoded_function = EncodedFunction(fx, Encoder(inputs), output_encoder)
    return encoded_function

class Functional(EncodedFunctionContainer):
    def __init__(self, var=None, right=None, **kwargs):
        super().__init__(**kwargs)
        self.projectable = Projectable()
        self.f = feed_forward_solver(self.encoded_functions, 
                                     self)
        if var is not None:
            self.add_var(var, right)

    def add_var(self, var, right):
        res_sympy = partial_inversion(right, 
                                     old_output=var,
                                     flatten_residuals=False)
        f = encode_sympy(right, var)
        r = encode_sympy(res_sympy) 
        self.add_encoded_functions(f)
        self.projectable.add_encoded_functions(r)

    def Var(self, name, right):
        new_var = Var(name)
        self.add_var(new_var, right)
        return new_var

def generate_opt_objects(problem, obj, ineq, eq, eliminate):
    def optimizer_function(*args, x_initial=None, random_generator=None):
        parameters_dict = decode(args, problem.encoder.order)
        #@np_cache
        def eval_obj_constraints(x):
            local_dict = decode(x, problem.decoder.order, unflatten=True)
            local_dict.update(parameters_dict)
            inter_dict = eliminate.dict_in_dict_out(
                local_dict)
            local_dict.update(inter_dict)
            objval = float(obj.dict_in_only(local_dict))
            ineqval = ineq.dict_in_flat_out(local_dict)
            eqval = eq.dict_in_flat_out(local_dict)
            return objval, ineqval, eqval
        x0 = encode(x_initial, problem.decoder.order, flatten=True,
                missingarg=random_generator)
        return eval_obj_constraints, x0
    return optimizer_function

def optimizer_solver(problem, obj, ineq, eq, eliminate, bounds):
    obj_generator = generate_opt_objects(problem, obj, ineq, eq, eliminate)
    def optimizer_function(*args, x_initial=None, random_generator=None):
        eval_obj_constraints, x0 = obj_generator(*args, 
                                                 x_initial=x_initial, 
                                                 random_generator=random_generator)
        ineq_constraints, eq_constraints = tuple(), tuple()
        if len(ineq.encoded_functions)>= 1:
            ineq_function = lambda x: eval_obj_constraints(x)[1][:] #shallow copy
            ineq_constraints = (NonlinearConstraint(ineq_function, 
                                                  -np.inf, 0),)
        if len(eq.encoded_functions)>= 1:
            eq_function = lambda x: eval_obj_constraints(x)[2][:] #shallow copy
            eq_constraints = (NonlinearConstraint(eq_function, 0, 0),)
        constraints = eq_constraints+ineq_constraints
        obj_function = lambda x: eval_obj_constraints(x)[0]
        solution_obj = minimize(obj_function, x0, 
                          constraints=constraints, bounds=bounds)
        return solution_obj.x
    return optimizer_function

class Problem(EncodedFunction):
    def __init__(self, obj, ineqs=None, eqs=None, 
                 eliminate=None, bounds=None, parameters=None):
        super().__init__(None)
        ineqs = ineqs if ineqs is not None else tuple()
        eqs = eqs if eqs is not None else tuple()
        self.obj = encode_sympy(obj, single_output=True)
        encoded_ineq_functions = [encode_sympy(ineq) for ineq in ineqs]
        self.ineqs = Projectable(encoded_ineq_functions)
        encoded_eq_functions = [encode_sympy(eq) for eq in eqs]
        self.eqs = Projectable(encoded_eq_functions)
        self.eliminate = eliminate
        self.decoder = merge_encoders(self.obj.encoder, self.ineqs.encoder, 
                                      self.eqs.encoder, self.eliminate.encoder,
                                      exclude_encoder=self.eliminate.decoder)
        self.encoder = Encoder()
        bounds = dict() if bounds is None else bounds
        self.bounds = encode(bounds, self.decoder.order, missingarg=lambda :(None,None)) 
        self.f = optimizer_solver(self, self.obj, self.ineqs, 
                                  self.eqs, self.eliminate, self.bounds)

def intersection(*functionals):
    random_generator = np.random.default_rng(seed=2023).random
    F = Functional(random_generator=random_generator)
    F.add_encoded_functions(*functionals)
    merged_residuals = sum([F.projectable.encoded_functions 
                            for F in functionals], [])
    new_projectable = Projectable(merged_residuals)
    F.projectable = new_projectable
    F.f = residual_solver(F)
    return F