from modeling.arghandling import encoder_diff, decode, encode
from modeling.arghandling import flatten_args, merge_encoders
from modeling.arghandling import EncodedFunction, Encoding
from scipy.optimize import fsolve
from scipy.optimize import minimize, NonlinearConstraint
import numpy as np

def sequenced_encode_decode(encoded_function_order):
    encoder = Encoding()
    decoder = Encoding()
    for elt in encoded_function_order:
        encoder = merge_encoders(encoder,
                        elt.encoder, 
                        exclude_encoder=decoder)
        decoder = merge_encoders(decoder, elt.decoder)

    return encoder, decoder

def feed_forward(encoded_function_order):
    encoder, decoder = sequenced_encode_decode(encoded_function_order)
    def function(*args):
        local_dict = decode(args, encoder.order, encoder.shapes)
        final_output_dict = {}
        for f in encoded_function_order:
            output_dict = f.dict_in_dict_out(local_dict)
            final_output_dict.update(output_dict)
            local_dict.update(output_dict)
        output = encode(final_output_dict, decoder.order)
        return output
    return EncodedFunction(function, encoder, decoder)

def concatenate_residuals(encoded_function_order):
    encoder = merge_encoders(*[elt.encoder for elt in encoded_function_order])
    def function(*args):
        local_dict = decode(args, encoder.order, encoder.shapes)
        residuals_list = [f.dict_in_flat_out(local_dict) 
                          for f in encoded_function_order]
        flattened_residuals = flatten_args(residuals_list)
        return flattened_residuals
    return EncodedFunction(function, encoder)

def eliminate_vars(encoded_function, elimination_functional=None):
    encoder = merge_encoders(encoded_function.encoder,
                             elimination_functional.encoder, 
                             exclude_encoder=elimination_functional.decoder)
    decoder = None
    if encoded_function.decoder.order:
        decoder = encoder_diff(encoded_function.decoder, 
                            elimination_functional.decoder)
    def function(*args):
        local_dict = decode(args, encoder.order, encoder.shapes)
        if elimination_functional is not None:
            inter_dict = elimination_functional.dict_in_dict_out(
                local_dict)
            local_dict.update(inter_dict)
        output = encoded_function.dict_in_flat_out(local_dict)
        return output
    return EncodedFunction(function, encoder, decoder)

def partial_function(encoded_function, fixed_vars_encoder, solve_vars_encoder):
    def fixed_function(*fixed_args):
        def function(*args):
            fixed_vars_dict = decode(fixed_args, fixed_vars_encoder.order, 
                                     fixed_vars_encoder.shapes)
            x_dict = decode(args, solve_vars_encoder.order, 
                            solve_vars_encoder.shapes)
            x_dict.update(fixed_vars_dict)
            return encoded_function.dict_in_flat_out(x_dict)
        return EncodedFunction(function, solve_vars_encoder, 
                               encoded_function.decoder)
    return EncodedFunction(fixed_function, fixed_vars_encoder)

def residual_computer(encoded_function, solve_vars_encoder):
    fixed_input_encoder = encoder_diff(encoded_function.encoder, solve_vars_encoder)
    return partial_function(encoded_function, fixed_input_encoder, solve_vars_encoder)

def residual_solver_determined(residual_function):
    decoder = residual_function.encoder
    def function(x_initial=None, random_generator=None):
        x_initial = dict() if x_initial is None else x_initial
        x0 = encode(x_initial, decoder.order,
                    missingarg=random_generator, flatten=True)
        x_root = fsolve(residual_function.flat_in_only, x0)
        return x_root
    return EncodedFunction(function, None, decoder)

def residual_solver(encoded_function, solve_vars_encoder, 
                    x_initial=None, random_generator=None):
    RC = residual_computer(encoded_function, solve_vars_encoder)
    def function(*args):
        PS = RC.f(*args)
        RS = residual_solver_determined(PS)
        x_solved = RS.f(x_initial=x_initial, 
             random_generator=random_generator)
        #x_initial = x_solved # store for next iteration
        return solve_vars_encoder.unflatten(x_solved)
    return EncodedFunction(function, RC.encoder, solve_vars_encoder)

def external_encoder(f, decoder):
    def function(x):
        d = decode(x, decoder.order, decoder.shapes, unflatten=True)
        return f.dict_in_flat_out(d)
    return function

def optimizer_solver(obj, ineqs=None, eqs=None, bounds=None):
    eqs = eqs if eqs is not None else tuple()
    ineqs = ineqs if ineqs is not None else tuple()
    decoder = merge_encoders(obj.encoder,
                             *(con.encoder for con in ineqs+eqs))
    obj_flattened = external_encoder(obj, decoder)
    ineq_con = tuple(NonlinearConstraint(
        external_encoder(ineq, decoder), -np.inf, 0)
                 for ineq in ineqs) 
    eq_con = tuple(NonlinearConstraint(
        external_encoder(eq, decoder), 0, 0)
                 for eq in eqs) 
    constraints = ineq_con + eq_con
    bounds_tuples = encode(bounds, decoder.order, 
                           missingarg=lambda :(None,None)) 
    def function(x0=None, random_generator=None, 
                 optimizer='SLSQP'):
        x0 = dict() if x0 is None else x0
        x0_array = encode(x0, decoder.order,
                    missingarg=random_generator, flatten=True)
        solution_obj = minimize(obj_flattened, x0_array, 
                          constraints=constraints, 
                          bounds=bounds_tuples,
                          method=optimizer)
        print(solution_obj.message)
        return solution_obj.x
    return EncodedFunction(function, None, decoder)