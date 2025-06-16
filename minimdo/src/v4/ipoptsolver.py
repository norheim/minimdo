import jax
jax.config.update('jax_platform_name', 'cpu')
import numpy as np
import jax.numpy as anp
from jax import jacobian


def unflatten(x0, order):
    unflattened = {}
    idx = 0
    for var in order:
        size = np.prod(var.shape)
        unflattened[var.name] = anp.reshape(x0[idx:idx + size], var.shape)
        idx += size
    return unflattened

def compute_residuals_generic(funcs_with_io, order):
    def inner(x0):
        variables = unflatten(x0, order)
        residuals = []

        for idx, (func, inputs, outputs) in enumerate(funcs_with_io):
            input_vars = tuple(variables[input_var.name] for input_var in inputs)
            output_vars = tuple(variables[output_var.name] if output_var else None for output_var in outputs)
            func_output = func(*input_vars)
            for output_var, func_val in zip(output_vars, func_output):
                residual = func_val
                if output_var is not None:
                    residual = output_var - func_val
                else:
                    residual = (func_val,)
                residuals.extend(residual)

        return anp.array(residuals)

    return inner

def compute_structure(funcs_with_io, order):
    structure = []
    for func, inputs, outputs in funcs_with_io:
        for output_var in outputs:
            out_shape = output_var.shape if output_var != None else 1
            row = tuple(np.ones((np.prod(out_shape), np.prod(var.shape))) if var in inputs else np.zeros((np.prod(out_shape), np.prod(var.shape))) for var in order)
            structure.append(np.hstack(row))
    structure = np.vstack(structure)
    return structure

class ProblemIPOPT:
    def __init__(self, n, constraints, constraints_jacobian, constraints_jacobian_structure):
        self.n = n
        self.constraints = constraints
        self.jacobian = constraints_jacobian

    def objective(self, x):
        return 0
    def gradient(self, x):
        return np.zeros(self.n)

    def constraints(self, x):
        return self.constraints(x)
    
    def jacobian(self, x):
        return self.jacobian(x)

def get_index_ranges(variables):
    index_ranges = []
    start_idx = 0
    for var in variables:
        size = np.prod(var.shape)
        end_idx = start_idx + size
        index_ranges.append((start_idx, end_idx))
        start_idx = end_idx
    return index_ranges

def subset_index_ranges(all_variables, selected_subset):
    all_index_ranges = get_index_ranges(all_variables)
    subset_index_ranges = [all_index_ranges[all_variables.index(var)] for var in selected_subset]
    return subset_index_ranges

def select_subset(flat_vector, subset_index_ranges):
    return anp.hstack([flat_vector[start_idx:end_idx] for start_idx, end_idx in subset_index_ranges])

def get_precomputed_info(variables):
    precomputed = []
    start_idx = 0
    for var in variables:
        size = np.prod(var.shape)
        end_idx = start_idx + size
        precomputed.append((start_idx, end_idx, var.shape))
        start_idx = end_idx
    return precomputed

def get_precomputed_indices(all_variables, selected_subset):
    all_info = get_precomputed_info(all_variables)
    selected_indices = []
    for var, (start_idx, end_idx, shape) in zip(all_variables, all_info):
        if var in selected_subset:
            selected_indices.append((start_idx, end_idx, shape))
    return selected_indices

def setup_ipopt(components, variable_order, y):
    funcs_with_io = [(c.function, c.mapped_inputs, c.mapped_outputs) for c in components]
    independent = list({var for comp in components for var in comp.mapped_outputs})
    inputs = list({var for comp in components for var in comp.mapped_inputs+comp.mapped_outputs})
    residuals_full = compute_residuals_generic(funcs_with_io, inputs)
    ycon = select_subset(y, subset_index_ranges(variable_order, inputs))
    yaddress = get_precomputed_indices(inputs, independent)
    xaddress = get_precomputed_indices(independent, independent)
    def constraints_plug_in_projected(x):
        yx = anp.copy(ycon)
        for (xstart,xend,_),(ystart,yend,_) in zip(xaddress,yaddress):
            yx = jax.lax.dynamic_update_slice(yx, x[xstart:xend], (ystart,) )
        return residuals_full(yx)
    constraints_jacobian = jacobian(constraints_plug_in_projected)
    constraints_jacobian_structure = compute_structure(funcs_with_io, independent)
    _,n = constraints_jacobian_structure.shape
    return ProblemIPOPT(n, constraints_plug_in_projected, constraints_jacobian, constraints_jacobian_structure), independent