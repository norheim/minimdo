from inputresolver import reassigneq
from datastructures.execution import Component
import sympy as sp
from collections import OrderedDict
from datastructures.graphutils import all_variables, Node, VAR
from datastructures.operators import eqv_to_edges_tree
from compute import Var
from randompoly import random_bijective_polynomial
from testproblems import generate_random_prob
import numpy as np

def get_arg_mapping(var_mapping, symbol_map=False):
    return {key:symbol if symbol_map else name for key, (symbol, name) in var_mapping.items()}

def directed_poly_executables(var_mapping, polynomials, output_set):
    arg_mapping = get_arg_mapping(var_mapping)
    new_components = []
    for idx, eq in polynomials.items():
        left = output_set[idx]
        leftvar,_ = var_mapping[left]
        function = sp.simplify(reassigneq(None, eq, leftvar))
        new_comp = Component.fromsympy(function, leftvar, component=idx, arg_mapping=arg_mapping)
        new_components.append(new_comp)
    return new_components

def residual_poly_executables(var_mapping, polynomials):
    arg_mapping = get_arg_mapping(var_mapping)
    return [Component.fromsympy(function, component=component, arg_mapping=arg_mapping) for component,function in polynomials.items()]


def generate_random_polynomials(eqv, output_set, n_eqs, rng=None):
    rng = rng if rng else np.random.default_rng(12345)
    edges, tree, output_set = eqv_to_edges_tree(eqv, output_set, n_eqs, offset=True)
    Ein, Eout, _ = edges
    all_vars = all_variables(Ein,Eout)
    var_mapping = {idx: (Var(str(Node(idx,VAR)), varid=idx), str(Node(idx,VAR))) for idx in all_vars}
    polynomials = {idx: random_bijective_polynomial(rng, (var_mapping[elt-n_eqs][0] for elt in vrs)) for idx,vrs in eqv.items()}
    components = residual_poly_executables(var_mapping, polynomials)
    directed_components = directed_poly_executables(var_mapping, polynomials, output_set)
    return polynomials, var_mapping, edges, tree, components+directed_components