from sympy import sympify
from itertools import chain
import torch
import re

def extract_number(string):
    return int(re.search(r'\d+', string).group())

def process_json(data):
    functional_sets = data["functional_sets"]
    objective = data['objective']
    
    objective_sympy = None
    polynomials = {}
    symb_mapping = {}
    symb_str_mapping = {}
    edges = (dict(),dict())
    
    functional_set_info = ((functional_set['residual'], functional_set['functionalvar']) 
                           for functional_set in functional_sets)

    for idx, functional_set in chain(enumerate(functional_set_info), [('objective', [objective, None])]) :
        function_str, output_var_str = functional_set
        function = sympify(function_str, locals=symb_str_mapping)
        if idx == 'objective':
            objective_sympy = function
        else:
            polynomials[idx] = function
        
        input_vars, output_var = tuple(), tuple()
        for symbol in function.free_symbols:
            if str(symbol) not in symb_str_mapping:
                symb_str_mapping[str(symbol)] = symbol
            if str(symbol) == output_var_str:
                symb_mapping[idx] = symb_str_mapping[str(symbol)]
                output_var = (symbol,)
            else:
                input_vars += (symbol,)

        if idx != 'objective':
            edges[0][idx] = input_vars
            edges[1][idx] = output_var

    indices = {elt: torch.tensor([int(i)]) for i, elt in 
               enumerate(sorted(symb_str_mapping.values(), key=lambda item: extract_number(str(item))))}

    return polynomials, indices, edges, objective_sympy