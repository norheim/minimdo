from torch.autograd.functional import jacobian, hessian
from sympy import sympify
import sympy as sp
import numpy as np
import torch
import json
import os
import re

def generate_indices(keys):
    indices = {k: torch.tensor([idx], dtype=torch.long) for idx,k in enumerate(keys)}
    return indices

def reverse_indices(idxvectors, indices):
    revindex = {val.item():key for key,val in indices.items()}
    return tuple(revindex[elt.item()] for elt in idxvectors)

def symbols(symbolargs, dim='scalar'):
    all_symbols = sp.symbols(symbolargs)
    indices = generate_indices(all_symbols)
    return all_symbols + (indices,)

def load_file(file_name, path_to_file='../applications/data/'):
    file_path = f'{path_to_file}{file_name}.json'
    os.path.isfile(file_path)
    with open(file_path, 'r') as file:
        json_str = file.read()
    data = json.loads(json_str)
    return data

def extract_number(string):
    return int(re.search(r'\d+', string).group())

def process_expression(exprstr, symb_str_mapping):
    exprsympy = sympify(exprstr, locals=symb_str_mapping)
    # free_symbols returns a set which is not ordered
    sorted_symbols = sorted(exprsympy.free_symbols, key=lambda s: s.name)
    for symbol in sorted_symbols:
        if str(symbol) not in symb_str_mapping:
            symb_str_mapping[str(symbol)] = symbol
    return exprsympy

def process_json(functional_sets, symb_str_mapping=None):
    symb_str_mapping = symb_str_mapping if symb_str_mapping else {}
    analysismap = []
    
    for functional_set in functional_sets:
        analysis_str = functional_set.get('analysis',None)
        residual_str = functional_set.get('residual',None)
        output_var_str= functional_set.get('functionalvar',None)
        analysis = process_expression(analysis_str, symb_str_mapping) if analysis_str is not None else None
        residual = process_expression(residual_str, symb_str_mapping) if residual_str is not None else None
        if output_var_str not in symb_str_mapping:
            outputvar = sp.Symbol(output_var_str)
            symb_str_mapping[output_var_str] = outputvar
        else:
            outputvar = symb_str_mapping[output_var_str]
        analysismap.append((analysis, outputvar, residual))
    
    return analysismap, symb_str_mapping

def load_multiple_files(prob_names, file_name=None):
    # if file_name is none, then prob_names are interpreted as file names
    all_analyses = {}
    symb_str_mapping = {}
    equality_constraints_sympy = []
    inequality_constraints_sympy = []
    if file_name is not None:
        all_data = load_file(file_name)
    for prob_name in prob_names:
        data = all_data[prob_name] if file_name is not None else load_file(file_name)
        equality_constraints_sympy += [
            process_expression(elt, symb_str_mapping) 
            for elt in data.get('equality_constraints',[])]
        inequality_constraints_sympy += [
            process_expression(elt, symb_str_mapping) 
            for elt in data.get('inequality_constraints',[])]
        objective = data.get('objective',None)
        if objective is not None:
            objective = process_expression(objective, symb_str_mapping)
        functional_sets = data.get('functional_sets',[])
        analysismap, symb_str_mapping = process_json(
            functional_sets, symb_str_mapping)
        all_analyses[prob_name] = analysismap
    return (all_analyses, objective, equality_constraints_sympy, 
            inequality_constraints_sympy, symb_str_mapping)


def load_vals(file_name, indices, path_to_file=None, x0=None, default=0, isdict=False, dtype=torch.float64):
    xvalsdict = load_file(file_name) if not isdict else file_name
    x0 = x0 if x0 is not None else torch.ones(sum(len(idx) for idx in indices.values()), dtype=dtype)*default
    for key, val in indices.items():
        xval = xvalsdict.get(str(key), x0[val])
        if len(val) > 1:
            xval = xval.to(dtype)
        x0[val] = xval
    return x0

def perturb(x0, delta, indices, seed=42):
    n = len(x0)
    rng = np.random.default_rng(seed=seed)
    points = rng.normal(size=(n,))
    points *= delta
    points += 1
    xnew = x0.clone()
    xnew.requires_grad_(False)
    for idx in indices:
        xnew[idx] = x0[idx]* points[idx]
    xnew.requires_grad_(True);
    return xnew

# precursor to optim_func
def transfer_value(xfrom, xto, copy_indices_tuple):
    for indices_in,indices_out  in copy_indices_tuple:
        xto[indices_out] = xfrom[indices_in]
    return xto

class ExpandVector():
    def __init__(self, indicesin, indicesout):
        self.indicesout_len =  len(indicesout.values())
        copy_over_indices = indicesin.keys()
        self.copy_indices_tuple = [(indicesin[key], indicesout[key]) for key in copy_over_indices]
        structure = torch.cat([indicesin[key] for key in copy_over_indices])
        self.structure = (structure, structure)
        structure_by_name = {key: key for key in indicesout.keys() if key in copy_over_indices}
        self.structure_full = {elt.item():indicesout[val] for key,val in structure_by_name.items() for elt in indicesout[key]}

    def __call__(self, x):
        xzero = torch.empty(self.indicesout_len, dtype=x.dtype)
        output = transfer_value(x, xzero, self.copy_indices_tuple)
        return output

def generate_optim_functions(optim_funcs, solvefor_indices, x, 
                             inequality_direction='negative-null',
                             objective=None, 
                             residuals=None, 
                             equalities=None,
                             inequalities=None):
    def eval_adaptive(x):
        fvals = optim_funcs(x)
        objval = fvals[objective] if objective is not None else torch.tensor(0)
        res = fvals[residuals] if residuals is not None else torch.tensor([])
        eqval = fvals[equalities] if equalities is not None else torch.tensor([])
        alleqval = torch.cat((eqval, res)) 
        ineqval = fvals[inequalities] if inequalities is not None else torch.tensor([])
        if inequality_direction == 'negative-null':
            return objval, ineqval, alleqval
        elif inequality_direction == 'positive-null':
            return objval, -ineqval, alleqval

    def eval_all(y):
        x.requires_grad_(False)
        x[solvefor_indices] = torch.from_numpy(y).to(x.dtype)
        objval, ineqval, eqval = eval_adaptive(x)
        x.requires_grad_(True)
        return objval.item(), ineqval, eqval
    
    def obj_function(y):
        objval, _, _ = eval_all(y)
        return objval

    def ineq_function(y=None):
        _, ineqval, _ = eval_all(y)
        return ineqval
    
    def eq_function(y=None):
        _, _, eqval = eval_all(y)
        return eqval
    
    def dobj(y):
        xc = x.detach() # remove gradient tracking
        xc[solvefor_indices] = torch.from_numpy(y).to(x.dtype)
        dobj = jacobian(lambda w: eval_adaptive(w)[0], xc).numpy()[0]
        return dobj[solvefor_indices]
    
    def hobj(y):
        xc = x.detach() # remove gradient tracking
        xc[solvefor_indices] = torch.from_numpy(y).to(x.dtype)
        hobj = hessian(lambda w: eval_adaptive(w)[0], xc).numpy()
        return hobj[np.ix_(solvefor_indices,solvefor_indices)]

    def dineq(y=None):
        xc = x.detach()
        xc[solvefor_indices] = torch.from_numpy(y).to(x.dtype)
        dineq = jacobian(lambda w: eval_adaptive(w)[1], xc).numpy()
        return np.take(dineq, solvefor_indices, axis=1) # replaces dineq[:,solvefor_indices], which is not consistent when solvefor_indices only has one item vs multiple items
    
    def deq(y=None):
        xc = x.detach()
        xc[solvefor_indices] = torch.from_numpy(y).to(x.dtype)
        deq = jacobian(lambda w: torch.cat(eval_adaptive(w)[2:]), xc).numpy()
        return np.take(deq, solvefor_indices, axis=1)

    xguess = x[solvefor_indices].detach().numpy()
    
    return xguess, obj_function, ineq_function, eq_function, dobj, dineq, deq, hobj

from itertools import zip_longest

def fmt(x):
    if np.isclose(x, 0.0):
        return '0'
    if abs(x) < 0.01:
        return f"{x:.2e}".replace('+0', '')
    elif abs(x) > 10000:
        return f"{x:.2e}".replace('+0', '')
    else:
        return f"{x:.3f}".rstrip('0').rstrip('.')

def print_formatted_table(xsol, indices, idxrev, subset=None):
    subset = subset if subset is not None else torch.tensor(list(indices.values()))
    header = [str(idxrev[k.item()]) for k in subset]
    xdisp = [x[subset] for x in xsol]
    matrix = [header, *map(lambda row: [fmt(num) for num in row], xdisp)]
    col_widths = [max(map(len, col)) for col in zip(*matrix)]

    print(' '.join(f"{name:<{width}}" for name, width in zip(header, col_widths)))
    for row in matrix[1:]:
        print(' '.join(f"{entry:<{width}}" for entry, width in zip(row, col_widths)))