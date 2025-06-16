from graph.graphutils import all_component_nodes, edges_to_Ein_Eout, flat_graph_formulation
from src.v2.execution import Component
import sympy as sp
import networkx as nx
import numpy as np
import jax.numpy as anp

def partial_inversion(old_expression, old_output=None, new_output=None, flatten_residuals=True):
    # old_expression needs to be a sympy expr
    # old_output can be None or any sympy variable
    # new_output can be None or any sympy variable part of old_expression and old_output
    if old_output == new_output:
        return old_expression
    diff = old_expression if old_output == None else old_output-old_expression 
    if new_output:
        if diff.atoms(sp.Abs): #temporary HACK to ensure positive square root
            expr2, rep = sp.posify(diff)
            rep_rev = {var:key for key,var in rep.items()}
            new_expression = sp.solve(expr2, rep_rev.get(new_output,new_output))
            new_expression = [expr.subs(rep) for expr in new_expression]
        else:
            new_expression = sp.solve(diff, new_output)
        sol_idx = 1 if len(new_expression) == 2 else 0
        new_expression = new_expression[sol_idx]
        return sp.expand(new_expression)
    else:
        if flatten_residuals:
            num, _ = sp.fraction(sp.together(diff))
            return num
        else:
            return diff

def flatten_output(scalar_or_array):
    return (np.array(scalar_or_array).flatten() 
            if isinstance(scalar_or_array, (np.ndarray, list)) 
            else np.array([scalar_or_array]))
    
# TODO: this now lives in transformations
def newfx(c, *x):
    fxval = c.function(*x[:sum(c.indims)])
    outval = x[sum(c.indims):]
    out = [outval[idx]-elt for idx,elt in enumerate(fxval)]
    # IMPORTANT: flatten the output!
    return [elt for vector in out for elt in vector]

def residual_component(c, idx=0):
    newinputs = c.inputs + c.outputs
    newindims = c.indims + c.outdims
    fx = lambda *x: newfx(c, *x)
    return Component(fx, newinputs, (None,), idx, newindims, c.outdims)

def generate_components_and_residuals(components, edges):
    rcomps = [residual_component(c, c.id) for c in components if c.id in edges[2].keys()]
    return components+rcomps

def flatten_component(comp, newid=None):
    new_inputs = comp.inputs + comp.outputs
    get_inputs = lambda args: args[:len(comp.inputs)]
    get_outputs = lambda args: args[len(comp.inputs):]
    new_function = lambda *args: [-anp.hstack(
        comp.function(*get_inputs(args)))+anp.hstack(get_outputs(args))]
    new_indims = comp.indims + comp.outdims
    new_outdims = (sum(sum(outdim) if isinstance(outdim, tuple) 
                      else outdim for outdim in comp.outdims),)
    new_fxdisp = '{}-({})'.format(comp.outputs[0] if len(comp.outputs)==1 
                                else str(list(comp.outputs)), 
                                comp.fxdisp) if comp.fxdisp is not None else None
    if newid is None:
        newid = comp.id
    new_comp = Component(new_function, new_inputs, (None,), newid, 
                         new_indims, new_outdims, new_fxdisp, comp.arg_mapping)
    return new_comp

def var_from_mapping(arg_mapping, Eout, comp):
    if Eout[comp][0] is not None:
        return arg_mapping[Eout[comp][0]]
    else:
        return None

def transform_components(oldedges, newedges, components, arg_mapping):
    _, Eout = edges_to_Ein_Eout(oldedges)
    # Test that graphs are the same
    old_edges = flat_graph_formulation(*oldedges).to_undirected()
    new_edges = flat_graph_formulation(*newedges).to_undirected()
    graph_isomorphic = {frozenset(elt) for elt in new_edges.edges()}=={frozenset(elt) for elt in old_edges.edges()}
    assert graph_isomorphic
    # This is just to make sure that the new edges have an isomorphic undirected graph, which should be the case
    _, newEout = edges_to_Ein_Eout(newedges)
    new_components = []
    for comp in (comp for comp in components if comp.id in all_component_nodes(oldedges)): # could have also used newedges
        compid = comp.id
        old_out = var_from_mapping(arg_mapping, Eout, compid)
        new_out = var_from_mapping(arg_mapping, newEout, compid) 
        if old_out != new_out:
            new_function_expression = partial_inversion(comp.fxdisp, old_out, new_out, flatten_residuals=False) # flatten_residuals to false is required for unit conversion to work further down
            ignoretovar = False
            if new_out == None: # to still cary out the unit conversion
                ignoretovar = True 
                new_out = old_out
            newcomponent = Component.fromsympy(new_function_expression, new_out, ignoretovar=ignoretovar, component=compid, arg_mapping=True)
            new_components.append(newcomponent)
    return new_components