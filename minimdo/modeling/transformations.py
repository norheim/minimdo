from graph.graphutils import all_component_nodes, edges_to_Ein_Eout, flat_graph_formulation
from modeling.execution import Component
import sympy as sp
import networkx as nx

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
    for comp in (comp for comp in components if comp.component in all_component_nodes(oldedges)): # could have also used newedges
        compid = comp.component
        old_out = var_from_mapping(arg_mapping, Eout, compid)
        new_out = var_from_mapping(arg_mapping, newEout, compid) 
        if old_out != new_out:
            new_function_expression = partial_inversion(comp.fxdisp, old_out, new_out, flatten_residuals=False) # flatten_residuals to false is required for unit conversion to work further down
            ignoretovar = False
            if new_out == None: # to still cary out the unit conversion
                ignoretovar = True 
                new_out = old_out
            newcomponent = Component.fromsympy(new_function_expression, new_out, ignoretovar=ignoretovar, component=compid)
            new_components.append(newcomponent)
    return new_components