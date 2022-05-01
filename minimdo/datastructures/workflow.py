from itertools import chain
from enum import Enum
from collections import OrderedDict
from graphutils import merge_edges, solver_children, end_components, Node, SOLVER, VAR, COMP
from utils import normalize_name

ExecutionTypes = Enum('ExecutionTypes', 'EXPL IMPL EQCST SOLVE OPT')
ExecutionTypes.__repr__ = lambda x: x.name
EXPL, IMPL, EQCST, SOLVE, OPT = ExecutionTypes.EXPL,ExecutionTypes.IMPL,ExecutionTypes.EQCST,ExecutionTypes.SOLVE,ExecutionTypes.OPT

def path(Stree, s, visited=None):
    visited = visited if visited else set()
    out = []
    if s in chain(Stree.values(),Stree.keys()):
        q = {s}  
    else:
        q = set()
        out = [s] if s not in visited else [] # we should at least return the parent node
    while q:
        s = q.pop()
        if s not in visited:
            out.append(s)
            if s in Stree:
                q.add(Stree[s])
        visited.add(s)
    return out

def get_f(f, edges):
    #TODO: ensure that elt.inputs is of same type as Ein[comp], eg. str
    f_lookup = {(frozenset(elt.inputs),elt.component,frozenset(elt.outputs)): elt for elt in f}
    Ein = merge_edges(edges[0], edges[2])
    Eout = edges[1]
    def lookup(comp):
        key = (frozenset(Ein[comp]),comp,frozenset(Eout[comp]))
        return f_lookup[key] 
    return lookup

def addexpcomp_args(lookup_f, parent_solver_node, key, debug=False):
    component = lookup_f(key)
    input_names = [str(Node(inputvar,VAR)) for inputvar in component.inputs]
    output_names = [str(Node(outputvar, VAR)) for outputvar in component.outputs]
    args = (EXPL, str(parent_solver_node), str(Node(key, COMP)), input_names, output_names, component.evaldict, component.graddict, debug)
    return args

def implicit_comp_name(comps):
    return "res_{}".format('_'.join(map(lambda x: normalize_name(str(x)),comps)))

def addimpcomp_args(lookup_f, parent_solver_node, fends, output_names):
    impl_components = []
    comps = []
    for idx, comp_node in enumerate(fends):
        component = lookup_f(comp_node)
        input_names = [str(Node(inputvar,VAR)) for inputvar in component.inputs]
        output_name = [str(Node(output_names[idx], VAR))] #TODO: THIS IS VERY HACKY BUT WILL WORK FOR NOW
        impl_components.append((input_names, output_name, component.evaldict, component.graddict, 1.))
        comps.append(Node(comp_node, COMP))
    args = (str(parent_solver_node), implicit_comp_name(comps), impl_components)
    return args

def generate_workflow(lookup_f, edges, tree, solvers_options=None, debug=False):
    solvers_options = solvers_options if solvers_options else dict()
    Fend = end_components(edges[1])
    Ftree, Stree, Vtree = tree
    visited = set()
    workflow = []
    Ftree_copy = OrderedDict(Ftree)
    flag = False
    while Ftree_copy:
        key, parentsolver = Ftree_copy.popitem(last=False)
        parent_solver_node = Node(parentsolver,SOLVER)
        out = path(Stree, parentsolver, visited)
        out_rev = out[::-1]
        visited = visited.union(out)
        f = [elt for elt in solver_children(Ftree_copy, parentsolver) if elt in Fend]
        for solver_idx in out_rev:
            parent_solver = Stree.get(solver_idx, None)
            parent_solver_name = str(Node(parent_solver,SOLVER)) if parent_solver else None
            solver_options = solvers_options.get(solver_idx, dict()) 
            workflow.append((SOLVE, parent_solver_name, str(Node(solver_idx, SOLVER)), solver_options))
        if key in Fend and not f: #the last one
            fends = [elt for elt in solver_children(Ftree, parentsolver) if elt in Fend]
            args = addimpcomp_args(lookup_f, parent_solver_node, fends, list(solver_children(Vtree, parentsolver)))
            if not flag:
                workflow.append((IMPL, *args))
            else:
                workflow.append((EQCST, *args)) #this would add them as design variable
        elif key not in Fend: 
            args = addexpcomp_args(lookup_f, parent_solver_node, key, debug)
            workflow.append(args)
    return workflow