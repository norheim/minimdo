from enum import Enum
from graph.graphutils import dfs_tree, merge_edges, solver_children, end_components, Node, SOLVER, VAR, COMP, all_solvers, path
from utils.utils import normalize_name
from collections import defaultdict

NodeTypesExtended = Enum('NodeTypesExtended', 'ENDCOMP')
NodeTypesExtended.__repr__ = lambda x: x.name
ENDCOMP = NodeTypesExtended.ENDCOMP
ExecutionTypes = Enum('ExecutionTypes', 'EXPL IMPL EQ NEQ OBJ SOLVE OPT')
ExecutionTypes.__repr__ = lambda x: x.name
EXPL, IMPL, EQ, NEQ, OBJ, SOLVE, OPT = ExecutionTypes.EXPL,ExecutionTypes.IMPL,ExecutionTypes.EQ,ExecutionTypes.NEQ,ExecutionTypes.OBJ,ExecutionTypes.SOLVE,ExecutionTypes.OPT

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

def default_solver_options(tree, solvers_options=None):
    solvers_options = dict(solvers_options) if solvers_options else dict()
    allsolvers = all_solvers(tree)
    Vtree = tree[2]
    for solver in allsolvers:
        if solver not in solvers_options:
            solvers_options[solver] = dict()
        if 'type' not in solvers_options[solver]:
            solvers_options[solver]['type']=SOLVE
        solvers_options[solver]['designvars'] = tuple(solver_children(Vtree, solver))
    return solvers_options

def order_from_tree(Ftree, Stree, Eout, includesolver=True, mergeendcomp=True):
    endcomps = {key: None in var for key,var in Eout.items()}
    visited_solvers = set()
    #allsolvers = all_solvers(Stree)
    sequence = []
    endcompqueue = defaultdict(list)
    endsolvers = []
    queue = list(Ftree.items())
    while queue:
        component, parent = queue.pop(0)
        ancestors = path(Stree, parent, visited_solvers)
        reverse_ancestors = ancestors[::-1]
        visited_solvers = visited_solvers.union(reverse_ancestors)
        if includesolver:
            sequence += [(SOLVER, solver, Stree.get(solver,None)) for solver in reverse_ancestors]
        if endcomps[component]:
            endcompqueue[parent].append(component)
        else:
            sequence += [(COMP, component, parent)]
        remainingcomps = len([elt for elt in solver_children(dict(queue), parent)])
        lastchildcomp = remainingcomps==0
        if lastchildcomp:
            endsolvers += [parent]
            remaining_child_solvers = [solver for solver in dfs_tree(Stree, parent) if solver not in visited_solvers]
            if not remaining_child_solvers:
                for solver in endsolvers:
                    if mergeendcomp and endcompqueue.get(solver,False):
                        sequence += [(ENDCOMP, endcompqueue[solver], solver)]
                    else:
                        sequence += [(ENDCOMP, endcomp, solver) for endcomp in endcompqueue[solver]]
                endsolvers = []
    return sequence

def mdao_workflow(sequence, solvers_options, comp_options=None, var_options=None):
    comp_options = comp_options if comp_options else dict()
    workflow = []
    for elt_type, content, parent in sequence:
        if elt_type == SOLVER:
            solver_options = solvers_options[content]
            solver_type = solver_options['type']
            other_solver_options = {key:val for key,val in solver_options.items() if key != 'type'}
            var_options_filtered = {key:var_options[key] for key in other_solver_options.get('designvars',[]) if key in var_options}
            wf_line = (solver_type, content, parent, other_solver_options, var_options_filtered)
            workflow.append(wf_line)
        elif elt_type == COMP:
            workflow.append((EXPL, content, parent))
        elif elt_type == ENDCOMP:
            if solvers_options[parent]['type'] == OPT:
                workflow += [(comp_options.get(comp, EQ), comp, parent) for comp in content]
            else:
                designvars = solvers_options[parent]['designvars']
                workflow.append((IMPL, content, designvars, parent))
    return workflow 
