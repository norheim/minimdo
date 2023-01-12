from graph.graphutils import solver_children
from collections import OrderedDict, defaultdict
from graph.graphutils import COMP, SOLVER, all_components, Node, flat_graph_formulation, default_tree, upstream
from itertools import chain, islice
import networkx as nx

def filter_solver_comps(x):
    return x.nodetype in [COMP, SOLVER]

def invert_edges(Ein, Eout=None, newout=None):
    newout = newout if newout else {}
    all_comps = all_components(Ein)
    Ein_new = defaultdict(tuple)
    Eout_new = defaultdict(tuple)
    for comp in all_comps:
        outvar = newout.get(comp,None)
        Ein_new[comp] = tuple(elt for elt in chain(Ein[comp], Eout[comp] if Eout else []) if elt != outvar and elt != None)
        Eout_new[comp] = (outvar,)
    return dict(Ein_new), dict(Eout_new), {}

def eqv_to_edges_tree(Ein, output_set=None, n_eqs=None, offset=True):
    n_eqs = len(Ein) if n_eqs is None else n_eqs
    if offset:
        Ein = {key:tuple(vr-n_eqs for vr in var) for key,var in Ein.items()}
        if output_set:
            output_set = {key:outvar-n_eqs for key,outvar in output_set.items()}
    edges = invert_edges(Ein, newout=output_set)
    tree = default_tree(Ein.keys())
    return edges, tree, output_set

def sort_scc(G, filterfx=None):
    filterfx = filterfx if filterfx else filter_solver_comps
    C = nx.condensation(G)
    order = []
    for n in nx.topological_sort(C):
        filtereqs = {elt for elt in C.nodes[n]['members'] if filterfx(elt)}
        if filtereqs:
            order.append(filtereqs)
    return order

def copy_dicts(dicts):
    return tuple(dict(d) for d in dicts)

def standardize_comp(edges, Vtree, Ftree, comp, vrs=None):
    Ein, Eout, Rin = copy_dicts(edges)
    if vrs:
        updated_outs = tuple(elt for elt in Eout[comp] if elt not in vrs)
    else:
        vrs, updated_outs = zip(*((elt,None) for elt in Eout[comp])) 
    Eout[comp] = updated_outs
    Rin[comp] = vrs
    Vtree = dict(Vtree)
    parent = Ftree[comp]
    Vtree.update({vr:parent for vr in vrs})
    return (Ein, Eout, Rin), Vtree

# TODO: how to standardize with respect to variables?
def standardize_solver(tree, solver_idx):
    Ftree, Stree = copy_dicts(tree)
    parent_solver = Stree[solver_idx]
    comp_children = solver_children(Ftree, solver_idx)
    for comp in comp_children:
        Ftree[comp] = parent_solver
    subsolvers = solver_children(Stree, solver_idx)
    for subsolver in subsolvers:
        Stree[subsolver] = parent_solver
    Stree.pop(solver_idx)
    return Ftree, Stree

def merge_and_standardize(edges, tree, mergecomps, parentidx=1, newidx=2, mdf=True):
    Ftree, Stree, Vtree = tree
    Ftree = OrderedDict(Ftree)
    Stree = dict(Stree)
    if mdf:
        Stree[newidx]=parentidx
    else:
        newidx = parentidx
    for node in mergecomps:
        if node.nodetype == COMP:
            assert Ftree[node.name] == parentidx # can only merge nodes at the same level
            Ftree[node.name] = newidx
            edges, Vtree = standardize_comp(edges, Vtree, Ftree, 
                                            node.name, vrs=None)
        else:
            assert Stree[node.name] == parentidx
            # we remove the solver and raise all children one level up
            Ftree, Stree = standardize_solver(Ftree, Stree, node.name)
    return edges, (Ftree, Stree, Vtree)

def keep_original_order(Ftree, merge_order, reorder_only_scc=True):
    # Re-order (when possible) the merge_order to align with initial order
    original_comp_order = Ftree.keys()
    # This is n^2 should fix for efficiency in the future
    order_based_on_original = ([Node(comp, COMP) for comp in original_comp_order if Node(comp, COMP) in partition] for partition in merge_order)
    if not reorder_only_scc:
        # HACK: this definitly needs some fixing
        order_based_on_original = sorted(order_based_on_original, key=lambda x: min(idx for idx,comp in enumerate(Ftree[0].keys()) if Node(comp,COMP) in x))
    return order_based_on_original

def reorder_merge_solve(edges, tree, merge_order, solver_idx, mdf=True):
    tree = tuple((dict(d) if idx !=0 else OrderedDict(d)) for idx, d in enumerate(tree))
    nFtree = OrderedDict()
    for connected_components in merge_order:
        if len(connected_components) > 1:
            Stree = tree[1]
            new_idx = max(chain(Stree.keys(),(1,)))+1
            edges, tree = merge_and_standardize(edges, tree, connected_components, solver_idx, new_idx, mdf)
        Ftree = tree[0]
        for node in connected_components:
                if node.nodetype == COMP:
                    nFtree[node.name] = Ftree[node.name]
    return edges, (nFtree, tree[1], tree[2])

def reformulate(edges, tree, outset_initial=None, new_outset=None, not_outputs=None, root_solver_name='root', mdf=True, based_on_original2=False, solveforvars=True):
    if new_outset:
        edges_new = invert_edges(edges[0], edges[1], newout=new_outset)
        if solveforvars == 2: #HACK to use 2 for now
            solvevars = {vr: root_solver_name for vr in outset_initial.values() if vr not in new_outset.values()}
            #Alternative:
            # {outvar: root_solver_name for comp,outvar in outset_initial.items() if comp not in new_outset.keys()}
        elif solveforvars:
            reduced_comps = {comp for comp in outset_initial.keys() if comp not in new_outset}
            upstream_possibilites = set()
            for elt in reduced_comps:
                upstream_possibilites.update(upstream(edges_new, elt))
            upstream_possibilites.difference_update(new_outset.values())
            if not_outputs:
                upstream_possibilites.difference_update(not_outputs)
            solvevars = {invar: root_solver_name for invar in islice(upstream_possibilites, len(reduced_comps))}
        else:
            solvevars = {}
        tree_new = tree[0], {}, solvevars
    else:
        edges_new = edges
        tree_new = tree
    G = flat_graph_formulation(*edges_new)
    order = sort_scc(G)
    order_based_on_original = keep_original_order(tree[0], order, not based_on_original2)
    edges_tear_ordered, tree_tear_ordered = reorder_merge_solve(edges_new, tree_new, order_based_on_original, root_solver_name, mdf=mdf)
    return edges_tear_ordered, tree_tear_ordered