from datastructures.graphutils import solver_children
from collections import OrderedDict
from datastructures.graphutils import COMP, SOLVER
from itertools import chain
import networkx as nx

def filter_solver_comps(x):
    return x.nodetype in [COMP, SOLVER]

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