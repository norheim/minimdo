from collections import defaultdict
from graphutils import flat_graph_formulation, solver_children, Node, COMP, SOLVER, VAR
from mergegraph import merge_graph

def level_order_tree(tree, root=1):
    level_order = [root]
    next_dict = defaultdict(list)
    for key,val in tree.items():
        next_dict[val].append(key) 
    q = [root]
    while q:
        idx = q.pop()
        elts = next_dict[idx]
        level_order+=elts
        q+=elts
    return level_order
    
def typed_solver_children(tree, solver_idx, node_type, nodetyperepr):
    return {Node(comp, node_type, nodetyperepr) for comp in solver_children(tree, solver_idx)}

def build_typedgraph(edges, tree, nodetyperepr):
    Ftree, Stree, Vtree = tree
    graphs = dict()
    G_parent = flat_graph_formulation(*edges, nodetyperepr=nodetyperepr)
    merge_order = level_order_tree(Stree)[::-1]
    for solver_idx in merge_order:
        solve_vars = typed_solver_children(Vtree, solver_idx, VAR, nodetyperepr)
        component_nodes = typed_solver_children(Ftree, solver_idx, COMP, nodetyperepr)
        solver_nodes = typed_solver_children(Stree, solver_idx, SOLVER, nodetyperepr)
        merge_comps = component_nodes.union(solver_nodes)
        G_parent, graphs[solver_idx] = merge_graph(G_parent, merge_comps, solve_vars, solver_idx, nodetyperepr)
    return graphs