from collections import defaultdict
from telnetlib import GA
from datastructures.graphutils import flat_graph_formulation, solver_children, Node, COMP, SOLVER, VAR, root_solver, all_components, sources, edges_to_Ein_Eout
from datastructures.mergegraph import merge_graph, split_graph

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
    root_id = root_solver(tree)
    merge_order = level_order_tree(Stree, root=root_id)[::-1]
    for solver_idx in merge_order:
        solve_vars = typed_solver_children(Vtree, solver_idx, VAR, nodetyperepr)
        component_nodes = typed_solver_children(Ftree, solver_idx, COMP, nodetyperepr)
        solver_nodes = typed_solver_children(Stree, solver_idx, SOLVER, nodetyperepr)
        merge_comps = component_nodes.union(solver_nodes)
        G_parent, graphs[solver_idx] = merge_graph(G_parent, merge_comps, solve_vars, solver_idx, nodetyperepr)
    return graphs

def root_sources(edges, trees):
    all_srcs = sources(*edges_to_Ein_Eout(edges))
    _,_,Vtree = trees
    srcs = all_srcs - Vtree.keys()
    # root = root_solver(tree)
    # g = build_typedgraph(edges, tree, nodetyperepr)
    # G = g[root]
    # Ein = defaultdict(list)
    # Eout = defaultdict(list)
    # for fr,to in G.edges():
    #     if fr.nodetype == VAR:
    #         Ein[to].append(fr)
    #     else:
    #         Eout[fr].append(to)
    # srcs = sources(Ein, Eout)
    return srcs