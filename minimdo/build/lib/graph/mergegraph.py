from graph.graphutils import Node, VAR, SOLVER, filter_comps, all_edges, all_varnodes, sources, merge_edges, all_variables, intermediary_variables, end_components
from functools import reduce
import networkx as nx

def get_edges(G):
    Ein, Eout = dict(), dict()
    for node in filter(lambda x: x.nodetype!=VAR, G.nodes()):
        Ein[node] = set(G.predecessors(node))
        Eout[node] = set(G.successors(node))
    return Ein, Eout

def split_graph(G, typed_mergelts):
    graph_Ein, graph_Eout = get_edges(G)
    typed_nonmergelts = {elt for elt in G.nodes() if elt.nodetype!=VAR and elt not in typed_mergelts}
    subgraph_Ein = dict(filter_comps(graph_Ein, typed_mergelts))
    subgraph_Eout= dict(filter_comps(graph_Eout, typed_mergelts))
    mergegraph_Ein = dict(filter_comps(graph_Ein, typed_nonmergelts))
    mergegraph_Eout= dict(filter_comps(graph_Eout, typed_nonmergelts))
    return (subgraph_Ein, subgraph_Eout), (mergegraph_Ein, mergegraph_Eout)

def subgraph_ins(subgraph, mergegraph, exclude_unique_sources=True):
    # Only source variables from the subgraph are candidates for inputs
    # Any intermediary or source variable in the graph is a candidate
    srcs_subgraph = sources(*subgraph)
    if exclude_unique_sources:
        graph_vars = all_variables(*mergegraph)
        return srcs_subgraph.intersection(graph_vars)
    else:
        return srcs_subgraph

def subgraph_outs(subgraph, mergegraph):
    # Both sinks and intermediary variables can be outputs:
    _, subgraph_Eout = subgraph
    mergegraph_Ein, _ = mergegraph
    return intermediary_variables(mergegraph_Ein, subgraph_Eout)

def generate_edges(*graphs):
    return reduce(lambda x,y: x+y, (all_edges(Ein, Eout) for Ein, Eout in graphs))

def merged_graph(subgraph, mergegraph, solver_idx, typed_solve_vars, nodetyperepr=None, exclude_unique_sources=True):
    ins_from_graph = subgraph_ins(subgraph, mergegraph,  exclude_unique_sources)
    outs_used_in_graph = subgraph_outs(subgraph, mergegraph)
    subgraph_node_ins = ins_from_graph-typed_solve_vars
    subgraph_node_outs = outs_used_in_graph.union(typed_solve_vars)
    solver_node = Node(solver_idx, SOLVER, nodetyperepr)
    solver_Einout = ({solver_node:subgraph_node_ins},
         {solver_node:subgraph_node_outs})
    edges = generate_edges(mergegraph, solver_Einout)
    return edges

def merge_graph(G, typed_mergelts, typed_solve_vars, solver_idx=0, nodetyperepr=None, exclude_unique_sources=True):
    subgraph, mergegraph = split_graph(G, typed_mergelts)
    # Allow for any source except it if it an output from the parent graph
    allowable_solvevars = sources(*subgraph)-all_varnodes(mergegraph[1])
    assert all(var in allowable_solvevars for var in typed_solve_vars)
    subgraph_edges = all_edges(*subgraph)
    subgraph_G = nx.DiGraph(subgraph_edges)
    mergedgraph_edges = merged_graph(subgraph, mergegraph, solver_idx, typed_solve_vars, nodetyperepr, exclude_unique_sources)
    mergegraph_G = nx.DiGraph(mergedgraph_edges)
    return mergegraph_G, subgraph_G