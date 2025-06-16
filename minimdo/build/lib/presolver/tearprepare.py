from graph.graphutils import copy_dicts, all_variables, all_edges
from src.v2.tearing import dir_graph, min_arc_set_assign

def execute_tearing(edges, not_input, not_output):
    edges_for_solving = copy_dicts(edges)
    eqnidxs = list(edges_for_solving[1].keys())
    varidxs = all_variables(*edges_for_solving)
    graph_edges_minassign = all_edges(*edges_for_solving)
    edges_left_right = list(dir_graph(graph_edges_minassign, eqnidxs, {}))
    xsol,m = min_arc_set_assign(edges_left_right, varidxs, eqnidxs, 
                                not_input, not_output)
    outset_opt = {right:left for left, right in edges_left_right 
                  if (left,right) in edges_left_right 
                  and xsol[left, right] > 0.5}
    return outset_opt