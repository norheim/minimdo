import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from itertools import islice

def feedbacks(D, order=None):
    visited = set()
    guess = set()
    feedback_components = set()
    for node in order:
        feedback_vars = {elt for elt in D.successors(node) if visited.intersection(D.successors(elt))}
        if feedback_vars:
            feedback_components.add(node)
        guess = guess.union(feedback_vars)
        visited.add(node)
    return guess, feedback_components    

def dir_graph(undir_edges, rightset, selected=None):
    selected = selected if selected !=None else []
    # edge order independent in selected and undir_edges
    for node1, node2 in undir_edges:
        if ((node1,node2) in selected or (node2,node1) in selected):
            yield (node2,node1) if node2 in rightset else (node1, node2)
        else:
            yield (node1,node2) if node2 in rightset else (node2, node1)

def limited_simple_cycles(D, limit=100):
    return islice(nx.simple_cycles(D),limit)

def get_cycles(yval, yref, X):
    # elimination set = nodes with 1s
    elimset = {i for i in yref.keys() if yval[i] > 0.5}
    D = X.subgraph(X.nodes()-elimset)
    S = nx.strongly_connected_components(D)
    cycles = [elt for elt in S if len(elt)>1]
    return cycles, elimset

# Runs significantly slower
def get_cycles2(yval, yref, X):
    elimset = {i for i in yref.keys() if yval[i] > 0.5}
    D = X.subgraph(X.nodes()-elimset)
    cycles = limited_simple_cycles(D)
    return cycles, None

def sccelim(model, where):
    if where == GRB.Callback.MIPSOL:
        y_sol = model.cbGetSolution(model._y)
        cycles,_ = get_cycles(y_sol, model._y, model._X)
        for _, cycle in enumerate(cycles):
            cycle_eqnodes = set(cycle).intersection(model._y.keys())
            model.cbLazy(gp.quicksum(model._y[node] for node in cycle_eqnodes)>=1)

def min_arc_set(edges, dout, vrs, eqns):
    X = nx.DiGraph(dir_graph(edges, vrs, dout.items()))
    # make sure edges are in the right order
    m = gp.Model('cycles')
    m.setParam('OutputFlag', False )
    m.setParam('TimeLimit', 100)
    y = m.addVars(eqns, name="elimination", vtype=GRB.BINARY)
    m.setObjective(y.sum('*'), GRB.MINIMIZE)
    m._edges = edges
    m._X = X
    m._y = y
    m.Params.lazyConstraints = 1
    m.optimize(sccelim)
    cycles, elimset = get_cycles(m.getAttr('x', y), y, X)
    return cycles, elimset, m

## Changing inputs and outputs
def heuristic_permute_tear(undir_edges, rightset):
    # edge order independent
    G = nx.Graph(undir_edges)
    G_original = G.copy()
    assignment = tuple()
    vertexelim = set()
    while G.nodes():
        degree = dict(G.degree(rightset))
        mindegarg = min(degree, key=degree.get)
        argneighbors = list(G.neighbors(mindegarg))
        G.remove_node(mindegarg)
        if degree[mindegarg] != 0:
            assignment += ((mindegarg,argneighbors[0]),)
        else:
            vertexelim.add((mindegarg, tuple(G_original.neighbors(mindegarg))))
        for neighbor in argneighbors:
            G.remove_node(neighbor)
    return assignment, vertexelim


def generate_all_scc2(dedges, velim):
    all_cycles = set() # make sure we don't run into repeated cycles
    for v, neighbors in velim:
        D = nx.DiGraph(dedges)
        for u in neighbors:
            D.remove_edge(u,v)
            D.add_edge(v,u)
            scycles = limited_simple_cycles(D)
            for cycle in scycles:
                all_cycles.add(tuple(cycle))
    return all_cycles

def assign_get_cycles_heuristic2(xval, xref, rightset):
    edges_left_right = xref.keys()
    selected = tuple((right, left) for left, right in edges_left_right if xval[left, right] > 0.5)
    D = nx.DiGraph(dir_graph(edges_left_right, rightset, selected))
    S = limited_simple_cycles(D)
    cycles_original = {tuple(elt) for elt in S}
    keep_edges = [(left,right) for (left,right) in edges_left_right if (right,left) not in selected]
    #print(selected)
    #print(edges_left_right)
    #print(keep_edges)
    assignment, vertexelim = heuristic_permute_tear(keep_edges, rightset)
    #print(assignment, vertexelim)
    dedges = list(dir_graph(keep_edges, rightset, assignment))
    #print(dedges)
    cycles = generate_all_scc2(dedges, vertexelim)
    dedges2 = list(dir_graph(edges_left_right, rightset, assignment+selected))
    cycles2 = generate_all_scc2(dedges2, vertexelim)
    return cycles_original.union(cycles).union(cycles2)

def assigncycleelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        x_sol = model.cbGetSolution(model._x)
        cycles = assign_get_cycles_heuristic2(x_sol, model._x, model._rightset)
        for idx, cycle in enumerate(cycles):
            g = model._G.subgraph(cycle)
            model.cbLazy(gp.quicksum(model._x[edge] if edge in model._x else model._x[edge[::-1]] for edge in g.edges())<=len(cycle)/2-1)

def var_matched_cons(x, j, not_input=None):
    if not_input == None:
        not_input = []
    if j in not_input:
        return x.sum('*',j) == 1
    else:
        return x.sum('*',j) <= 1

def var_matched_cons_reversed(x, i, not_input=None, not_output=None):
    not_input = [] if not_input == None else not_input
    not_output = [] if not_output == None else not_output
    eqconstant = 1 if i not in not_output else 0
    if i in not_input:
        return x.sum(i, '*') == eqconstant
    else:
        return x.sum(i, '*') <= eqconstant

def min_arc_set_assign(edges_left_right, leftset, rightset, not_input=None, not_output=None):
    G = nx.Graph(edges_left_right)
    m = gp.Model('cycles')
    m.setParam('OutputFlag', False )
    m.setParam('TimeLimit', 100)
    x = m.addVars(edges_left_right, name="assign", vtype=GRB.BINARY)
    # A variable node can have maximum one ouput edge (possibly of none)
    m.addConstrs((x.sum('*',i) <= 1 for i in rightset), name='equations')
    # An equation node shall have one output edge unless part of elimination set
    m.addConstrs((var_matched_cons_reversed(x,i, not_input, not_output) for i in leftset), name='variables')
    m.setObjective(x.sum('*'), GRB.MAXIMIZE)
    m._rightset = rightset
    m._G = G
    m._x = x
    m.Params.lazyConstraints = 1
    m.optimize(assigncycleelim)
    return m.getAttr('x', x), m

def scc(sol, edges):
    D = nx.DiGraph([(r,j) if (r,j) in sol else (j,r) for (r,j) in edges])
    return list(nx.strongly_connected_components(D))

def outset_from_assignment(xval, xref):
    outset = tuple((i, j) for i, j in xref.keys()
                                if xval[i, j] > 0.5)
    return outset

def outset_from_solution(m):
    return outset_from_assignment(m.getAttr('x', m._x), m._x)

def recoversol(xval, xref, edges):
    selected = outset_from_assignment(xval, xref)
    dir_edges = ([(r,j) if (r,j) in selected else (j,r) for (r,j) in edges])
    D = nx.DiGraph(dir_edges)
    S = nx.strongly_connected_components(D)
    cycles = [elt for elt in S if len(elt)>1]
    return cycles, D

def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        x_sol = model.cbGetSolution(model._x)
        cycles, _ = recoversol(x_sol, model._x, model._edges)
        for idx, cycle in enumerate(cycles):
            g = model._G.subgraph(cycle)
            model.cbLazy(gp.quicksum(model._x[edge] if edge in model._x else model._x[edge[::-1]] for edge in g.edges())<=model._c)

def min_max_scc(edges, vrs, eqns):
    G = nx.Graph(edges)
    # make sure edges are in the right order
    m = gp.Model('cycles')
    m.setParam('OutputFlag', False )
    m.setParam('TimeLimit', 10)
    x = m.addVars(edges, name="assign", vtype=GRB.BINARY)
    c = m.addVar(lb=0.0)
    # Matching eqs:
    m.addConstrs((x.sum(j,'*') == 1 for j in eqns), name='equations')
    m.addConstrs((var_matched_cons(x, j) for j in vrs), name='variables')
    m.setObjective(c, GRB.MINIMIZE)
    m._edges = edges
    m._G = G
    m._x = x
    m._c = c
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim)
    cycles, D = recoversol(m.getAttr('x', x), x, edges)
    return max([len(cycle)/2 for cycle in cycles]+[0]), m.getAttr('x', x), m

def get_scc(xval, xref, rightset):
    edges_left_right = xref.keys()
    selected = tuple((right, left) for left, right in edges_left_right if xval[left, right] > 0.5)
    D = nx.DiGraph(dir_graph(edges_left_right, rightset, selected))
    S = nx.strongly_connected_components(D)
    scc = [elt for elt in S if len(elt)>1]
    return scc, D

# def assignminscc(model, where):
#     if where == GRB.Callback.MIPSOL:
#         # make a list of edges selected in the solution
#         x_sol = model.cbGetSolution(model._x)
#         allscc, _ = get_scc(x_sol, model._x, model._rightset)
#         print(allscc)
#         for idx, scc in enumerate(allscc):
#             g = model._G.subgraph(scc)
#             model.cbLazy(gp.quicksum(model._x[edge] if edge in model._x else model._x[edge[::-1]] for edge in g.edges())<=model._c)

def assignminscc2(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        x_sol = model.cbGetSolution(model._x)
        sol = outset_from_assignment(x_sol, model._x)
        D = nx.DiGraph([(r,j) if (r,j) in sol else (j,r) for (r,j) in model._x])
        #cycles = limited_simple_cycles(D)
        S = nx.strongly_connected_components(D)
        cycles = [elt for elt in S if len(elt)>1]
        #cycles = assign_get_cycles_heuristic2(x_sol, model._x, model._rightset)
        for idx, cycle in enumerate(cycles):
            comps_in_cycle = len(cycle)/2
            g = model._G.subgraph(cycle) #g.edges()
            #gedges = list(zip(cycle, cycle[1:]+[cycle[0]]))
            model.cbLazy(gp.quicksum(model._x[edge] if edge in model._x else model._x[edge[::-1]] for edge in g.edges())<=comps_in_cycle-1+model._c/comps_in_cycle)

def min_max_scc2(edges_left_right, leftset, rightset, not_input=None, not_output=None, timeout=10):
    G = nx.Graph(edges_left_right)
    m = gp.Model('cycles')
    m.setParam('OutputFlag', False )
    m.setParam('TimeLimit', timeout)
    x = m.addVars(edges_left_right, name="assign", vtype=GRB.BINARY)
    c = m.addVar(lb=0.0)
    # A variable node can have maximum one ouput edge (possibly of none)
    m.addConstrs((x.sum('*',i) == 1 for i in rightset), name='equations') #needs to be equality
    # An equation node shall have one output edge unless part of elimination set
    m.addConstrs((var_matched_cons_reversed(x,i, not_input, not_output) for i in leftset), name='variables')
    m.setObjective(c, GRB.MINIMIZE)
    m._rightset = rightset
    m._G = G
    m._x = x
    m._c = c
    m.Params.lazyConstraints = 1
    m.optimize(assignminscc2)
    if m.Status == 9: # time limit
        return False, m
    else:
        return m.getAttr('x', x), m