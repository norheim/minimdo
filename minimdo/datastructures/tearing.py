import networkx as nx
import gurobipy as gp
from gurobipy import GRB

def dir_graph(undir_edges, rightset, selected=None):
    selected = selected if selected !=None else []
    # edge order independent in selected and undir_edges
    for node1, node2 in undir_edges:
        if ((node1,node2) in selected or (node2,node1) in selected):
            yield (node2,node1) if node2 in rightset else (node1, node2)
        else:
            yield (node1,node2) if node2 in rightset else (node2, node1)

def get_cycles(yval, yref, X):
    # elimination set = nodes with 1s
    elimset = {i for i in yref.keys() if yval[i] > 0.5}
    D = X.subgraph(X.nodes()-elimset)
    S = nx.strongly_connected_components(D)
    cycles = [elt for elt in S if len(elt)>1]
    return cycles, elimset

def sccelim(model, where):
    if where == GRB.Callback.MIPSOL:
        y_sol = model.cbGetSolution(model._y)
        cycles,_ = get_cycles(y_sol, model._y, model._X)
        for _, cycle in enumerate(cycles):
            cycle_eqnodes = cycle.intersection(model._y.keys())
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
    return cycles, elimset

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
            scycles = nx.simple_cycles(D)
            for cycle in scycles:
                all_cycles.add(tuple(cycle))
    return all_cycles

def assign_get_cycles_heuristic2(xval, xref, rightset):
    edges_left_right = xref.keys()
    selected = tuple((right, left) for left, right in edges_left_right if xval[left, right] > 0.5)
    D = nx.DiGraph(dir_graph(edges_left_right, rightset, selected))
    S = nx.simple_cycles(D)
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

def min_arc_set_assign(edges_left_right, leftset, rightset):
    G = nx.Graph(edges_left_right)
    m = gp.Model('cycles')
    m.setParam('OutputFlag', False )
    m.setParam('TimeLimit', 3600)
    x = m.addVars(edges_left_right, name="assign", vtype=GRB.BINARY)
    # A variable node can have maximum one ouput edge (possibly of none)
    m.addConstrs((x.sum('*',i) <= 1 for i in rightset), name='equations')
    # An equation node shall have one output edge unless part of elimination set
    m.addConstrs((x.sum(i,'*') <= 1 for i in leftset), name='variables')
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

def var_matched_cons(x, j, not_input=None):
    if not_input == None:
        not_input = []
    if j in not_input:
        return x.sum('*',j) == 1
    else:
        return x.sum('*',j) <= 1

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