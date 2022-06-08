import networkx as nx
import gurobipy as gp
from gurobipy import GRB

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

def min_max_scc(edges, vrs, eqns, n_eqs):
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
    return max([len(cycle)/2 for cycle in cycles]+[0]), m