import sympy as sp
from utils import invmap
import networkx as nx
from sympy import S
import gurobipy as gp
from gurobipy import GRB

def reassigneq(left, right, var):
    return sp.solve(left-right, var)[0]

def reassign(eqs, outset):
    neweqs = {}
    for key, (left, right) in eqs.items():
        newleft = outset[key]
        neweqs[key] = (newleft, reassigneq(left, right, newleft))
    return neweqs

def getallvars(eqs):
    vrs = set()
    for left, right in eqs.values():
        vrs = vrs.union({left}).union(right.free_symbols)
    return vrs

def getdofs(eqs):
    ins = set()
    outvrs = set()
    for left, _ in eqs.values():
        outvrs = outvrs.union({left})
    for _, right in eqs.values():
        ins = ins.union({elt for elt in right.free_symbols 
            if elt not in outvrs})
    return ins

def set_ins(ins, default_output, eqvars, eqs):
    full_output = default_output.copy() # shallow copy should be fine
    full_eqs = eqs.copy()
    full_eqvars = eqvars.copy()
    for idx, elt in enumerate(ins):
        eqname = 'in{}'.format(idx)
        full_eqs[eqname] = (elt, S(0.99))
        full_eqvars[eqname] = {elt}
        full_output[eqname] = elt
    return full_output, full_eqvars, full_eqs

def possible_outputs(expr):
    expr_expanded = sp.expand(expr)
    drop_args = set()
    solve_args = ()
    for arg in sp.preorder_traversal(expr_expanded):
        if arg.func == sp.Pow:
            base, power = arg.args
            if base.func == sp.Symbol and power != -1:
                drop_args.add(base)
        elif arg.func == sp.Symbol and arg not in drop_args:
            solve_args += (arg,)
    return solve_args

def encode_condensation(C, inv_outset):
    order = []
    for n in nx.topological_sort(C):
        members = C.nodes[n]['members']
        if len(members)==1:
            elt = next(iter(members))
            if elt in inv_outset:
                order.append(inv_outset[elt])
        else:
            order.append(tuple(inv_outset[elt] for elt in members))
    return order

def mdf_order(eqvars, outset):
    edges = [(inp, outset[eq]) for eq, inps in eqvars.items() 
        for inp in inps if inp != outset[eq]]
    D = nx.DiGraph(edges)
    C = nx.condensation(D)
    inv_outset = {val: key for key,val in outset.items()}
    order = encode_condensation(C, inv_outset)
    return order

def idf_order(eqvars, outset):
    edges = [(inp, outset[eq]) for eq, inps in eqvars.items() 
        for inp in inps if inp != outset[eq]]
    D = nx.DiGraph(edges)
    C = nx.condensation(D)
    order = [C.nodes[n]['members'] for n in nx.topological_sort(C)]
    inv_outset = invmap(outset)
    return [inv_outset[elt] for elt in order]

def invert_edges(edges):
    return [(b, a) for (a,b) in edges]

def scc(x, edges):
    D = nx.DiGraph([(r,j) if x[r,j].x>1e-6 else (j,r) for (r, j) in edges])
    return list(nx.strongly_connected_components(D))

def resolve(eqns, vrs, edges, maxiter=50):
    G = nx.Graph(edges)
    n_eqs = len(eqns)
    # make sure edges are in the right order
    m = gp.Model('cycles')
    m.setParam('OutputFlag', False )
    x = m.addVars(edges, name="assign", vtype=GRB.BINARY)
    c = m.addVar(lb=0.0)
    #r = m.addVars(eqs, name="reverse")
    # Matching eqs:
    fcons = m.addConstrs((x.sum(j,'*') == 1 for j in eqns), name='equations')
    varcons = m.addConstrs((x.sum('*',j) <= 1 for j in vrs), name='variables')
    #rev = m.addConstrs((1-gp.quicksum([x[(key, elt)] for elt in var])<=r[key] for key,var in allowed.items()), name='reversals')
    m.setObjective(c, GRB.MINIMIZE)
    converged = False
    counter = 0
    graphs = []
    sccvars = []
    results = []
    while not converged and counter<maxiter:
        counter+=1
        m.optimize()
        if m.status!=2:
            print('No feasible solution')
            break
        sol = [(r,j) for (r, j) in edges if x[r,j].x>1e-6]
        #rev = [elt[0] for elt in r if r[elt].x>1e-6]
        #print('S',sol)#, rev)
        graphs.append(nx.DiGraph([(r,j) if x[r,j].x>1e-6 else (j,r) for (r, j) in edges]))
        cycles = [elt for elt in scc(x, edges) if len(elt)>1]
        print('C', cycles)
        print(counter, m.objVal, [len(cycle)/2 for cycle in cycles]) #[any(elt in cycle for cycle in cycles) for elt in rev]
        results.append([len(cycle)/2 for cycle in cycles])
        #if len(cycles)==0:
        #    converged = True
        #else:
        for idx, cycle in enumerate(cycles):
            g = G.subgraph(cycle)
            nf_cycle = len(cycle)//2
            #print([edge for edge in g.edges()])
            #sccvars.append(m.addVar(name='z_'+str(nf_cycle)+"_"+str(idx)))
            #m.addConstr(sccvars[-1]==1)
            m.addConstr(gp.quicksum(x[edge] if edge in x else x[edge[::-1]] for edge in g.edges())<=c)
        #m.write("myfile.lp")
        m.addConstr(gp.quicksum(x[(r,j)] for r,j in sol)<= n_eqs-1) # eliminate the full set of combination
        #print("----------------------------")
    return sol