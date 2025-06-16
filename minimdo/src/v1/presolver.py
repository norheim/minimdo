import gurobipy as gp
from gurobipy import GRB
import networkx as nx

def encode_condensation(C, inv_outset, formtype='mdf'):
    order = []
    for n in nx.topological_sort(C):
        members = C.nodes[n]['members']
        if len(members)==1:
            elt = next(iter(members))
            if elt in inv_outset:
                order.append(inv_outset[elt])
        else:
            cycle = tuple(inv_outset[elt] for elt in members)
            if formtype=='mdf':
                order.append(cycle)
            elif formtype=='idf':
                order.extend(cycle)
    return order

def mdf_order(eqvars, outset):
    # TODO: need to make sure we cannot have two equations with same output
    edges = [(inp, outset[eq]) for eq, inps in eqvars.items() 
        for inp in inps if (inp != outset[eq] and outset[eq] is not None)]
    D = nx.DiGraph(edges)
    C = nx.condensation(D)
    inv_outset = {val: key for key,val in outset.items()}
    order = encode_condensation(C, inv_outset)
    return order

def flatten_and_transform_condensation(c_topo, outset):
    rev_lookup = {var:key for key,var in outset.items()}
    flattened_order = [rev_lookup[elt] for s in c_topo for elt in s if elt in rev_lookup]
    return flattened_order

def flatten_order(order): 
    order_flattened = []
    for elt in order:
        if type(elt)==tuple:
            order_flattened.extend(elt)
        else:
            order_flattened.append(elt)
    return order_flattened

def idf_order(eqvars, outset):
    edges = [(inp, outset[eq]) for eq, inps in eqvars.items() 
        for inp in inps if inp != outset[eq]]
    D = nx.DiGraph(edges)
    C = nx.condensation(D)
    inv_outset = {val: key for key,val in outset.items()}
    order = encode_condensation(C, inv_outset, formtype='idf')
    return order

def var_matched_cons(x, j, not_input):
    if j in not_input:
        return x.sum('*',j) == 1
    else:
        return x.sum('*',j) <= 1

def scc(x, edges):
    D = nx.DiGraph([(r,j) if x[r,j].x>1e-6 else (j,r) for (r, j) in edges])
    return list(nx.strongly_connected_components(D))

def resolve(eqns, vrs, edges, maxiter=50, not_input=None):
    if not_input == None:
        not_input = []
    output = []
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
    varcons = m.addConstrs((var_matched_cons(x, j, not_input) for j in vrs), name='variables')
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
        output.append({
            'C': cycles,
            'OBJ': m.objVal,
            'CLEN': [len(cycle)/2 for cycle in cycles],
            'SOL': sol
        })
        #[any(elt in cycle for cycle in cycles) for elt in rev]
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
    return output