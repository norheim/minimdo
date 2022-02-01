import sympy as sp
from utils import invmap
import networkx as nx
from sympy import S
import gurobipy as gp
from gurobipy import GRB

# Takes a variable incidence mapping, an outputset and generates a directed equations data structure
def direct_eqs(eqv, outset):
    return {idx: (outset[idx], tuple(elt for elt in val if elt != outset[idx])) for idx,val in eqv.items()}

def reassigneq(left, right, var):
    if left == var:
        return right
    diff = right if left == None else left-right
    sol = sp.solve(diff, var)
    if len(sol) == 2:
        return sol[1] # return the bigger number for now for quadratics
    else:
        return sol[0]

# TODO: Not sure if the Equation class is really necessary
class Equation():
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.default_output = left if len(left.free_symbols)==1 else None

    def reassign(self, var):
        newright = reassigneq(self.left, self.right, var)
        return Equation(var, newright)

    def __repr__(self) -> str:
        return '{} = {}'.format(self.left, self.right)

def idx_eqlist(eqlist):
    return {idx: eq for idx, eq in enumerate(eqlist)}

def reassign(eqs, outset):
    neweqs = {}
    for key, (left, right) in eqs.items():
        newleft = outset[key]
        neweqs[key] = (newleft, reassigneq(left, right, newleft))
    return neweqs

def eqsonly(eqs):
    return {idx: (eq.left, eq.right) for idx, eq in enumerate(eqs)}

def eqvars(eqs):
    return {key: right.free_symbols | (left.free_symbols if left else set()) for key,(left,right) 
    in eqs.items()}

def default_out(eqs):
    return {key: left for key,(left,right) in eqs.items()}

def getallvars(eqs):
    vrs = set()
    for left, right in eqs.values():
        vrs = vrs.union({left}).union(right.free_symbols)
    return vrs

# TODO: technically only works if we don't have an ill defined set of eqs where multiple equations can have same output
def default_in(eqlist, eqdictin=True, count_never_output=True):
    if eqdictin:
        eqlist = eqlist.values()
    ins = set()
    outvrs = set()
    for left, _ in eqlist:
        if left:
            outvrs = outvrs.union({left})
    for _, right in eqlist:
        ins = ins.union({elt for elt in right.free_symbols 
            if elt not in outvrs and (not elt.always_input or count_never_output)})
    return ins

# This one is indepedent of sympy
def default_in_raw(eqlist, eqdictin=True):
    if eqdictin:
        eqlist = eqlist.values()
    ins = set()
    outvrs = set()
    for left, _ in eqlist:
        if left:
            outvrs = outvrs.union({left})
    for _, right in eqlist:
        ins = ins.union({elt for elt in right 
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
        for inp in inps if inp != outset[eq]]
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

def invert_edges(edges):
    return [(b, a) for (a,b) in edges]

def scc(x, edges):
    D = nx.DiGraph([(r,j) if x[r,j].x>1e-6 else (j,r) for (r, j) in edges])
    return list(nx.strongly_connected_components(D))

def var_matched_cons(x, j, not_input):
    if j in not_input:
        return x.sum('*',j) == 1
    else:
        return x.sum('*',j) <= 1


def resolve(eqns, vrs, edges, maxiter=50, not_input=None):
    if not_input == None:
        not_input = []
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