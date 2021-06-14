import sympy as sp
from utils import invmap
import networkx as nx
from sympy import S

def reassigneq(left, right, var):
    return sp.solve(left-right, var)[0]

def reassign(eqs, outset):
    neweqs = {}
    for key, (left, right) in eqs.items():
        newleft = outset[key]
        neweqs[key] = (newleft, reassigneq(left, right, newleft))
    return neweqs

def getdofs(eqs):
    ins = set()
    vrs = set()
    for left, _ in eqs.values():
        vrs = vrs.union({left})
    for _, right in eqs.values():
        ins = ins.union({elt for elt in right.free_symbols if elt not in vrs})
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

def encode_condensation(C, inv_outset):
    order = []
    for n in nx.topological_sort(C):
        members = C.nodes[n]['members']
        if len(members)==1:
            elt = next(iter(members))
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