import sympy as sp
import networkx as nx
from sympy import S

# Takes a variable incidence mapping, an outputset and generates a directed equations data structure
def direct_eqs(eqv, outset):
    return {idx: (outset[idx], tuple(elt for elt in val if elt != outset[idx])) for idx,val in eqv.items()}

def reassigneq(left, right, var=None, **kwargs):
    if left == var:
        return right
    diff = right if left == None else left-right
    if var == None:
        return diff
    sol = sp.solve(diff, var, **kwargs)
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
    return {key: left for key,(left,_) in eqs.items()}

def getallvars(eqs, sympy=True):
    vrs = set()
    for left, right in eqs.values():
        right_symbols = right.free_symbols if sympy else set(right)
        vrs = vrs.union({left}).union(right_symbols)
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

def invert_edges(edges):
    return [(b, a) for (a,b) in edges]
