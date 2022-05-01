from collections import OrderedDict
from itertools import chain
from execution import sympy_fx_inputs, Component
from unitutils import get_unit, ureg
from compute import Var

class SolverRef():
    def __init__(self, idx, modelref):
        self.model = modelref
        self.idx = idx
class Model():
    def __init__(self):
        self.components = []
        self.Ftree = OrderedDict()
        self.Stree = dict()
        self.Vtree = dict()
        self.root = SolverRef(1,self)
        self.comp_by_var = dict()

def var_from_expr(name, expr, unit=None, forceunit=False):
    newvar = Var(name, unit=unit)
    newvar.forceunit=forceunit # TODO: HACK for sympy function
    if not forceunit:
        fx, inputs, inpunitsflat = sympy_fx_inputs(expr)
        rhs_unit = get_unit(fx, inpunitsflat)[0] #get_unit returns a vector output depending on fx
        if unit != None:
            assert ureg(unit).dimensionality == rhs_unit.dimensionality
    if unit == None:
        newvar.varunit = ureg.Quantity(1, rhs_unit.to_base_units().units)
    return newvar

def adda(solver, left, right, *args, **kwargs):
    model = solver.model
    comp_idx = len(model.Ftree)
    if isinstance(left, str):
        left = var_from_expr(left, right, *args, **kwargs)
    comp = Component.fromsympy(right, left, component=comp_idx)
    model.components.append(comp)
    model.Ftree[comp_idx] = solver.idx
    model.comp_by_var[left]=comp_idx
    return left

def addf(solver, right, name=None):
    model = solver.model
    comp_idx = name if name else len(model.Ftree)
    comp = Component.fromsympy(right, component=comp_idx)
    model.components.append(comp)
    model.Ftree[comp_idx] = solver.idx
    return comp_idx

def addsolver(solver, comps=None, solvefor=None):
    comps = comps if comps else []
    solvefor = solvefor if solvefor else []
    model = solver.model
    next_solver_idx = max(chain(model.Stree.values(),[1]))+1
    model.Stree[next_solver_idx] = solver.idx
    for elt in comps:
        model.Ftree[elt] = next_solver_idx
    for elt in solvefor:
        model.Vtree[elt] = next_solver_idx
    return SolverRef(next_solver_idx, model)

def setsolvefor(solver, solvefor):
    model = solver.model
    for elt in solvefor:
        model.Vtree[elt.varid] = solver.idx