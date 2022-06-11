from collections import OrderedDict
from itertools import chain
from datastructures.execution import sympy_fx_inputs, Component, edges_from_components
from datastructures.unitutils import get_unit, ureg
from compute import Var
from datastructures.workflow import NEQ, EQ, OBJ, OPT, SOLVE
from datastructures.graphutils import VAR, COMP, SOLVER
from datastructures.runpipeline import nestedform_to_mdao
from datastructures.unitutils import fx_with_units
import numpy as np

class SolverRef():
    def __init__(self, idx, modelref):
        self.model = modelref
        self.idx = idx
class Model():
    def __init__(self, solver=None, nametyperepr=None):
        solvers_options = {1: {"type":OPT}} if solver==OPT else {}
        self.solvers_options = solvers_options
        self.nametyperepr = nametyperepr if nametyperepr is not None else {VAR: '{}', COMP: 'f{}', SOLVER: 's{}'}
        self.components = []
        self.comp_options = {}
        self.var_options = {}
        self.Ftree = OrderedDict()
        self.Stree = dict()
        self.Vtree = dict()
        self.root = SolverRef(1,self)
        self.comp_by_var = dict()
        self.idmapping = {}

    def generate_formulation(self):
        edges = edges_from_components(self.components)
        tree = self.Ftree, self.Stree, self.Vtree
        return edges, tree

    def generate_mdao(self, mdf=True):
        edges, tree = self.generate_formulation()
        return nestedform_to_mdao(edges, tree, self.components, self.solvers_options, self.comp_options, self.var_options, self. nametyperepr, mdf)

def var_from_expr(name, expr, unit=None, forceunit=False):
    newvar = Var(name, unit=unit)
    newvar.forceunit=forceunit # TODO: HACK for sympy function
    if not forceunit:
        fx, inputs, inpunitsflat = sympy_fx_inputs(expr)
        rhs_unit = get_unit(fx, inpunitsflat)[0] #get_unit returns a vector output depending on fx
        if unit != None:
            assert ureg(unit).dimensionality == rhs_unit.dimensionality
        else:
            newvar.varunit = ureg.Quantity(1, rhs_unit.to_base_units().units)
    return newvar

def addvars(model, right, left=None):
    d = {elt.varid: elt for elt in right}
    if left:
        d[left.varid] = left
    model.idmapping.update(d)

def calculateval(invars, fx):
    varvals = tuple()
    assumed = dict()
    defaultvarval = 1.0
    for var in invars:
        if var.varval is not None:
            varvals += (var.varval,)
            if var.assumed:
                assumed.update(var.assumed)
        else:
            assumed[var] = defaultvarval
            varvals += (defaultvarval,)
    varval = np.squeeze(fx(*varvals)) #TODO: remove squeeze
    return varval, assumed

def evalexpr(right, unit=None):
    fx, invars, inpunitsflat = sympy_fx_inputs(right)
    fxunits = fx_with_units(fx, inpunitsflat, (ureg(unit),), True)
    varval, assumed = calculateval(invars, fxunits)
    dummyvar = Var('dummy', varval, unit)
    dummyvar.assumed = assumed
    return dummyvar

def adda(solver, left, right, *args, **kwargs):
    model = solver.model
    comp_idx = len(model.Ftree)
    if callable(right):
        invars = args[0]
        inunitsflat = tuple(invar.varunit for invar in invars)
        outunit = kwargs.get('unit', None)
        outunitsflat = (ureg(outunit),)
        fx = fx_with_units(right, inunitsflat, outunitsflat, overrideoutunits=True)
        outvars = (left,)
        if isinstance(left, str):
            left = Var(left, unit=outunit)
        addvars(model, invars, left)
        invars = tuple(str(var) for var in invars)
        comp = Component(fx, invars, outvars, component=comp_idx)
    else:
        if isinstance(left, str):
            left = var_from_expr(left, right, *args, **kwargs)
        addvars(model, right.free_symbols, left)
        comp = Component.fromsympy(right, left, component=comp_idx)
        invars = right.free_symbols
    # precompute the value of left if available
    left.varval, left.assumed = calculateval(invars, comp.function)
    model.components.append(comp)
    model.Ftree[comp_idx] = solver.idx
    model.comp_by_var[left]=comp_idx
    return left

def addf(solver, right, name=None):
    model = solver.model
    addvars(model, right.free_symbols)
    comp_idx = name if name else len(model.Ftree)
    comp = Component.fromsympy(right, component=comp_idx)
    model.components.append(comp)
    model.Ftree[comp_idx] = solver.idx
    return comp_idx

def addsolver(solver, comps=None, solvefor=None):
    comps = comps if comps else []
    solvefor = solvefor if solvefor else []
    model = solver.model
    next_solver_idx = max(chain(model.Stree.keys(),[1]))+1
    model.Stree[next_solver_idx] = solver.idx
    for elt in comps:
        model.Ftree[elt] = next_solver_idx
    for elt in solvefor:
        model.Vtree[elt] = next_solver_idx
    return SolverRef(next_solver_idx, model)

def setsolvefor(solver, solvefor, varoptions=None):
    model = solver.model
    for elt in solvefor:
        model.Vtree[elt.varid] = solver.idx
    if varoptions:
        varoptionsstr = {str(key):var for key,var in varoptions.items()}
    else:
        varoptionsstr = {}
    model.var_options.update(varoptionsstr)

def addoptfunc(solver, right, name=None, functype=EQ):
    comp_idx = addf(solver, right, name=None)
    model = solver.model
    model.comp_options[comp_idx] = functype

def addineq(solver, right, name=None):
    addoptfunc(solver, right, name, NEQ)

def addeq(solver, right, name=None):
    addoptfunc(solver, right, name, EQ)

def addobj(solver, right, name=None):
    addoptfunc(solver, right, name, OBJ)