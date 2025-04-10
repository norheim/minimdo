from collections import OrderedDict
from itertools import chain
from modeling.execution import sympy_fx_inputs, Component, edges_from_components, component_hash
from modeling.unitutils import get_unit, ureg
from modeling.compute import Var
from graph.workflow import NEQ, EQ, OBJ, OPT, SOLVE
from graph.graphutils import VAR, COMP, SOLVER, all_variables, copy_dicts, edges_to_Ein_Eout
from solver.runpipeline import nestedform_to_mdao
from modeling.unitutils import fx_with_units
import numpy as np

class SolverRef():
    def __init__(self, idx, modelref, name=None):
        self.model = modelref
        self.idx = idx
        self.name = name
class Model():
    def __init__(self, solver=None, nametyperepr=None, rootname=1):
        self.nametyperepr = nametyperepr if nametyperepr is not None else {VAR: '{}', COMP: 'f{}', SOLVER: 's{}'}
        solvers_options = {1: {"type":OPT}} if solver==OPT else {}
        self.solvers_options = solvers_options
        self.comp_options = {}
        self.var_options = {}
        self.components = []
        self.Ftree = OrderedDict()
        self.Stree = dict()
        self.Vtree = dict()
        self.root = SolverRef(rootname, self, name=rootname)
        self.comp_by_var = dict()
        self.idmapping = {}

    def generate_formulation(self):
        edges = edges_from_components(self.components)
        tree = self.Ftree, self.Stree, self.Vtree
        return edges, tree

    def generate_mdao(self, mdf=True):
        edges, tree = self.generate_formulation()
        return nestedform_to_mdao(edges, tree, self.components, self.solvers_options, self.comp_options, self.var_options, self. nametyperepr, mdf)

def var_params(model, edges):
    return {var for var in all_variables(*edges_to_Ein_Eout(edges)) if model.idmapping[var].always_input}

def edges_no_param(model, edges):
    Ein, Eout, Rin = edges
    Ein_noparam = {comp:tuple(var for var in compvars if not model.idmapping[var].always_input) for comp,compvars in Ein.items()}
    return Ein_noparam, Eout, Rin

#TODO: this overlaps significantly with Component.fromsympy, consider merging
def var_from_expr(name, expr, unit=None, forceunit=False):
    newvar = Var(name, unit=unit)
    newvar.forceunit=forceunit # TODO: HACK for sympy function
    if not forceunit:
        fx, inputs, inpunitsflat = sympy_fx_inputs(expr, "numpy")
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
                assumed[var] = var.varval
        else:
            assumed[var] = defaultvarval
            varvals += (defaultvarval,)
    varval = np.squeeze(fx(*varvals)) #TODO: remove squeeze
    return varval, assumed

def evalexpr(right, unit=None, override_out_unit=True):
    fx, invars, inpunitsflat = sympy_fx_inputs(right)
    fxunits = fx_with_units(fx, inpunitsflat, (ureg(unit),), override_out_unit)
    varval, assumed = calculateval(invars, fxunits)
    dummyvar = Var('dummy', varval, unit)
    dummyvar.assumed = assumed
    return dummyvar

def check_for_component(components, expr, outputs):
    hash_comps = [hash(elt) for elt in components]
    new_hash = component_hash(expr, outputs)
    if new_hash in hash_comps:
        idx_comp = hash_comps.index(new_hash)
        return components[idx_comp]
    else:
        return False

def create_component(comp_id, left, right, args, kwargs):
    # left could be None in the case that there is no left hand side
    # right could be either a function or a sympy expression
    # args and kwargs are arguments that relate to the left variable (e.g. varid, varval, unit)
    if callable(right):
        invars = args[0]
        inunitsflat = tuple(invar.varunit for invar in invars)
        outunit = kwargs.get('unit', None)
        outunitsflat = (ureg(outunit),)
        fx = fx_with_units(right, inunitsflat, outunitsflat, overrideoutunits=True)
        outvars = (left,)
        if isinstance(left, str):
            left = Var(left, unit=outunit)
        comp = Component(fx, invars, outvars, comp_id=comp_id, arg_mapping=True)
    else:
        if isinstance(left, str):
            left = var_from_expr(left, right, *args, **kwargs)
        comp = Component.fromsympy(right, left, component=comp_id, arg_mapping=True)
    return left, comp

def generate_new_component_id(components):
    ids = [component.id for component in components]
    newid = max(ids)+1 if ids else 0
    return newid

def addequation(components, right, *args, **kwargs):
    # creates a new component if it does not exists, and returns 
    # the existing component otherwise
    left, args = args[0], args[1:] if args else None # allows for left to be optional
    eqcomp = check_for_component(components, right, (left,))
    comp_isnew = False
    leftvar = None
    if not eqcomp:
        comp_id = kwargs.pop('compname', generate_new_component_id(components))
        leftvar, eqcomp = create_component(comp_id, left, right, args, kwargs)
        comp_isnew = True
    # precompute the value of left if available
    # if the component already exists it re-evaluates the value of left
    if leftvar is not None:
        leftvar.varval, leftvar.assumed = calculateval(eqcomp.mapped_inputs, eqcomp.function)

    return leftvar, comp_isnew, eqcomp

def adda(solver, left, right, *args, **kwargs):
    model = solver.model
    leftvar, comp_isnew, eqcomp = addequation(model.components, right, left, *args, **kwargs)
    if comp_isnew:
        addvars(model, eqcomp.mapped_inputs, leftvar)
        model.components.append(eqcomp)
        model.comp_by_var[left] = eqcomp.id
    else:
        leftvar = model.idmapping[eqcomp.outputs[0]]
    model.Ftree[eqcomp.id] = solver.idx
    return leftvar

def addf(solver, right, name=None):
    model = solver.model
    leftvar, comp_isnew, eqcomp = addequation(model.components, right, name=name)
    if comp_isnew:
        addvars(model, eqcomp.mapped_inputs, leftvar)
        model.components.append(eqcomp)
    model.Ftree[eqcomp.id] = solver.idx
    return eqcomp.id

def addsolver(solver, comps=None, solvefor=None, name=None, idbyname=False):
    comps = comps if comps else []
    solvefor = solvefor if solvefor else []
    model = solver.model
    next_solver_idx = name if idbyname else max(chain(model.Stree.keys(),[1]))+1
    model.Stree[next_solver_idx] = solver.idx
    for elt in comps:
        model.Ftree[elt] = next_solver_idx
    for elt in solvefor:
        model.Vtree[elt] = next_solver_idx
    return SolverRef(next_solver_idx, model, name)

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

def merge(formulation, edges, tree, copysolvers=True, copyvars=True):
    medges,mtree = formulation
    new_edges = tuple(E | {key:val for key,val in mE.items() if key not in E} for mE,E in zip(medges, edges))
    Ftree,Vtree,Stree = copy_dicts(mtree)
    new_Ftree = OrderedDict(tree[0])
    for key,val in Ftree.items():
        if key not in edges[0]:
            new_Ftree[key] = val
    new_treeSV = ()
    for i in range(2):
        mE = mtree[i+1] if [copysolvers, copyvars][i] else {}
        E = tree[i+1]
        new_treeSV += (E | {key:val for key,val in mE.items() if key not in E},)
    return new_edges, (new_Ftree, new_treeSV[0], new_treeSV[1])