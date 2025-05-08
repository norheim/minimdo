from modeling.gen6.api import symbolic, get_constraints, EqualsTo
from engine.torchengine import EliminateAnalysis, EliminateAnalysisMergeResiduals, ParallelResiduals, ElimResidual, ParallelAnalysis, ipoptsolver, AnalyticalSetSympy, FunctionSympy
from engine.torchdata import generate_optim_functions
from functools import partial
from engine.torchdata import load_vals, print_formatted_table
import torch
import uuid, base64
import sympy as sp
from dataclasses import dataclass
from scipy import optimize

@dataclass
class SetProps:
    indices: object
    analysis: object
    residual: object


def generate_short_id():
    # Generate a UUID
    full_uuid = uuid.uuid4()
    # Convert UUID to bytes and encode in base64
    uuid_bytes = full_uuid.bytes
    encoded = base64.urlsafe_b64encode(uuid_bytes)
    # Take first 8 characters and remove any base64 padding
    short_id = encoded[:8].decode('ascii').rstrip('=')
    return short_id

def build_leafs(sets, indices, mfs):
    if len(mfs.supersets) == 1:
        cid, _ = mfs.supersets[0]
        return SetProps(
            indices=indices,
            residual=sets[cid].residual, 
            analysis=sets[cid].analysis
            )
    else:
        return SetProps(
            indices=indices,
            residual=EliminateAnalysisMergeResiduals(functions=[sets[cid].residual for cid,_ in mfs.supersets]),
            analysis=EliminateAnalysis([sets[cid].analysis for cid,_ in mfs.supersets])
            )
    
def build_recursive(sets, indices, mfs, return_residual=False):
    if isinstance(mfs, MFunctionalSetLeaf):
        return build_leafs(sets, indices, mfs)
    else:
        return build(sets, indices, mfs.elim, mfs.parallel, mfs.residuals, return_residual)

# OLD version: parallel and elim will contain analysis objects and res will contain residual objects
def build(sets, indices, elim, parallel, res, return_residual=False):
    built_res_obj = [build_recursive(sets, indices, mfs) for mfs in res]
    built_res = [b.residual for b in built_res_obj]
    built_parallel = [build_recursive(sets, indices, mfs) for mfs in parallel]
    built_parallel_analysis = [b.analysis for b in built_parallel]
    built_elim = [build_recursive(sets, indices, mfs) for mfs in elim]
    built_elim_analysis = [b.analysis for b in built_elim]
    
    if built_res and not built_parallel_analysis:
        RES = [EliminateAnalysisMergeResiduals(functions=built_res)]
    else:
        RES = []

    if built_parallel_analysis and not built_res:
        T = [ParallelResiduals(analyses=built_parallel_analysis, 
                               functions=[])]
    elif built_res:
        T1 = [ParallelResiduals(analyses=built_parallel_analysis, functions=built_res)]
        T = [EliminateAnalysisMergeResiduals(functions=T1, flatten=True)]
    else:
        T = RES
    
    R = EliminateAnalysis(built_elim_analysis, T, flatten=True)
    
    if T:
        solvefor_parallel = torch.tensor([p.structure[1] for p in built_parallel_analysis], dtype=torch.int64)
        solvefor_residual = torch.tensor([r.analysis.structure[1] for r in built_res_obj], dtype=torch.int64)
        solvefor= torch.cat([solvefor_parallel, solvefor_residual])
        bnds = [(None, None) for _ in solvefor]
        ipsolver = partial(ipoptsolver, bnds_problem=bnds)
        An = ElimResidual(R, solvefor, indices, 
                solver=ipsolver,
                solvefor_raw=True)
        Res = R
    else:
        An = R
        Res = EliminateAnalysisMergeResiduals(functions=[b.residual for b in built_elim])
    return SetProps(indices=indices, analysis=An, residual=Res)

def build_opt(sets, indices, elim, parallel, res, eqs, ineq, obj, x0):
    built_res_obj = [build_recursive(sets, indices, mfs) for mfs in res]
    built_res = [b.residual for b in built_res_obj]+eqs
    built_parallel = [build_recursive(sets, indices, mfs) for mfs in parallel]
    built_parallel_analysis = [b.analysis for b in built_parallel]
    built_elim = [build_recursive(sets, indices, mfs) for mfs in elim]
    built_elim_analysis = [b.analysis for b in built_elim]
    
    if built_res and not built_parallel_analysis:
        RES = EliminateAnalysisMergeResiduals(functions=built_res)
    else:
        RES = None

    if built_parallel_analysis and not built_res:
        T = ParallelResiduals(analyses=built_parallel_analysis, 
                               functions=[])
    elif built_res:
        T1 = [ParallelResiduals(analyses=built_parallel_analysis, functions=built_res)]
        T = EliminateAnalysisMergeResiduals(functions=T1, flatten=True)
    else:
        T = RES
    
    obj_ineq_eq = [obj]
    if ineq:
        obj_ineq_eq.append(ineq)
    if T:
        obj_ineq_eq.append(T)
    P = EliminateAnalysis(built_elim_analysis, obj_ineq_eq)
    solvefor_indices = P.structure[0]

    objidx = 0
    ineqidx = 1 if ineq else None
    residx = 1 + (ineqidx==1) if T else None
    xguess, obj_function, ineq_function, eq_function, dobj, dineq, deq, hobj =  generate_optim_functions(P, solvefor_indices, x0, objective=objidx, residuals=residx, inequalities=ineqidx, inequality_direction='positive-null')

    ineqlen, eqlen = len(ineq_function(xguess)), len(eq_function(xguess))
    constraints = [{'type': 'eq', 'fun': eq_function, 'jac': deq}] if eqlen >= 1 else []
    constraints.append({'type': 'ineq', 'fun': ineq_function, 'jac': dineq}) if ineqlen >= 1 else []
    
    return obj_function, dobj, xguess, constraints, indices, solvefor_indices

def interpet_constraint(sets, indices, constraints):
    ineq_constraints =[]
    eq_constraints = []
    for constraint in constraints:
        if isinstance(constraint, EqualsTo):
            lhs, rhs = constraint.lhs, constraint.rhs
            if isinstance(lhs, sp.Expr) and isinstance(rhs, sp.Symbol):
                sets[rhs] = AnalyticalSetSympy(lhs, outputvar=rhs, indices=indices)
            elif isinstance(rhs, sp.Expr) and isinstance(lhs, sp.Symbol):
                sets[lhs] = AnalyticalSetSympy(rhs, outputvar=lhs, indices=indices)
        else:
            lhs, rhs = constraint.lhs, constraint.rhs
            # check if it is <= or >= based on sp.LessThan and sp.GreaterThan
            if isinstance(constraint, sp.LessThan):
                ineq_constraints.append(FunctionSympy(lhs - rhs, indices))
            elif isinstance(constraint, sp.GreaterThan):
                ineq_constraints.append(FunctionSympy(rhs - lhs, indices))
            else:
                eq_constraints.append(lhs - rhs)
    return ineq_constraints, eq_constraints

class MFunctionalSet():
    def __init__(self, *supersets, constraints=None, objective=None):
        self.supersets = supersets
        self.elim = []
        self.parallel = []
        self.residuals = []
        self.constraints = constraints if constraints is not None else []
        self.obj = objective if objective is not None else None

    def functionalsubsetof(self, *supersets):
        self.supersets += supersets
        return self
    
    def subsetof(self, *constraints):
        self.constraints += constraints
        return self
    
    def minimize(self, objective):
        self.obj = objective
        return self
    
    def gather_sets(self, indices=None):
        all_constraints = [(None, c) for c in self.constraints]
        constraints = list(self.supersets) #deep copy
        counter = 0
        while constraints:
            counter += 1
            c = constraints.pop()
            if hasattr(c, "supersets"):
                constraints += c.supersets
            else:
                all_constraints.append(c)
        obj_expr = self.obj.expr if self.obj else None
        sets, ineq_constraints, eq_constraints, obj, indices = get_constraints(all_constraints, obj_expr, indices=indices)
        ineq_constraints_merged = EliminateAnalysisMergeResiduals(functions=ineq_constraints) if ineq_constraints else []
        return sets, ineq_constraints_merged, eq_constraints, obj, indices

    def config(self, elim=None, parallel=None, residuals=None):
        if elim is None and parallel is None and residuals is None:
            return self
        MFS = MFunctionalSet(*self.supersets, constraints=self.constraints, objective=self.obj)
        MFS.elim = elim if elim is not None else []
        MFS.parallel = parallel if parallel is not None else []
        MFS.residuals = residuals if residuals is not None else []
        return MFS
    
    def build(self, sets=None, indices=None, return_residual=False):
        sets, _, _, _, indices_for_build =self.gather_sets(indices) if (sets is None or indices is None) else (sets, None, None, None, indices)
        return build(sets, indices_for_build, self.elim, self.parallel, self.residuals, return_residual=return_residual)
    
    def build_opt(self, sets=None, indices=None, x0=None):
        sets, ineqs, eqs, obj, indices =self.gather_sets(indices)
        x0array = load_vals(x0, indices, isdict=True)
        return build_opt(sets, indices, self.elim, self.parallel, self.residuals, eqs, ineqs, obj, x0array)
    
    def solve(self, x0=None):
        obj_function, dobj, xguess, constraints, idxs, solidxs = self.build_opt(x0=x0)
        xsol = optimize.minimize(obj_function, xguess, jac=dobj, constraints=constraints, method='SLSQP')
        return xsol, idxs, solidxs


class MFunctionalSetLeaf(MFunctionalSet):
    def __init__(self, *supersets, autoid=True):
        supersets = supersets if supersets is not None else []
        supersets = [(generate_short_id(), c) for c in supersets] if autoid else supersets
        super().__init__(*supersets)

    def build(self, sets=None, indices=None, return_residual=False):
        sets, _, _, _, indices =self.gather_sets(indices) if (sets is None or indices is None) else (sets, None, None, None, indices)
        return build_leafs(sets, indices, self)


def univariate_set(mfs):
    sets, _ = mfs.gather_sets()
    return list(sets.values())[0]