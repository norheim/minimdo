from graph.workflow import SOLVER
from src.v4.torchengine import EliminateAnalysis, EliminateAnalysisMergeResiduals, ParallelResiduals, ElimResidual, ParallelAnalysis, ipoptsolver, AnalyticalSetSympy, FunctionSympy
from src.v4.torchdata import generate_optim_functions
from functools import partial
from src.v4.torchdata import load_vals, print_formatted_table
from src.v5.problem import symbolic, get_constraints, EqualsTo
import torch
import uuid, base64
import sympy as sp
from dataclasses import dataclass
from scipy import optimize
from src.v1.inputresolver import reassigneq

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

def universal_combine(tensors):
    if not tensors:
        return torch.tensor([], dtype=torch.int64)
    tensors = [t.unsqueeze(0) if t.dim() == 0 else t for t in tensors]
    return torch.cat(tensors)

def get_sharedvars(built_parallel_analysis):
    elt1, elt2 = zip(*[elt.structure for elt in built_parallel_analysis])
    sharedidxs = set([elt.item() for elt in torch.cat(elt1)]).intersection(set([elt.item() for elt in torch.cat(elt2)]))
    return sharedidxs
    

# OLD version: parallel and elim will contain analysis objects and res will contain residual objects
def build(sets, indices, elim, parallel, res, eliminate_parallelstatevars=True, return_residual=False):
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

    if built_parallel_analysis:
        sharedidxs = get_sharedvars(built_parallel_analysis) if  eliminate_parallelstatevars else None
        if not built_res:
            T = [ParallelResiduals(analyses=built_parallel_analysis, 
                               functions=[], sharedvars=sharedidxs)]
        else:
            T1 = [ParallelResiduals(analyses=built_parallel_analysis, functions=built_res, sharedvars=sharedidxs)]
            T = [EliminateAnalysisMergeResiduals(functions=T1, flatten=True)]
    else:
        T = RES
    
    R = EliminateAnalysis(built_elim_analysis, T, flatten=True)
    
    if T:
        solvefor_parallel = universal_combine([p.structure[1] for p in built_parallel_analysis])
        solvefor_residual = universal_combine([r.analysis.structure[1] for r in built_res_obj])
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

def build_opt(sets, indices, elim, parallel, res, eqs, ineq, obj, x0, eliminate_parallelstatevars=True):
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

    if built_parallel_analysis:
        sharedidxs = get_sharedvars(built_parallel_analysis+[ineq]+eqs+[obj]) if  eliminate_parallelstatevars else None
        if not built_res:
            T = [ParallelResiduals(analyses=built_parallel_analysis, 
                               functions=[], sharedvars=sharedidxs)]
        else:
            T1 = [ParallelResiduals(analyses=built_parallel_analysis, functions=built_res, sharedvars=sharedidxs)]
            T = [EliminateAnalysisMergeResiduals(functions=T1, flatten=True)]
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
    
    def gather_constraints(self):
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
        return all_constraints

    def gather_sets(self, indices=None):
        all_constraints = self.gather_constraints()
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
        sets, ineqs, eqs, obj, indices_new =self.gather_sets(indices)
        x0array = load_vals(x0, indices_new, isdict=True)
        return build_opt(sets, indices_new, self.elim, self.parallel, self.residuals, eqs, ineqs, obj, x0array)
    
    def solve(self, x0=None):
        obj_function, dobj, xguess, constraints, idxs, solidxs = self.build_opt(x0=x0)
        xsol = optimize.minimize(obj_function, xguess, jac=dobj, constraints=constraints, method='SLSQP')
        return xsol, idxs, solidxs
    
    def reconfigure(self, output_set):
        # recursive implementation
        all_supersets = []
        for elt in self.supersets:
            if hasattr(elt, "supersets"):
                new_item = elt.reconfigure(output_set)
            elif isinstance(elt, tuple):
                cid, constraint = elt
                if cid in output_set:
                    new_lhs = output_set[cid].expr
                    new_rhs = reassigneq(constraint.lhs, constraint.rhs, new_lhs)
                    new_item = (cid, EqualsTo(new_lhs, new_rhs))
                else:
                    new_item = elt
            all_supersets.append(new_item)
        newMFS = MFunctionalSet(*all_supersets, constraints=self.constraints, objective=self.obj)
        return newMFS
    
    def config_from_order(self, elim_order):
        all_constraints = self.gather_constraints()
        all_functionsets = {idval: c for idval, c in all_constraints if isinstance(c, EqualsTo)}
        mfsets = [MFunctionalSetLeaf(*(all_functionsets[idval] for idval in sorted(group)), idvals=sorted(group)) for group in elim_order] #sorted to ensure order is preserved
        MFS = MFunctionalSet(*mfsets, constraints=self.constraints, objective=self.obj)
        MFS.elim = mfsets
        return MFS
    
    def config_from_workflow(self, workflow_order):
        MFS_root = MFunctionalSet()
        new_mfs = None
        all_constraints = self.gather_constraints()
        all_functionsets = {idval: c for idval, c in all_constraints if isinstance(c, EqualsTo)}
        for elt in workflow_order:
            if elt[0] == SOLVER:
                if new_mfs is not None:
                    new_mfs.elim = [new_mfs] # HACK for build_opt to work
                new_mfs = MFunctionalSetLeaf()
                MFS_root.functionalsubsetof(new_mfs)
            else:
                constraint_idx = elt[1]
                mf_constraint = all_functionsets[constraint_idx]
                new_mfs.supersets += ((constraint_idx, mf_constraint),)
        new_mfs.elim = [new_mfs]
        if len(MFS_root.supersets) == 1:
            MFS_root = MFS_root.supersets[0]
        MFS_root.constraints = self.constraints
        MFS_root.obj = self.obj
        return MFS_root # FIX: this should be the root solver 
            

class MFunctionalSetLeaf(MFunctionalSet):
    def __init__(self, *supersets, autoid=True, idvals=None):
        supersets = supersets if supersets is not None else []
        supersets = ([(generate_short_id(), c) for c in supersets] if not idvals else [(idval,c) for idval, c in zip(idvals, supersets)]) if autoid else supersets
        super().__init__(*supersets)

    def build(self, sets=None, indices=None, return_residual=False):
        sets, _, _, _, indices =self.gather_sets(indices) if (sets is None or indices is None) else (sets, None, None, None, indices)
        return build_leafs(sets, indices, self)


def univariate_set(mfs):
    sets, _ = mfs.gather_sets()
    return list(sets.values())[0]