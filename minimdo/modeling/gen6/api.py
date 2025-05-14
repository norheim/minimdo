from typing import Any
import sympy as sp
from engine.torchengine import AnalyticalSetSympy, FunctionSympy, EliminateAnalysisMergeResiduals, ParallelResiduals, ElimResidual, EliminateAnalysis, ipoptsolvercon
from engine.torchdata import generate_indices, load_vals, generate_optim_functions
from scipy import optimize
import torch

class SymbolicExpression(sp.Expr):
    def __new__(cls, expr):
        if isinstance(expr, SymbolicExpression):
            return expr
        elif isinstance(expr, sp.Expr):
            obj = sp.Expr.__new__(cls)
            obj.expr = expr
            obj._unique_id = id(obj)
            return obj
        else:
            # Assume expr is a string representing a symbol name
            symbol = sp.Symbol(expr)
            obj = sp.Expr.__new__(cls)
            obj.expr = symbol
            obj._unique_id = id(obj)
            return obj

    def __add__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return SymbolicExpression(self.expr + other_expr)

    def __radd__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return SymbolicExpression(other_expr + self.expr)

    def __sub__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return SymbolicExpression(self.expr - other_expr)

    def __rsub__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return SymbolicExpression(other_expr - self.expr)

    def __mul__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return SymbolicExpression(self.expr * other_expr)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return SymbolicExpression(self.expr / other_expr)

    def __rtruediv__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return SymbolicExpression(other_expr / self.expr)

    def __pow__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return SymbolicExpression(self.expr ** other_expr)

    def __rpow__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return SymbolicExpression(other_expr ** self.expr)

    def __neg__(self):
        return SymbolicExpression(-self.expr)

    def __pos__(self):
        return self

    def __eq__(self, other):
        if isinstance(other, SymbolicExpression):
            return EqualsTo(self.expr, other.expr)
        else:
            return EqualsTo(self.expr, other)
        
    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return sp.StrictLessThan(self.expr, other_expr)

    def __le__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return sp.LessThan(self.expr, other_expr)

    def __gt__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return sp.StrictGreaterThan(self.expr, other_expr)

    def __ge__(self, other):
        other_expr = other.expr if isinstance(other, SymbolicExpression) else other
        return sp.GreaterThan(self.expr, other_expr)
    
    def __hash__(self):
        return hash((self.expr, self._unique_id))

    def __repr__(self):
        return f"SymbolicExpression({repr(self.expr)})"

    def __str__(self):
        return str(self.expr)
    
    def true_equals(self, other):
        if isinstance(other, SymbolicExpression):
            return self.expr.equals(other.expr)
        else:
            return self.expr.equals(other)
        
def wrap_sympy_function(func):
    def wrapped(*args, **kwargs):
        new_args = [a.expr if isinstance(a, SymbolicExpression) else a for a in args]
        new_kwargs = {
            k: (v.expr if isinstance(v, SymbolicExpression) else v)
            for k, v in kwargs.items()
        }
        result = func(*new_args, **new_kwargs)
        return SymbolicExpression(result)
    return wrapped

acos = wrap_sympy_function(sp.acos)
sin = wrap_sympy_function(sp.sin)
exp = wrap_sympy_function(sp.exp)
log = wrap_sympy_function(sp.log)
sqrt = wrap_sympy_function(sp.sqrt)

def symbolic(*args):
    return [SymbolicExpression(arg) for arg in args]

def get_indices_only(constraints, objective=None):
    symbols = set()
    for _, constraint in constraints:
        if hasattr(constraint.lhs, 'free_symbols'):
            symbols.update(constraint.lhs.free_symbols)
        if hasattr(constraint.rhs, 'free_symbols'):
            symbols.update(constraint.rhs.free_symbols)
    if objective is not None:
        symbols.update(objective.free_symbols)
    indices = generate_indices(symbols)
    return indices

def get_constraints(constraints, objective=None, indices=None):
    indices = indices if indices is not None else get_indices_only(constraints, objective)
    sets = {}
    ineq_constraints = []
    eq_constraints = []
    for cid, constraint in constraints:
        if isinstance(constraint, EqualsTo):
            lhs, rhs = constraint.lhs, constraint.rhs
            # check LHS first - otherwise will get bug
            if isinstance(rhs, (int, float, sp.Rational, sp.Float, sp.Expr)) and isinstance(lhs, sp.Symbol):
                sets[cid] = AnalyticalSetSympy(rhs, outputvar=lhs, indices=indices)
            elif isinstance(lhs, (int, float, sp.Rational, sp.Float, sp.Expr)) and isinstance(rhs, sp.Symbol):
                sets[cid] = AnalyticalSetSympy(lhs, outputvar=rhs, indices=indices)
        else:
            lhs, rhs = constraint.lhs, constraint.rhs
            # check if it is <= or >= based on sp.LessThan and sp.GreaterThan
            if isinstance(constraint, sp.LessThan):
                ineq_constraints.append(FunctionSympy(lhs - rhs, indices))
            elif isinstance(constraint, sp.GreaterThan):
                ineq_constraints.append(FunctionSympy(rhs - lhs, indices))
            else:
                eq_constraints.append(lhs - rhs)
    obj = FunctionSympy(objective, indices) if objective is not None else None        
    return sets, ineq_constraints, eq_constraints, obj, indices

class EqualsTo:
    def __init__(self, lhs, rhs):
        self.set = None           
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs} = {self.rhs}"

def lookup(symbol, sets):
    if isinstance(symbol, sp.Symbol):
        return sets[symbol]
    else:
        return symbol

class AnalyticalSetRaw():
    def __init__(self, analysis, residuals):
        self.analysis = analysis
        self.residual = residuals
        self._unique_id = None

    def __hash__(self):
        return hash(self._unique_id)

def gather_residuals(sets, indices, parallel_analysis=None, residuals=None, elimination_order=None):
    parallel_analysis = parallel_analysis if parallel_analysis is not None else []
    residuals = residuals if residuals is not None else []
    elimination_order = elimination_order if elimination_order is not None else []
    PA = [ParallelResiduals(analyses=[lookup(idx, sets).analysis for idx in parallel_analysis], functions=[], indices=indices)] if parallel_analysis else []
    pres_solvevars = torch.cat([lookup(idx, sets).analysis.structure[1] for idx in parallel_analysis]) if parallel_analysis else torch.tensor([], dtype=torch.int64)
    res = [lookup(idx, sets).residual for idx in residuals]
    res_solvevars = torch.cat([lookup(idx, sets).analysis.structure[1] for idx in residuals]) if residuals else torch.tensor([], dtype=torch.int64)
    solvevars = torch.cat([pres_solvevars, res_solvevars])
    res_functions = PA+res if len(solvevars) > 0 else None
    EA = EliminateAnalysisMergeResiduals(analyses=[lookup(idx, sets).analysis for idx in elimination_order], functions=res_functions)
    return EA, solvevars

class ConstraintSystem:
    def __init__(self, constraints=None):
        self.constraints = constraints if constraints is not None else []
        self.objective = None

    def add(self, *constraints):
        self.constraints.extend(constraints)

    def stategen(self, x0=None):
        x0 = x0 if x0 is not None else {}
        _,_,_,indices = self.formulate()
        return load_vals(x0, indices, isdict=True)
    
    def minimize(self, objf):
        _,_,_,indices = self.formulate()
        self.objective = FunctionSympy(objf, indices)
        return self
    
    @property
    def indices(self):
        _, _, _, indices = self.formulate()
        idxrev = {var.item():key for key,var in indices.items()}
        return indices, idxrev

    # make a getter
    @property
    def statevars(self):
        sets, _, _, _ = self.formulate()
        return list(sets.keys())

    def interpretation(self, elimination_order=None, parallel_analysis=None, residuals=None, default_solver=None):
        parallel_analysis = parallel_analysis if parallel_analysis is not None else []
        residuals = residuals if residuals is not None else []
        elimination_order = elimination_order if elimination_order is not None else []
        sets, _, _, indices = self.formulate()
        EA, solvevars = gather_residuals(sets, indices, parallel_analysis, residuals, elimination_order)

        all_residuals = EliminateAnalysisMergeResiduals(functions=[st.residual for st in sets.values()])
        if len(solvevars) > 0: # in case this is not a feedforward model
            ER = ElimResidual(EA, solvefor=solvevars, solvefor_raw=True, solver=default_solver, indices=indices)
            return AnalyticalSetRaw(ER, all_residuals)
        else:
            return AnalyticalSetRaw(EA, all_residuals)

    def formulate_advanced(self, eliminate_order=None, parallel_analysis=None, residuals=None):
        pass

    def formulate(self):
        # find all symbols used in the constraints
        symbols = set()
        for constraint in self.constraints:
            symbols.update(constraint.lhs.free_symbols)
            symbols.update(constraint.rhs.free_symbols)
        indices = generate_indices(symbols)
        sets = {}
        ineq_constraints = []
        eq_constraints = []
        for constraint in self.constraints:
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
        ineqs = EliminateAnalysisMergeResiduals(functions=ineq_constraints)
        eqs = EliminateAnalysisMergeResiduals(functions=eq_constraints)
        return sets, ineqs, eqs, indices

    def __repr__(self):
        return "\n".join(str(constraint) for constraint in self.constraints)
    
class Problem:
    def __init__(self) -> None:
        self.P = None

    def solve(self, x0=None, interpretations=None, **kwargs):
        # interp_dict = {}
        # for elt in interpretations:
        #     interp_dict[elt] = elt.interpretation(**kwargs)
        # sets, ineq_constraints, eq_constraints, indices = self.formulate()
        # EA = self.gather_residuals(sets, indices, **kwargs)
        # P = EliminateAnalysis(functions=[self.objective, EA, eq_constraints, ineq_constraints])
        solvefor_indices = self.P.structure[0]
        objidx, residx, eqidx, ineqidx = None,None,None,None
        xguess, obj_function, ineq_function, eq_function, dobj, dineq, deq, hobj =  generate_optim_functions(self.P, solvefor_indices, x0, objidx, residx, eqidx, ineqidx)
        ineqlen, eqlen = len(ineq_function(xguess)), len(eq_function(xguess))
        constraints = [{'type': 'eq', 'fun': eq_function, 'jac': deq}] if eqlen >= 1 else []
        constraints.append({'type': 'ineq', 'fun': ineq_function, 'jac': dineq}) if ineqlen >= 1 else []
        xsol = optimize.minimize(obj_function, xguess, constraints=constraints, method='SLSQP')
        return xsol