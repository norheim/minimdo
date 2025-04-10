from trash.inputresolver import reassigneq
import itertools
import sympy as sp
from scipy.optimize import fsolve
import numpy as np
import torch
import time

class AnalysisFunction():
    def __init__(self, triplet, indices):
        inputs, outputs, function = triplet
        self.function = function
        self.arg_indices_in = [indices[i] for i in inputs]
        self.arg_indices_out = [indices[i] for i in outputs]
        input_indices = torch.cat(self.arg_indices_in) if self.arg_indices_in else torch.tensor([]) 
        output_indices = torch.cat(self.arg_indices_out) if self.arg_indices_out else torch.tensor([])
        self.structure = (input_indices, output_indices) 
        self.structure_full = {elt:input_indices.tolist() for elt in output_indices.tolist()}

    def __call__(self, x):
        all_inputs = [x[arg_indices] for arg_indices in self.arg_indices_in]
        out = self.function(*all_inputs)
        for i, arg_index_out in enumerate(self.arg_indices_out):
            x = x.index_put((arg_index_out,), out[i])
        return x

class FunctionRawIndices():
    def __init__(self, function, rawindices):
        self.arg_indices = rawindices
        self.function = function
        input_indices = torch.cat(self.arg_indices) if rawindices else torch.tensor([])
        self.structure = (input_indices, torch.tensor([]))
        self.structure_full = (input_indices.tolist(),) # assumes only one output

    def __call__(self, x):
        all_inputs = [x[arg_indices] for arg_indices in self.arg_indices]
        return self.function(*all_inputs)

class Function(FunctionRawIndices):
    def __init__(self, tupl, indices):
        inputs, function = tupl
        indices_raw = [indices[i] for i in inputs]
        super().__init__(function, indices_raw)


class AnalyticalSet():
    def __init__(self, triplet, indices, forceresidual=None):
        inputs, outputs, function = triplet
        self.analysis = AnalysisFunction(triplet, indices)
        input_indices = (torch.cat([indices[i] for i in inputs]) if inputs else torch.tensor([])) #flatten
        output_indices = (torch.cat([indices[i] for i in outputs]) if outputs else torch.tensor([])) #flatten
        if forceresidual:
            residual_function = Function(forceresidual, indices)
        else:
            residual_function = FunctionRawIndices(lambda x, y: y-function(*x), 
                                                       (input_indices, output_indices))
        self.residual = residual_function

def lambdify_with_variables(expression, variables=None):
    variables = tuple(sorted(expression.free_symbols,
                             key=lambda x: x.name)) if variables is None else variables
    if isinstance(expression, (int, float, sp.Rational, sp.Float)):
        tensor = torch.tensor([float(expression)], dtype=torch.float64)
        function = lambda : tensor
    else:
        function = sp.lambdify(variables, expression, torch)
    return function, variables

class FunctionSympy(Function):
    def __init__(self, expression, indices):
        function, variables = lambdify_with_variables(expression)
        self.expression = expression
        super().__init__((variables, function), indices)


class AnalyticalSetSympy(AnalyticalSet):
    def __init__(self, expression, outputvar=None, indices=None, 
                 forceresidual=None, normalize=False):
        outputvars = (outputvar,) if outputvar is not None else ()
        outputvar = 0 if outputvar is None else outputvar
        self.indices = indices
        self.expression = expression
        self.outputvar = outputvar
        analysis_function, variables = lambdify_with_variables(expression)
        self.variables = variables
        self.residualexpr = (expression/outputvar-1 if normalize else expression-outputvar) if not forceresidual else forceresidual
        residual_function, residual_variables = lambdify_with_variables(self.residualexpr) 
        triplet = (variables, outputvars, analysis_function)
        tuplet = (residual_variables, residual_function)
        super().__init__(triplet, indices, forceresidual=tuplet)

    def reassign(self, new_outputvar=None, **kwargs):
        if new_outputvar == self.outputvar:
            return self
        outputvar = 0
        if self.outputvar != 0:
            outputvar = self.outputvar
        newexpr = reassigneq(None, self.expression-outputvar, new_outputvar, **kwargs)
        return AnalyticalSetSympy(newexpr, new_outputvar, self.indices, forceresidual=self.residualexpr)

def get_analysis_structure(analysis_structures):
    eliminated_output = {}
    full_structure = {}
    for structure in analysis_structures:
        eliminated_output_buffer = {}
        for struct_out, struct_in in structure.items():
            full_structure[struct_out] = []
            for i in struct_in:
                extension = [i] if i not in eliminated_output else eliminated_output[i]
                for elt in extension:
                    if elt not in full_structure[struct_out]:
                        full_structure[struct_out] += [elt] # the last part is to avoid self reference
            eliminated_output_buffer[struct_out] = eliminated_output.get(struct_out,[])+full_structure[struct_out]
        eliminated_output.update(eliminated_output_buffer)
    flattened_list = list(itertools.chain.from_iterable(full_structure.values()))
    structure_in = torch.tensor(sorted(dict.fromkeys(flattened_list))) #hack
    structure_out = torch.tensor(list(full_structure))
    return structure_in, structure_out, full_structure

def get_function_structure(functional_structures, full_structure=None):
    full_structure = full_structure if full_structure is not None else {}
    full_func_structure = ()
    for struct_in in functional_structures:
        all_inputs = []
        for i in struct_in:
            all_inputs += [elt for elt in full_structure.get(i, [i]) if elt not in all_inputs]
        full_func_structure += (all_inputs,)
    flattened_list = list(itertools.chain.from_iterable(full_func_structure))
    structure_in = torch.tensor(list(dict.fromkeys(flattened_list)))
    structure_out = torch.tensor([])
    return structure_in, structure_out, full_func_structure

def get_full_structure(analysis_structures=None, functional_structures=None):
    full_structure = {}
    structure_in = torch.tensor([])
    structure_out = torch.tensor([])
    if analysis_structures:
        structure_in, structure_out, full_structure = get_analysis_structure(analysis_structures)
    if functional_structures:
        structure_in, structure_out, full_structure = get_function_structure(functional_structures, full_structure)
    return structure_in, structure_out, full_structure

class EliminateAnalysis():
    def __init__(self, analyses=None, functions=None, flatten=False):
        self.flatten = flatten
        self.analyses_missing = analyses is None
        self.analyses = analyses if analyses else []
        self.functions = functions if functions else []
        structure_input, structure_output, structure_full = get_full_structure(
            [a.structure_full for a in self.analyses],
            [felt for f in self.functions for felt in f.structure_full])
        self.structure = (structure_input, structure_output)
        self.structure_full = structure_full

    def __call__(self, x):
        for a in self.analyses:
            x = a(x)
        if not self.functions:
            if self.analyses_missing:
                return [torch.tensor([])] # empty vector
            else:
                return x
        if self.flatten: # single function
            return self.functions[0](x)
        else:
            return [f(x) for f in self.functions]

class EliminateAnalysisMergeResiduals(EliminateAnalysis):
    def __init__(self, analyses=None, functions=None, flatten=False):
        super().__init__(analyses, functions, flatten)

    def __call__(self, x):
        output = super().__call__(x)
        if self.functions:
            return torch.cat(output)
        else:
            return output
    
class ParallelAnalysis():    
    # sharedvars have to be in the input
    def __init__(self, analyses):
        self.analyses = analyses
        structure_full = {key:val for a in analyses 
                          for key,val in a.structure_full.items()}
        inputs = torch.tensor([idx for a in analyses for idx in a.structure[0]])         
        outputs = torch.tensor([idx for a in analyses for idx in a.structure[1]]) #assumes no repeated outputs
        self.structure = (inputs, outputs)
        self.structure_full = structure_full

    def __call__(self, x):
        y = x.clone()
        for a in self.analyses:
            output_indices = a.structure[1]
            y[output_indices] = a(x)[output_indices]
        return y

def parallel_structure(analyses, functions, sharedvars):
        # TODO: fix this
        shared_structure = {key: a.structure[0].tolist() for a in analyses for key,val in a.structure_full.items()}
        functional_structures = [elt for f in functions for elt in f.structure_full]
        structure_full = ()
        structure_full += get_function_structure(functional_structures, shared_structure)[2]
        structure_full += tuple(list(dict.fromkeys([key]+val)) for key,val in shared_structure.items()  if key in sharedvars)
        flattened_list = list(itertools.chain.from_iterable(structure_full))
        structure_in = torch.tensor(list(dict.fromkeys(flattened_list)))
        return structure_in, structure_full
# Check that shared variables are in input structure
class ParallelResiduals():
    def __init__(self, analyses, functions, sharedvars=None, indices=None):
        # sharedvars have to be in the input
        if sharedvars is None:
            self.sharedvars = [idx for a in analyses for idx in a.structure[1]] #assumes no repeated outputs
        else:
            if indices is None:
                self.sharedvars = sharedvars
            else:
                self.sharedvars = [idx for i in sharedvars for idx in indices[i]]
        self.analyses = analyses
        self.functions = functions
        structure_in, structure_full = parallel_structure(analyses, functions, self.sharedvars)
        self.structure = (structure_in, torch.tensor([]))
        self.structure_full = structure_full

    def __call__(self, x):
        y = x.clone()
        r = []
        for a in self.analyses:
            output_indices = a.structure[1] #this needs to be optimized
            constraint_indices = torch.tensor([elt for elt in output_indices if elt in self.sharedvars])
            y[output_indices] = a(x)[output_indices]
            r.append(y[constraint_indices]-x[constraint_indices]) # or x[output_indices]/y[output_indices]-1
        if self.functions and self.analyses:
            return [f(y) for f in self.functions]+[torch.cat(r)]
        elif self.functions:
            return [f(y) for f in self.functions]+[torch.tensor([])]
        else:
            return torch.cat(r)


def generate_eval_and_gradient(function, solvefor, x):
    def eval_and_gradient():
        def eval_function(y):
            x[solvefor] = torch.from_numpy(y).to(x.dtype)
            r = function(x)
            result = r.detach().numpy()
            return result

        def recover_gradient(y):
            x[solvefor] = torch.from_numpy(y).to(x.dtype)
            jacobian = torch.autograd.functional.jacobian(function, x)
            J = jacobian[:, solvefor]
            return J
        
        def recover_jacobian(y=None):
            x[solvefor] = torch.from_numpy(y).to(x.dtype)
            jacobian = torch.autograd.functional.jacobian(function, x)
            return jacobian
        
        return eval_function, recover_gradient, recover_jacobian
        
    return eval_and_gradient

import cyipopt
from collections import namedtuple
OptProblem = namedtuple('OptProblem', ['objective', 'constraints', 'gradient', 'jacobian', 'intermediate'])

def ipoptsolvercon(xguess, obj_function, ineq_function, eq_function, dobj, dineq, deq, bnds_problem):
    ineqlen, eqlen = len(ineq_function(xguess)), len(eq_function(xguess))

    def all_constraints(x):
        return np.concatenate([ineq_function(x), eq_function(x)])

    def all_constraints_jac(x):
        if eqlen == 0:
            if ineqlen != 0:
                return dineq(x)
            else:
                return np.tensor([])
        elif ineqlen == 0:
            return deq(x)
        return np.concatenate([dineq(x), deq(x)], axis=0)

    OptProblem = namedtuple('OptProblem', ['objective', 'constraints', 'gradient', 'jacobian', 'intermediate'])

    lb,ub = zip(*bnds_problem)
    cl = np.concatenate([-np.inf*np.ones(ineqlen), np.zeros(eqlen)])
    cu = np.concatenate([np.zeros(ineqlen), np.zeros(eqlen)])

    storeiter = [0]

    def logiter(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        storeiter[0] = iter_count

    # define the problem
    probinfo = OptProblem(obj_function, all_constraints, dobj, all_constraints_jac, logiter)

    prob = cyipopt.Problem(n=len(xguess), m=len(cu), lb=lb, ub=ub, cl=cl, cu=cu, 
                        problem_obj=probinfo)
    prob.add_option('max_iter', 8000)
    #prob.add_option('acceptable_tol', 1e-6)
    start_time = time.time()
    sol, info = prob.solve(xguess)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return sol,info,storeiter,elapsed_time

def ipoptsolver(eval_function, xguess, fprime, bnds_problem=None, debug=False):
    # define the problem
    eqlen = len(xguess)
    lb,ub = zip(*bnds_problem)
    cl = np.zeros(eqlen)
    cu = np.zeros(eqlen)

    probinfo = OptProblem(lambda x: 0, eval_function, lambda x: np.zeros(eqlen), fprime, 
                        lambda alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials: None)

    prob = cyipopt.Problem(n=len(xguess), m=len(cu), lb=lb, ub=ub, cl=cl, cu=cu, 
                        problem_obj=probinfo)
    #prob.add_option('mu_strategy', 'adaptive')
    prob.add_option('tol', 1e-7)
    sol, info = prob.solve(xguess)
    if debug:
        print(info)
    return sol

class ElimResidualFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, solver, function, solvefor, inputs, x):
        eval_and_gradient = generate_eval_and_gradient(function, solvefor, x)
        eval_function, recover_gradient, recover_jacobian = eval_and_gradient()
        xguess = x.data[solvefor].numpy() 
        xsol = solver(eval_function, xguess, fprime=recover_gradient) #sets x in place
        J = recover_jacobian(xsol)
        ctx.save_for_backward(J, solvefor, inputs)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        J, solvefor, inputs = ctx.saved_tensors
        # we want J_x to be whatever is not in J_u (aka J[:, all elements except for solvefor])
        J_u = J[:, solvefor]
        J_x = J[:, inputs]
        dudx = np.linalg.solve(-J_u, J_x)
        # the next two lines take into account that the 
        # gradient is one for bypass variables and zero for solvefor
        result = torch.zeros_like(grad_output)
        result[inputs] = grad_output[solvefor] @ dudx
        result += grad_output
        result[solvefor] = torch.zeros_like(grad_output)[solvefor] #hack to get dtype right
        return None, None, None, None, result

class ElimResidual(torch.nn.Module):
    def __init__(self, function, solvefor, indices, solver=None, solvefor_raw=False):
        super(ElimResidual, self).__init__()
        self.function = function
        solvefor_indices = torch.cat([indices[i] for i in solvefor]) if not solvefor_raw else solvefor
        input_indices = torch.tensor([val for idx in indices.values() for val in idx if 
                                 (val not in solvefor_indices) and (val in function.structure[0])])
        self.solvefor = solvefor_indices
        self.inputs = input_indices
        self.solver = fsolve if solver is None else solver
        self.structure = (self.inputs, self.solvefor)
        self.structure_full = {solvevar.item(): self.inputs.tolist() for solvevar in self.solvefor}
    
    def forward(self, x):
        return ElimResidualFunc.apply(self.solver, self.function, self.solvefor, self.inputs, x)
