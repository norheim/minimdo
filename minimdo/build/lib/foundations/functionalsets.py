import numpy as np
from collections.abc import Callable
from modeling.transformations import flatten_component
from scipy.optimize import fsolve

class ProjactableSet():
    def __init__(self) -> None:
        self.variables = list() # these are all the variables

    def project(self, variables):
        pass

class FeedForwardSolver():
    def __init__(self, functionals):
        self.functionals = functionals

    def solve(self, input_dict):
        local_dict = input_dict.copy()
        for F in self.functionals:
            output_dict = F.solve(local_dict)
            local_dict.update(output_dict)
        return local_dict

class DefaultResidualSolver():
    # This solver will only work in conjunction with functionals whose projectable is a ResidualSet
    def __init__(self, functional) -> None:
        self.functional = functional

    def residuals(self, input_dict=None):
        projectable = self.functional.projectable
        return np.hstack([res.evaldict(input_dict) 
                          for res in projectable.components]).flatten()

    def generate_residual(self, input_dict=None) -> Callable:
        local_dict = input_dict.copy()
        def local_residual(x_dict):
            local_dict.update(x_dict)
            return self.residuals(local_dict)
        return local_residual
                      
    def generate_residual_vector(self, input_dict=None) -> Callable:
        local_dict = input_dict.copy()
        functional = self.functional
        residual_function_dict = self.generate_residual(local_dict)
        def local_residual(x):
            x_dict = {var: x[idx] for idx, var in 
                            enumerate(functional.independent)}
            return residual_function_dict(x_dict)
        return local_residual
   
    def solve(self, input_dict=None):
        functional = self.functional
        var_shapes = [(var, 1 if var.shape is None else var.shape) for var in functional.independent]
        x_init = map(lambda args: input_dict.get(args[0], np.random.rand(args[1])), var_shapes)
        x_initial = np.concatenate(list(x_init))
        res_function = self.generate_residual_vector(input_dict)
        x_root = fsolve(res_function, x_initial)
        return {var: x_root[idx] for idx, var in 
                            enumerate(functional.independent)}

class EliminationSolver(DefaultResidualSolver):
    def __init__(self, functional, eliminations) -> None:
        super().__init__(functional)
        self.forwardsolver = FeedForwardSolver(eliminations)

    def residuals(self, input_dict=None):
        projectable = self.functional.projectable
        elimination_dict = self.forwardsolver.solve(input_dict)
        elimination_dict.update(input_dict)
        return np.concatenate([np.atleast_1d(res.evaldict(elimination_dict)) for res in projectable.components])
    
    #TODO: solve function should also populate the elimination variables by subsitution

class Functional():
    def __init__(self, projectable=None, projected=None, fixed=None,
                 name=None, problemid=None, solver=None) -> None:
        self.projectable = projectable
        self.name = name
        self.id = 1 if problemid is None else problemid
        self.projected = projected if projected is not None else list() # these are the non zero projected variables
        self.projected_fixed = fixed if fixed is not None else dict()
        self.independent = tuple(vr for vr in self.projectable.variables if 
                                 vr not in self.projected) if self.projectable is not None else list()
        self.solver = solver
        # self.graph = None

    def filter_to_projected(self, local_dict):
        return {var:local_dict.get(var, var.varval) for var in self.projected}

    def solve(self, local_dict=None):
        input_dict = self.filter_to_projected(local_dict)
        input_dict.update(self.projected_fixed)
        out_dict = self.solver.solve(input_dict) #returns a dictionary
        return {var:out_dict[var] for var in self.independent}

class FunctionalComp(Functional):
    def __init__(self, comp, name=None, problemid=None) -> None:
        if None in comp.outputs:
            residual_comp = comp
        else:
            residual_comp = flatten_component(comp)
        projectable = ResidualSet([residual_comp])
        super().__init__(projectable, tuple(comp.mapped_inputs), name, problemid)
        self.independent = tuple(comp.mapped_outputs) #overwrite just in case
        self.projected_fixed = {var:var.varval for var in self.projected if var.always_input}
        self.component = comp
        
    def solve(self, local_dict=None, solver=None):
        input_dict = self.filter_to_projected(local_dict)
        out_tuple = self.component.evaldict(input_dict)
        return {var:out_tuple[idx]
                for idx, var in enumerate(self.independent)}

def unique_components_variables(components):
    variables = ()
    for comp in components:
        for var in comp.mapped_inputs + comp.mapped_outputs:
            if var not in variables and var is not None:
                variables += (var,)
    return variables

class ResidualSet(ProjactableSet):
    def __init__(self, components=None) -> None:
        super().__init__()
        self.components = components if components is not None else list()
        self.variables = unique_components_variables(self.components)
    
    def project(self, projected_variable=None):
        # inverted_components = [component.invert() for component in self.components]
        F = Functional(projectable=self, projected=projected_variable)
        F.solver = DefaultResidualSolver(F)
        return F

    def merge(self, other):
        return ResidualSet(self.components + other.components)
    
class EliminationSet(ResidualSet):
    def __init__(self, components, eliminate=None) -> None:
        self.components = components
        self.eliminate = eliminate
        eliminate_variables = eliminate.independent
        self.variables = tuple((var for var in 
                               unique_components_variables(self.components)
                               if var not in eliminate_variables)) 
    
    def project(self, projected_variable=None):
        projected_variable = projected_variable if projected_variable is not None else tuple()
        projected_variable += tuple((elt for elt in self.eliminate.projected 
                                     if elt not in self.variables))
        F = Functional(projectable=self, projected=projected_variable)
        F.solver = EliminationSolver(F, [self.eliminate])
        return F
    
class EliminationKeepSet(ResidualSet):
    def __init__(self, components, eliminate=None) -> None:
        all_components = components+eliminate.projectable.components
        super().__init__(all_components)
        self.elimination_functional = EliminationSet(components, eliminate).project()
    
    def project(self, projected_variable=None):
        F = Functional(projectable=self, projected=projected_variable)
        F.solver = FeedForwardSolver([self.elimination_functional, 
                                      self.elimination_functional.projectable.eliminate])
        return F