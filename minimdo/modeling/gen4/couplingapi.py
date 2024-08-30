from collections import OrderedDict, defaultdict
from itertools import chain
import numpy as np
from modeling.gen2.execution import edges_from_components
from modeling.gen2.transformations import flatten_component
from modeling.gen3.nesting import addequation, calculateval
from modeling.gen4.ipoptsolver import setup_ipopt
from graph.workflow import OPT, NEQ, EQ, OBJ, SOLVE
from graph.graphutils import VAR, COMP, SOLVER
from solver.runpipeline import nestedform_to_mdao

def find_indices(list1, list2):
    index_dict = {value: index for index, value in enumerate(list1)}
    return [index_dict.get(value, -1) for value in list2]

def get_index_ranges(variables):
    index_ranges = []
    start_idx = 0
    for var in variables:
        size = np.prod(var.shape)
        end_idx = start_idx + size
        index_ranges.append((start_idx, end_idx))
        start_idx = end_idx
    return index_ranges

def subset_index_ranges(all_variables, selected_subset):
    all_index_ranges = get_index_ranges(all_variables)
    subset_index_ranges = [all_index_ranges[all_variables.index(var)] for var in selected_subset]
    return subset_index_ranges

def select_subset(flat_vector, subset_index_ranges):
    result = []
    for start_idx, end_idx in subset_index_ranges:
        result.append(flat_vector[start_idx:end_idx])
    return np.concatenate(result)

def get_precomputed_info(variables):
    precomputed = []
    start_idx = 0
    for var in variables:
        size = np.prod(var.shape)
        end_idx = start_idx + size
        precomputed.append((start_idx, end_idx, var.shape))
        start_idx = end_idx
    return precomputed
    
def split_vector(flat_vector, precomputed_info):
    split_arrays = []
    for start_idx, end_idx, shape in precomputed_info:
        split_array = flat_vector[start_idx:end_idx].reshape(shape)
        split_arrays.append(split_array)
    return split_arrays

def get_precomputed_indices(all_variables, selected_subset):
    all_info = get_precomputed_info(all_variables)
    selected_indices = []
    for var, (start_idx, end_idx, shape) in zip(all_variables, all_info):
        if var in selected_subset:
            selected_indices.append((start_idx, end_idx, shape))
    return selected_indices

def set_subset(flat_vector, precomputed_indices, input_arrays):
    modified_vector = flat_vector.copy()
    for (start_idx, end_idx, shape), input_array in zip(precomputed_indices, input_arrays):
        if np.isscalar(input_array) and np.prod(shape) == 1:
            modified_vector[start_idx:end_idx] = input_array
        elif not np.isscalar(input_array) and input_array.shape == shape:
            modified_vector[start_idx:end_idx] = input_array.flatten()
    return modified_vector

def convert(val, asarray=True):
    return np.array(val) if asarray else val

def get_vals(P, independent_only=True, asarray=True):
    vars_to_get = P.independent if independent_only else chain(P.independent, P.projected)
    output = {var: convert(P.prob.get_val(str(var)), asarray) for var in vars_to_get}
    return output
    
class Subproblem():
    def __init__(self, name=None, problemid=None) -> None:
        self.name = name
        self.id = 1 if problemid is None else problemid
        self.variable_order = list() # these are all the variables
        self.projected = list() # these are the non zero projected variables
        self.independent = list() # these are the independent variables that we are solving for
        self.inputs = list()
        self.components = []
        # Formulation tree
        self.Ftree = OrderedDict()
        self.Stree = dict()
        self.Vtree = dict()
        # Solver specifics
        self.solver_options = dict()
        self.comp_options = None
        # MDAO related variables
        self.prob = None
        self.mdao_in = None

    # Needed for using the subproblem as a dictionary key
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __hash__(self):
        return hash(tuple(hash(component) for component in self.components))

    def add_component(self, component, parent_id=None):
        self.components.append(component)
        self.projected.extend([vr for vr in component.mapped_inputs if (vr not in self.projected) and (vr not in self.independent)])
        self.independent.extend([vr for vr in component.mapped_outputs if (vr is not None) and (vr not in self.independent)])
        self.Ftree[component.id] = self.id if parent_id is None else parent_id

    def add_equation(self, left, right, *args, **kwargs):
        leftvar, comp_isnew, eqcomp = addequation(self.components, right, left, *args, **kwargs)
        if comp_isnew:
            self.add_component(eqcomp)
        if leftvar is None: # this is for residual components
            return eqcomp.id
        return leftvar

    def Var(self, left, right, *args, **kwargs):
        left = self.add_equation(left, right, *args, **kwargs)
        return left

    def setup_ipopt(self, y):
        self.ipopt = setup_ipopt(self.components, 
                                 self.independent, y)
   
    def solve_with_ipopt(self, y):
        self.setup_ipopt(y)
        yout = np.copy(y)
        x0 = select_subset(y, subset_index_ranges(self.projection_order, self.independent))
        out = self.ipopt.solve(x0)
        xstar = out[0]
        xaddress = get_precomputed_indices(self.inputs, self.inputs)
        yaddress = get_precomputed_indices(self.projection_order, self.inputs)
        for (xstart,xend),(ystart,yend) in zip(xaddress,yaddress):
                yout[ystart:yend] = xstar[xstart:xend]
        return yout


    def setup(self, engine='openMDAO', mdf=True, solver='G'):
        edges = edges_from_components(self.components)
        tree = self.Ftree, self.Stree, self.Vtree
        solver_options = {self.id: {'solver':solver}} if not self.solver_options else self.solver_options
        comp_options = {} if not self.comp_options else self.comp_options
        var_options = {}
        nametyperepr = {VAR: '{}', COMP: 'f{}', SOLVER: 's{}'}
        prob, mdao_in, groups, (ordered_edges, ordered_tree), merge_order = nestedform_to_mdao(edges, tree, self.components, solver_options, comp_options, var_options, nametyperepr, mdf)
        self.mdao_in = mdao_in
        self.prob = prob

    def solve_with_engine(self, vardict=None, save_projected=False, optimize=False, asarray=False):
        input_dict = {var:var.varval for var in self.projected}
        full_output_dict = {var:var.varval for var in self.independent}
        local_dict = {**input_dict, **full_output_dict}
        if vardict != None:
            local_dict.update(vardict)
            if save_projected:
                for key,val in local_dict.items():
                    key.varval = val
        for elt,val in local_dict.items():
            self.prob[str(elt)]= val
        if optimize:
            self.prob.run_driver()
        else:
            self.prob.run_model()
        output = get_vals(self)
        return output

    def solve(self, vardict=None, lookup_projected=True, save_projected=False):
        input_dict = {var:var.varval for var in self.projected}
        full_output_dict = {var:var.varval for var in self.independent}
        local_dict = {**input_dict, **full_output_dict}
        if vardict is not None:
            local_dict.update(vardict)
            if save_projected:
                for key,val in local_dict.items():
                    key.varval = val
        for comp in self.components:
            z = comp.evaldict(local_dict)
            output_dict = {var:np.array(varval) for var,varval in zip(comp.mapped_outputs, z)}
            local_dict.update(output_dict)
        return {var:np.array(local_dict[var]) for var in self.independent}
            
def intersection(*subproblems, counter=0, **kwargs):
    mergesets = kwargs.pop('mergesets', False)
    # First we create mappings from old ids to new ids
    subproblem_counter = 0
    subproblem_id_mappings = defaultdict(dict)
    for subproblem in subproblems:
        for key in chain(subproblem.Stree.keys(), (subproblem.id,)):
            subproblem_id_mappings[subproblem][key] = subproblem_counter
            subproblem_counter+=1
    merged_subproblem = Subproblem(problemid=subproblem_counter)
    # Then we apply all the mappings
    for subproblem in subproblems:
        mapped_id = subproblem_id_mappings[subproblem] 
        for component in subproblem.components:
            old_parent_id = mapped_id[subproblem.Ftree[component.id]] if not mergesets else merged_subproblem.id
            if not mergesets:
                copy_component = component.copy(counter)
            else:
                copy_component = flatten_component(component, counter)
            merged_subproblem.add_component(copy_component, parent_id=old_parent_id)
            counter+=1
        if not mergesets:
            merged_subproblem.Stree[mapped_id[key]] = subproblem_counter
            merged_subproblem.Stree.update({mapped_id[key]:mapped_id[val] for key,val in subproblem.Stree.items()})
    return merged_subproblem

def problem(objective_function, constraint_functions, discipline_set, flatten=False, counter=0):
    coupled_problem = intersection(*discipline_set, counter=counter, mergesets=flatten)
    comp_options = {}
    if flatten:
        for component in coupled_problem.components:
            comp_options[component.id] = EQ
    objective_comp_id = coupled_problem.add_equation(None, objective_function)
    comp_options[objective_comp_id]=OBJ
    for constraint_function in constraint_functions:
        constraint_comp_id = coupled_problem.add_equation(None, constraint_function)
        comp_options.update({constraint_comp_id: NEQ})
    solver_options = {coupled_problem.id: {"type":OPT}}
    coupled_problem.comp_options = comp_options
    coupled_problem.solver_options = solver_options
    return coupled_problem
