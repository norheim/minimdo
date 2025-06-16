import numpy as np
import openmdao.api as om
from src.v1.symbolic import buildidpvars
from src.v1.mdaobuild import explicit, solver

# FUNCTIONS TO MAKE THE ARCHITECTURE EXECUTABLE AND RUN IT
# TODO: remove all 
def architecture_mapping(groups, eqs, optin):
    return  {
        'exp': lambda *args: explicit(groups, eqs, *args),
        'solver': lambda *args: solver(groups, optin, *args, maxiter=200)
    }

def optimizer_data(eqs, objective_function=None, eqs_are_residual=False):
    # eqs_are_residual is to give a better residual function if we are using equality constraint in optimization
    optin_opt = {key: (None, val if eqs_are_residual else val[0]-val[1]) for key,val in eqs.items()}
    if objective_function:
        optin_opt[objective_function] = eqs[objective_function]
    return optin_opt

def build_archi(model, exec_instructions_pick, optin):
    eqs, eqv, dout, dins = model.data_structures()
    # Build MDO model
    prob = om.Problem()
    mdo_model = prob.model
    groups = {0:mdo_model}
    inst_mapping = architecture_mapping(groups, eqs, optin)
    
    mdao_in = [str(elt) for elt in dins]
    buildidpvars(mdao_in, mdo_model)
    
    # This next step actually builds all the openMDAO components
    for comp_type, *comp_args in exec_instructions_pick:
        inst_mapping[comp_type](*comp_args)
    
    prob.setup();
    return prob
# TODO: remove all end

def generate_x0(varnames=None, optres=None, rand_range=(10,20), fixed_direction=None, rng=None):
    rng = rng if rng else np.random
    direction = lambda : fixed_direction if fixed_direction else rng.choice([-1,1]) 
    varnames = optres.keys() if optres else varnames
    x0 = {name:(optres.get(name, None) if optres else 0.1)+rng.uniform(*rand_range)*direction() for name in varnames}
    return x0

def perturb_inputs(root_ins, root_rand_range, solver_ins, solver_rand_range, xref=None, rng=None):
    if xref:
        xref = dict(xref) # copy for immutability
        x0_solvers = {key:xref[key] for key in solver_ins} 
        x0_solvers = generate_x0(optres=x0_solvers, rand_range=solver_rand_range, rng=rng)
    else:
        xref = generate_x0(root_ins, rand_range=root_rand_range, rng=rng)
        x0_solvers = generate_x0(solver_ins, rand_range=solver_rand_range, rng=rng)
    xref.update(x0_solvers)
    return xref

def extractvals(prob, vrs, tfx=None):
    tfx = tfx if tfx else lambda x: x.name
    return {tfx(key): prob.get_val(tfx(key))[0] for key in 
          vrs}

def run_and_save_archi(prob, x0, vrs):
    for key,val in x0.items():
        prob.set_val(key,val)
    prob.run_model()
    optres = extractvals(prob, vrs.values())
    optres_save = extractvals(prob, vrs.values())
    return optres, optres_save

def perturb(val, rand_range=(10,20), fixed_direction=None, rng=None):
    rng = rng if rng else np.random
    direction = lambda : fixed_direction if fixed_direction != None else rng.choice([-1,1]) 
    if direction()==0:
        return rng.uniform(*rand_range)
    else:
        return val+rng.uniform(*rand_range)*direction()

def partial_perturb(x0, perturb_entries=None, rand_range=(10,20), fixed_direction=None, rng=None):
    if isinstance(x0, list):
        x0 = {val: 0 for val in x0}
    perturb_entries = perturb_entries if perturb_entries else []
    x_new = {key:val if key not in perturb_entries else perturb(val, rand_range, fixed_direction, rng) for key,val in x0.items()}
    return x_new