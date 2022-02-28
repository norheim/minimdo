import numpy as np
import openmdao.api as om
from compute import buildidpvars
from notationcompute import explicit, solver

# FUNCTIONS TO MAKE THE ARCHITECTURE EXECUTABLE AND RUN IT
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

def generate_x0(varnames=None, optres=None):
    varnames = optres.values() if optres else varnames
    x0 = {name:(optres.get(name, None) if optres else 0.1)+np.random.uniform(10,20)*np.random.randint(-1,1) for name in varnames}
    return x0

def extractvals(prob, vrs):
    return {key.name: prob.get_val(key.name)[0] for key in 
          vrs.values()}

def run_and_save_archi(prob, x0, vrs, optres_all):
    for key,val in x0.items():
        prob.set_val(key,val)
    prob.run_model()
    optres = extractvals(prob, vrs)
    optres_all.append(extractvals(prob, vrs)) # for immutability?
    return optres