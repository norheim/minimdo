import numpy as np
import openmdao.api as om
from datastructures.graphutils import root_solver, nested_sources, VAR, namefromid
from datastructures.nestedgraph import root_sources
from datastructures.workflow import EXPL, IMPL, SOLVE, OPT, EQ, NEQ, OBJ
from datastructures.executionblocks import addexpcomp, addimpcomp, addsolver, addoptimizer, addoptfunc

architecture_mappings = {
        EXPL: addexpcomp,
        IMPL: addimpcomp,
        SOLVE: addsolver,
        OPT: addoptimizer,
        EQ: addoptfunc,
        NEQ: addoptfunc,
        OBJ: addoptfunc
    }

def buildidpvars(mdao_in_ids, model, namingfunc, idmapping=None):
    comp = om.IndepVarComp()
    np.random.seed(5)
    for varid in mdao_in_ids:
        varsymb = idmapping[varid] if idmapping else varid
        varname = namingfunc(varid, VAR)
        val = varsymb.varval if (hasattr(varsymb, 'varval') 
            and varsymb.varval != None) else np.random.rand()
        comp.add_output(varname, val)
    model.add_subsystem('inp', comp, promotes=['*'])
    return 

def build_archi(edges, tree, workflow, namingfunc, idmapping=None, opt=True):
    # Build MDO model
    prob = om.Problem()
    mdo_model = prob.model
    groups = {'prob': prob, None:mdo_model}
    #root = root_solver(tree)
    mdao_in_ids = root_sources(edges, tree)
    if not opt:
        # if we are not optimizing, remove variables we are solving from inputs
        mdao_in_ids -= set(tree[2].keys())
    buildidpvars(mdao_in_ids, mdo_model, namingfunc, idmapping)
    
    # This next step actually builds all the openMDAO components
    for comp_type, *comp_args in workflow:
        args = [comp_type] + comp_args if comp_type in [NEQ, EQ, OBJ] else comp_args
        architecture_mappings[comp_type](groups, *args)
    
    #mdo_model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.setup();
    return prob, mdao_in_ids, groups