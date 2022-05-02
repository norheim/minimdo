import numpy as np
import openmdao.api as om
from graphutils import root_solver, nested_sources
from workflow import EXPL, IMPL, SOLVE, OPT, EQ, NEQ, OBJ
from executionblocks import addexpcomp, addimpcomp, addsolver, addoptimizer, addoptfunc

architecture_mappings = {
        EXPL: addexpcomp,
        IMPL: addimpcomp,
        SOLVE: addsolver,
        OPT: addoptimizer,
        EQ: addoptfunc,
        NEQ: addoptfunc,
        OBJ: addoptfunc
    }

def buildidpvars(inputs, model):
    comp = om.IndepVarComp()
    np.random.seed(5)
    for elt in inputs:
        val = elt.varval if (hasattr(elt, 'varval') 
            and elt.varval != None) else np.random.rand()
        comp.add_output(str(elt), val)
    model.add_subsystem('inp', comp, promotes=['*'])

def build_archi(edges, tree, workflow):

    # Build MDO model
    prob = om.Problem()
    mdo_model = prob.model
    groups = {'prob': prob, None:mdo_model}
    root = root_solver(tree)
    mdao_in = nested_sources(edges, tree, root)
    buildidpvars(mdao_in, mdo_model)
    
    # This next step actually builds all the openMDAO components
    for comp_type, *comp_args in workflow:
        args = [comp_type] + comp_args if comp_type in [NEQ, EQ, OBJ] else comp_args
        architecture_mappings[comp_type](groups, *args)
    
    #mdo_model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.setup();
    return prob, mdao_in, groups