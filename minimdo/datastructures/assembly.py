import numpy as np
import openmdao.api as om
from graphutils import Node, VAR, sources, merge_edges
from workflow import EXPL, IMPL, SOLVE
from executionblocks import addexpcomp, addimpcomp, addsolver

architecture_mappings = {
        EXPL: addexpcomp,
        IMPL: addimpcomp,
        SOLVE: addsolver
    }

def buildidpvars(inputs, model):
    comp = om.IndepVarComp()
    np.random.seed(5)
    for elt in inputs:
        val = elt.varval if (hasattr(elt, 'varval') 
            and elt.varval != None) else np.random.rand()
        comp.add_output(str(elt), val)
    model.add_subsystem('inp', comp, promotes=['*'])

def build_archi(edges, tree, workflow, transform_inputs=True):
    Ftree, Stree, Vtree = tree
    Ein, Eout, Rin = edges
    Ein = merge_edges(Ein, Rin)
    # Build MDO model
    prob = om.Problem()
    mdo_model = prob.model
    groups = {None:mdo_model}
    mdao_in = {str(Node(elt, VAR)) if transform_inputs else elt for elt in sources(Ein, Eout)-Vtree.keys()}
    buildidpvars(mdao_in, mdo_model)
    
    # This next step actually builds all the openMDAO components
    for comp_type, *comp_args in workflow:
        #print(comp_type, comp_args[:3])
        architecture_mappings[comp_type](groups, *comp_args)
    
    #mdo_model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.setup();
    return prob, mdao_in, groups