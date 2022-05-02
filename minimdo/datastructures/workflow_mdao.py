from workflow import OPT, SOLVE, NEQ, EQ, OBJ, EXPL, IMPL
from graphutils import SOLVER, VAR, COMP
from utils import normalize_name
from execution import Component

def implicit_comp_name(comps):
    return "res_{}".format('_'.join(map(lambda x: normalize_name(str(x)),comps)))

#This is where name transformations happen
def optimargs_mdao(solvertype, solverid, parentid, options, varoptions, namefromid):
    solvername = namefromid(solverid, SOLVER)
    parentname = namefromid(parentid, SOLVER) if parentid else None
    design_vars = namefromid(options['designvars'], VAR, isiter=True) # and any transformation on their name
    if varoptions:
        varoptions = {namefromid(var, VAR):val for var,val in varoptions.items()}
    return (solvertype, parentname, solvername, design_vars, {key:var for key,var in options.items() if key !='designvars'}, varoptions)

def solveargs_mdao(solvertype, solverid, parentid, options, varoptions,namefromid):
    solvername = namefromid(solverid, SOLVER)
    parentname = namefromid(parentid, SOLVER) if parentid else None
    return (solvertype, parentname, solvername, {key:var for key,var in options.items() if key !='designvars'}, varoptions)

def implargs_mdao(functype, compids, designvars, parentid, complookup, namefromid):
    compnames = namefromid(compids, COMP, isiter=True)
    compname = implicit_comp_name(compnames)
    parentname = namefromid(parentid, SOLVER)
    impl_comps = []
    for idx, compid in enumerate(compids):
        comp = complookup(compid)
        inputs = namefromid(comp.inputs, VAR, isiter=True)
        output = namefromid(designvars[idx], VAR)
        impl_comps.append([inputs, (output,), comp.evaldict, comp.graddict, 1.0])
    return (functype, parentname, compname, impl_comps)

def explargs_mdao(functype, compid, parentid, complookup, namefromid):
    compname = namefromid(compid, COMP)
    parentname = namefromid(parentid, SOLVER)
    comp = complookup(compid)
    inputs, outputs = namefromid(comp.inputs, VAR, isiter=True), namefromid(comp.outputs, VAR, isiter=True)
    return (functype, parentname, compname, inputs, outputs)

def optfuncargs_mdao(functype, compid, parentid, complookup, namefromid):
    # need to define component name
    compname = namefromid(compid, COMP)
    parentname = namefromid(parentid, SOLVER)
    comp = complookup(compid)
    inputs = namefromid(comp.inputs, VAR, isiter=True)
    output = '{}{}'.format(repr(functype).lower(), compid)
    newcomp = Component(comp.function, inputs=inputs, outputs=(output,), component=comp.component, indims=comp.indims)
    return (functype, parentname, compname, inputs, output, newcomp.evaldict, newcomp.graddict) 

mapping = {
    OPT: optimargs_mdao,
    SOLVE: solveargs_mdao,
    IMPL: implargs_mdao,
    EXPL: explargs_mdao,
    OBJ: optfuncargs_mdao,
    EQ: optfuncargs_mdao,
    NEQ: optfuncargs_mdao
}

def mdao_workflow_with_args(workflow, complookup, namingfunc):
    workflow_mdao = []
    for elt in workflow:
        if elt[0] in [OPT, SOLVE]:
            workflow_mdao.append(mapping[elt[0]](*elt, namefromid=namingfunc))
        else:
            workflow_mdao.append(mapping[elt[0]](*elt, complookup=complookup, namefromid=namingfunc))
    return workflow_mdao