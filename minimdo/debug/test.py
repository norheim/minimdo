from graphutils import draw_graph_graphical_props, flat_graph_formulation, Node, VAR, COMP, SOLVER, nested_sources, merge_edges
from nestedgraph import build_typedgraph, root_sources
from compute import Var
from api import Model, adda, addf, addsolver, setsolvefor, addobj, addineq
from execution import edges_from_components, generate_components_and_residuals
from operators import sort_scc, reorder_merge_solve
from nestedgraph import build_typedgraph
from workflow import get_f, OPT, EQ, NEQ, OBJ
from workflow import order_from_tree, default_solver_options, mdao_workflow
from workflow_mdao import mdao_workflow_with_args
from assembly import build_archi
from sympy import exp
import openmdao.api as om
from assembly import buildidpvars, architecture_mappings
from runpipeline import nestedform_to_mdao

z1,z2,x,y2 = Var('z1'), Var('z2'), Var('x'), Var('y2')
x0,x1,x2,x3 = Var('x0'), Var('x1'), Var('x2'), Var('x3')

# model = Model(solver=OPT)
# m = model.root
# x0 = adda(m, 'x0', (-2/3*x1*x3+1/3)/x2)
# addobj(m, (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2) #addobj
# setsolvefor(m, [x1,x2,x3], {x1:[0,10], x2:[0,10], x3:[0,10]})
# prob, mdao_in, groups = model.generate_mdao()

model = Model(solver=OPT)
m = model.root
a = adda(m, 'a', z2+x-0.2*y2)
y1 = adda(m, 'y1', z1**2+a)
adda(m, y2, y1**0.5+z1+z2)
addobj(m, x**2+z2+y1+exp(-y2)) #addobj
addineq(m, 3.16-y1) #addineq
addineq(m, y2-24) #addineq
setsolvefor(m, [x,z1,z2], {x:[0,10], z1:[0,10], z2:[0,10]})
prob, mdao_in, groups = model.generate_mdao()

# components = model.components
# edges = edges_from_components(components)
# tree = model.Ftree, model.Stree, model.Vtree
# solvers_options = {1: {'type': OPT}}
# comp_options = {3:OBJ, 4:NEQ, 5:NEQ}
# var_options = {'x': [0,10], 'z1': [0,10], 'z2': [0,10]}
# nametyperepr = {VAR: '{}', COMP: 'f{}', SOLVER: 's{}'}

# prob, mdao_in, groups = nestedform_to_mdao(edges, tree, components, solvers_options, comp_options, var_options, nametyperepr)

# namingfunc = namefromid(nametyperepr)
# G = flat_graph_formulation(*edges)
# merge_order = sort_scc(G)
# nedges, ntree = reorder_merge_solve(edges, tree, merge_order, 1, True)
# sequence = order_from_tree(ntree[0], ntree[1], nedges[1])
# solvers_options = default_solver_options(ntree, solvers_options)
# wf = mdao_workflow(sequence, solvers_options, comp_options, var_options)
# all_components = generate_components_and_residuals(components, nedges)
# lookup_f = get_f(all_components, nedges)
# wfmdao = mdao_workflow_with_args(wf, lookup_f, namingfunc)
# prob, mdao_in, groups = build_archi(nedges, ntree, wfmdao)

# prob.set_val('x', 1.0)
# prob.set_val('z1', 1.0)
# prob.set_val('z2', 1.0)
#prob.run_model()
prob.run_driver()