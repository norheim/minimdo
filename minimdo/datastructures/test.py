from graphutils import draw_graph_graphical_props, flat_graph_formulation, Node, VAR, COMP, SOLVER, nested_sources, merge_edges
from nestedgraph import build_typedgraph
from compute import Var
from api import Model, adda, addf, addsolver, setsolvefor
from execution import edges_from_components, generate_components_and_residuals
from operators import sort_scc, reorder_merge_solve
from nestedgraph import build_typedgraph
from workflow import get_f, generate_workflow, OPT, EQ, NEQ, OBJ
from workflow import order_from_tree, default_solver_options, mdao_workflow, namefromid
from workflow_mdao import mdao_workflow_with_args
from sympy import exp
import openmdao.api as om
from assembly import buildidpvars, architecture_mappings

z1,z2,x,y2 = Var('z1'), Var('z2'), Var('x'), Var('y2')

model = Model()
m = model.root
a = adda(m, 'a', z2+x-0.2*y2)
y1 = adda(m, 'y1', z1**2+a)
adda(m, y2, y1**0.5+z1+z2)
addf(m, x**2+z1+y1+exp(-y2)) #addobj
addf(m, 3.16-y1) #addineq
addf(m, y2-24) #addineq
setsolvefor(m, [x,z1,z2])

edges = edges_from_components(model.components)
tree = model.Ftree, model.Stree, model.Vtree

G = flat_graph_formulation(*edges)
merge_order = sort_scc(G)
nedges, ntree = reorder_merge_solve(edges, tree, merge_order, 1, True)
components = generate_components_and_residuals(model, nedges)
lookup_f = get_f(components, nedges)
sequence = order_from_tree(ntree[0], ntree[1], nedges[1])
solvers_options = default_solver_options(ntree, {1: {'type': OPT}})
var_options = {'x': [0,10], 'z1': [0,10], 'z2': [0,10]}
wf = mdao_workflow(sequence, solvers_options, {3:OBJ, 4:NEQ, 5:NEQ}, var_options)
components = generate_components_and_residuals(model, nedges)
lookup_f = get_f(components, nedges)
namingfunc = namefromid({VAR: '{}', COMP: 'f{}', SOLVER: 's{}'})
wfmdao = mdao_workflow_with_args(wf, lookup_f, namingfunc)
desvars = nested_sources(nedges, ntree, 1)
Ftree, Stree, Vtree = ntree
Ein, Eout, Rin = nedges
Ein = merge_edges(Ein, Rin)
# Build MDO model
prob = om.Problem()
mdo_model = prob.model
groups = {None:mdo_model, 'prob':prob}
mdao_in = nested_sources(nedges, ntree, 1)
buildidpvars(mdao_in, mdo_model)
for comp_type, *comp_args in wfmdao:
    args = [comp_type] + comp_args if comp_type in [NEQ, EQ, OBJ] else comp_args
    architecture_mappings[comp_type](groups, *args)
prob.setup()
prob.set_val('x', 2.0)
prob.set_val('z1', 2.0)
prob.set_val('z2', 2.0)
#prob.run_model()
prob.run_driver()