from utils.polycasebuilder import generate_random_polynomials, get_arg_mapping, directed_poly_executables, eqv_to_edges_tree
from utils.randomstructure import generate_random_prob
from graph.graphutils import VAR, COMP, SOLVER, edges_to_Ein_Eout, sinks, sources, namefromid, all_components
from graph.workflow import OBJ, NEQ, EQ, OPT
from src.v2.runpipeline import nestedform_to_mdao
import numpy as np
from src.v2.execution import Component
from graph.operators import sort_scc, reorder_merge_solve
from graph.graphview import bipartite_repr
from src.v1.inputresolver import getallvars, direct_eqs, invert_edges
from presolver.tearing import min_max_scc, outset_from_solution
from src.v1.symbolic import Var

nodetyperepr = {VAR: 'x_{{{}}}', COMP: 'f_{}', SOLVER: 's_{}'}
#namefunc = namefromid(nodetyperepr)

n_eqs = 5
n_vars = 10
seed = 3#8 is triang#seed 10 is nice 42
sparsity = 1.7#0.8 1.1 #1.7 1.3
eqv, varinc, output_set = generate_random_prob(n_eqs, n_vars, seed, sparsity)
polynomials, var_mapping, edges, tree, components = generate_random_polynomials(eqv, output_set, n_eqs)

_, edges_original = bipartite_repr(eqv)
eqs=direct_eqs(eqv, output_set)
avrs = getallvars(eqs, sympy=False)
eqns = eqs.keys()
graph_edges = invert_edges(edges_original)

maxl, m = min_max_scc(graph_edges, avrs, eqns, len(eqns))

output_set2 = dict(outset_from_solution(m))
output_set2 = {key:var-n_eqs for key,var in output_set2.items()}
components2 = directed_poly_executables(var_mapping, polynomials, output_set2)
edges2, tree2 = eqv_to_edges_tree(eqv, output_set2, n_eqs)
inputvars = [var_mapping[elt][0] for elt in sources(*edges_to_Ein_Eout(edges2))]
fobj = sum([(elt-1)**2 for elt in inputvars])
newidx = max(all_components(edges2[0]))+1
c = Component.fromsympy(fobj, component=newidx)
edges2[0][newidx] = c.inputs
edges2[1][newidx] = c.outputs
tree2[0][newidx] = 1
components2 = components2+[c]

solvers_options = {1: {'type': OPT}, 2:{'maxiter':100}}
comp_options = {newidx:OBJ}
var_options = {}

mdaotyperepr = {VAR: 'x{}', COMP: 'f{}', SOLVER: 's{}'}
prob, mdao_in, groups = nestedform_to_mdao(edges2, tree2, components2, solvers_options, comp_options, var_options, mdaotyperepr, mdf=True)

prob.run_driver()