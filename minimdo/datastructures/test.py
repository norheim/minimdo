from graphutils import draw_graph_graphical_props, flat_graph_formulation, Node, VAR, COMP, SOLVER
from nestedgraph import build_typedgraph
from compute import Var
from api import Model, adda, addf, addsolver, setsolvefor
from execution import edges_from_components
from operators import sort_scc, reorder_merge_solve
from nestedgraph import build_typedgraph
from sympy import exp

z1,z2,x,y2 = Var('z1'), Var('z2'), Var('x'), Var('y2')

model = Model()
m = model.root
s = addsolver(m)
y1 = adda(s, 'y1', z1**2+z2+x-0.2*y2)
adda(s, y2, y1**0.5+z1+z2)
addf(s, x**2+z1+y1+exp(-y2))
addf(s, 3.16-y1)
addf(s, y2-24)
setsolvefor(s, [x,z1,z2])

edges = edges_from_components(model.components)
tree = model.Ftree, model.Stree, model.Vtree

nodetyperepr = {VAR: '{}', COMP: 'f_{}', SOLVER: 's_{}'}

graphs = build_typedgraph(edges, tree, nodetyperepr)