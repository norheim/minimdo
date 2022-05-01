from enum import Enum
from collections import namedtuple, defaultdict
from functools import partial
import networkx as nx
from representations import draw

NodeTypes = Enum('NodeTypes', 'VAR COMP SOLVER')
VAR, COMP, SOLVER = NodeTypes.VAR,NodeTypes.COMP,NodeTypes.SOLVER
default_nodetyperepr = {VAR: 'x_{}', COMP: 'f_{}', SOLVER: 's_{}'}
class Node():
    def __init__(self, name, nodetype, nodetyperepr=None):
        self.nodetyperepr = nodetyperepr if nodetyperepr else default_nodetyperepr
        self.nodetype = nodetype
        self.name = name

    def __repr__(self):
        return self.nodetyperepr[self.nodetype].format(self.name)

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.name, self.nodetype))

def make_nodetype(elts, nodetype=COMP):
    return tuple(Node(node, nodetype) for node in elts)

def filter_comps(E, filterto=None):
    return ((key,val) for key,val in E.items() if not filterto or key in filterto)

def transform_E(E, tfx=None, tvar=None):
    tfx = tfx if tfx else lambda fx: fx
    tvar = tvar if tvar else lambda var: var
    return {tfx(fx): tuple(tvar(var) if var!=None else None for var in vrs) for fx, vrs in E.items()} 

def edges_E(E, reverse=True, transform=None, filterto=None):
    E = dict(filter_comps(E, filterto))
    E = transform(E) if transform else E
    return [(var,fx) if reverse else (fx,var) for fx,vrs in E.items() for var in vrs if var]

def all_edges(Ein, Eout, transform=None, filterto=None):
    return edges_E(Ein, True, transform, filterto)+edges_E(Eout, False, transform, filterto)

def merge_edges(Ein, Rin):
    return {key: var+Rin.get(key, tuple()) for key,var in Ein.items()}

def edges_to_Ein_Eout(edges):
    Ein, Eout, Rin = edges
    return merge_edges(Ein, Rin), Eout

def all_components(E):
    return set(E.keys())

def end_components(E):
    return [key for key,var in E.items() if None in var]

def all_varnodes(E, filterto=None):
    return {var for key,val in filter_comps(E, filterto) for var in val}

def all_variables(Ein, Eout, filterto=None):
    return all_varnodes(Ein, filterto).union(all_varnodes(Eout, filterto)-{None})

def sources(Ein, Eout, filterto=None):
    return all_varnodes(Ein, filterto)-all_varnodes(Eout, filterto)

def sinks(Ein, Eout, filterto=None):
    return all_varnodes(Eout, filterto)-all_varnodes(Ein, filterto)

def intermediary_variables(Ein, Eout, filterto=None):
    return all_varnodes(Eout, filterto).intersection(all_varnodes(Ein,filterto))

def solver_children(tree, solver_idx, solverlist=False):
    solver_idx = solver_idx if solverlist else [solver_idx]
    return (comp for comp,parent_solver in tree.items() if parent_solver in solver_idx)

def flat_graph_formulation(Ein, Eout, Rin, nodetyperepr=None):
    edges = all_edges(merge_edges(Ein,Rin), Eout, partial(transform_E, tfx=lambda x: Node(x, COMP, nodetyperepr), tvar=lambda x: Node(x, VAR, nodetyperepr)))
    G = nx.DiGraph(edges)
    return G

def draw_graph_graphical_props(G, colormap=None, defaultcolor='w', **kwargs):
    var_names = [elt for elt in G.nodes() if elt.nodetype==VAR]
    node_shapes = {elt:'o' if elt in var_names else 's' for elt in G.nodes()}
    if colormap:
        colormap_rev = {vr:key for key,var in colormap.items() for vr in var}
        node_colors = {elt:colormap_rev.get(elt, defaultcolor) for elt in G.nodes()}
    else:
        node_colors = 'w'
    draw(G, node_shape=node_shapes, node_color=node_colors, latexlabels=False, **kwargs);

def dfs_tree(tree, branch):
    childrenmap = defaultdict(list)
    for key,val in tree.items():
        childrenmap[val].append(key)
    visited = set()
    q = [branch]
    while q:
        elt = q.pop()
        visited.add(elt)
        q.extend(childrenmap.get(elt, []))
    return visited

def nested_sources(edges, trees, branch):
    Ein,Eout = edges_to_Ein_Eout(edges)
    Ftree,_,Vtree=trees
    descendants = dfs_tree(Vtree, branch)
    inputs_strictly_below = solver_children(Vtree, descendants-{branch}, solverlist=True)
    comps_below = solver_children(Ftree, descendants-{branch}, solverlist=True)
    srcs = sources(Ein, Eout, filterto=comps_below)
    return srcs - set(inputs_strictly_below)
