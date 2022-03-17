from enum import Enum
from collections import namedtuple
from functools import partial
import networkx as nx
from representations import draw

NodeTypes = Enum('NodeTypes', 'VAR COMP SOLVER')
VAR, COMP, SOLVER = NodeTypes.VAR,NodeTypes.COMP,NodeTypes.SOLVER

nodetyperepr = {VAR: 'x_{}', COMP: 'f_{}', SOLVER: 's_{}'}
Node = namedtuple('NODE', ['name', 'nodetype'])
Node.__repr__ = lambda x: nodetyperepr[x.nodetype].format(x.name)
Node.__str__ = Node.__repr__

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

def intermediary_variables(Ein, Eout):
    return all_varnodes(Eout).intersection(all_varnodes(Ein))

def solver_children(tree, solver_idx):
    return (comp for comp,parent_solver in tree.items() if parent_solver==solver_idx)

def flat_graph_formulation(Ein, Eout, Rin):
    edges = all_edges(merge_edges(Ein,Rin), Eout, partial(transform_E, tfx=lambda x: Node(x, COMP), tvar=lambda x: Node(x, VAR)))
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