from enum import Enum
from collections import namedtuple, defaultdict
from functools import partial
from itertools import chain, product
from typing_extensions import OrderedDict
import networkx as nx
from representations import draw
from utils import normalize_name

NodeTypes = Enum('NodeTypes', 'VAR COMP SOLVER')
NodeTypes.__repr__ = lambda x: x.name
VAR, COMP, SOLVER = NodeTypes.VAR,NodeTypes.COMP,NodeTypes.SOLVER
default_nodetyperepr = {VAR: 'x_{}', COMP: 'f_{}', SOLVER: 's_{}'}

def namefromid(nodetyperepr):
    def nameingfunction(eltids, elttype, isiter=False):
        if isiter:
            return tuple(nodetyperepr[elttype].format(eltid) for eltid in eltids)
        else:
            return nodetyperepr[elttype].format(eltids)
    return nameingfunction

def namevar(eltid, elttype, nodetyperepr):
    if elttype == VAR:
        return normalize_name(eltid, keep_underscore=True)
    else:
        return nodetyperepr[elttype].format(eltid)

def namefromsympy(nodetyperepr): 
    def nameingfunction(eltids, elttype, isiter=False):
        if isiter:
            return tuple(namevar(eltid, elttype, nodetyperepr) for eltid in eltids)
        else:
            return namevar(eltids, elttype, nodetyperepr)
    return nameingfunction

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

def copy_dicts(dicts):
    return tuple(d.copy() for d in dicts)

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

def all_solvers(tree):
    Ftree, Stree, _ = tree
    return set(Stree.values()).union(set(Ftree.values()))

def solver_children(tree, solver_idx, solverlist=False):
    solver_idx = solver_idx if solverlist else [solver_idx]
    return (comp for comp,parent_solver in tree.items() if parent_solver in solver_idx)

def default_tree(idxs, solver_idx=1):
    return OrderedDict((key,1) for key in idxs),{},{}

def flat_graph_formulation(Ein, Eout, Rin, nodetyperepr=None, raw=False):
    transform_fx = None if raw else partial(transform_E, tfx=lambda x: Node(x, COMP, nodetyperepr), tvar=lambda x: Node(x, VAR, nodetyperepr)) 
    edges = all_edges(merge_edges(Ein,Rin), Eout, transform_fx)
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
    Ftree,Stree,Vtree=trees
    descendants = dfs_tree(Stree, branch)
    inputs_strictly_below = solver_children(Vtree, descendants-{branch}, solverlist=True)
    comps_below = solver_children(Ftree, descendants-{branch}, solverlist=True)
    srcs = sources(Ein, Eout, filterto=comps_below)
    return srcs - set(inputs_strictly_below)

def path(Stree, s, visited=None):
    visited = visited if visited else set()
    out = []
    if s in chain(Stree.values(),Stree.keys()):
        q = {s}  
    else:
        q = set()
        out = [s] if s not in visited else [] # we should at least return the parent node
    while q:
        s = q.pop()
        if s not in visited:
            out.append(s)
            if s in Stree:
                q.add(Stree[s])
        visited.add(s)
    return out

def root_solver(tree):
    Ftree, Stree, _ = tree
    spath = path(Stree, next(iter(Ftree.values())))
    root = spath[-1] # last element
    return root

def upstream(edges, comp):
    # TODO: verify that it works with cyclic components?
    Ein, Eout = edges_to_Ein_Eout(edges)
    Eoutrev = defaultdict(list)
    for c, vs in Eout.items():
        for v in vs:
            Eoutrev[v].append(c)
    q = [(comp,True)]
    v = []
    while q:
        c+=1
        node, is_comp = q.pop()
        E = Ein if is_comp else Eoutrev
        children = [elt for elt in product(E[node], [not is_comp]) if elt not in v]
        q += children
        v += children
    return set(key for key,val in v if not val) 

# def rearrange_edges(edges, output_set):
#     Ein, Eout, Rin = copy_dicts(edges)
#     Ein_new, Eout_new, Rin_new = {}, {}, Rin
#     for comp in Eout:
#         outputvar = output_set.get(comp, None)
#         reversedvar = None
#         if outputvar is None and Eout[comp][0] is not None:
#             reversedvar = Eout[comp]
#             Rin_new[comp] = (reversedvar,)
#         Eout_new[comp] = (outputvar,)
#         Ein_new[comp] = tuple(varn for varn in chain(Ein[comp],Eout[comp]) if varn not in [outputvar, reversedvar])