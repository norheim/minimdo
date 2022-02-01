import networkx as nx
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sympy.printing as printing
from anytree import RenderTree, PreOrderIter
from inputresolver import getallvars
from datastructures import all_vars_from_incidence
from compute import INTER, END

def generate_label(elt, latexlabel=True):
    if latexlabel:
        return r'${}$'.format(elt if isinstance(elt, str) else printing.latex(elt))
    else:
        return r'${}$'.format(str(elt))

def draw(g, pos=None, edge_color='k', width=2, arc=None, figsize=(6,6), 
        prog='neato', node_size=700, node_shape='s', latexlabels=True, **kwargs):
    node_actual_shape = node_shape
    node_color = 'w'
    linewidths=2
    label_kwargs = {
        'font_size': 20
    }
    if node_shape == 'b':
        node_actual_shape = 's'
        node_color = 'none'
        linewidths=0
        label_kwargs['bbox'] = dict(facecolor="white", edgecolor='black', boxstyle='round,pad=0.2')

    
    labels = OrderedDict([(elt,generate_label(elt, latexlabel=latexlabels)) for elt in g.nodes])
    if pos is None:
        pos = nx.drawing.nx_pydot.graphviz_layout(g, prog=prog, **kwargs)
    fig = plt.figure(figsize=figsize)
    plt.margins(0.15)
    if isinstance(node_shape, dict):
        for node, shape in node_shape.items():
            nx.draw_networkx_nodes(g, pos=pos, node_size=node_size, node_color=node_color, linewidths=linewidths, edgecolors='k', node_shape=shape, nodelist=[node])
    else:
        nx.draw_networkx_nodes(g, pos=pos, node_size=node_size, node_color=node_color, linewidths=linewidths, edgecolors='k', node_shape=node_actual_shape)
    #https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
    connectionstyle = 'arc3, rad = {}'.format(arc) if arc else 'arc3'
    nx.draw_networkx_edges(g, pos=pos, arrowsize=20, width=width, 
        edge_color=edge_color, node_size=node_size, connectionstyle=connectionstyle)
    nx.draw_networkx_labels(g, pos, labels, **label_kwargs)
    plt.gca().axis("off")
    return fig, plt.gca()

def circular_vars(graph, eqs):
    return {elt:'o' if elt in getallvars(eqs) else 's' for elt in graph.nodes()}

def drawfull(graph, eqs, prog='neato', figsize=(6,6)):
    draw(graph, node_shape=circular_vars(graph, eqs), latexlabels=False, arc=0.1, prog=prog, figsize=figsize);

def drawbipartite(g, left_nodes=None, M=None):
    if left_nodes is None:
        left_nodes,_ = nx.bipartite.sets(g)
    pos = nx.drawing.layout.bipartite_layout(g, left_nodes)
    edge_color = ['royalblue' if n1 in M and M[n1]==n2 else 'k' 
        for n1, n2 in g.edges()] if M else None
    edge_width = [4 if n1 in M and M[n1]==n2 else 2 
        for n1, n2 in g.edges()] if M else None
    draw(g, pos, edge_color, edge_width, figsize=(4,5))

def bipartite_repr(eqvars):
    edges = [(inp, eq) for eq, inps in eqvars.items() for inp in inps]
    return nx.Graph(edges), edges

def digraph_repr(eqvars, default_output, intermediary=False):
    edges = [(inp, (eq if intermediary else default_output[eq])) for eq, inps in eqvars.items() for inp in inps if inp != default_output[eq]]
    if intermediary:
        # end components might have none has their output
        edges += [(key,val) for key,val in default_output.items() if val != None ]
    return nx.DiGraph(edges), edges


def draw_dsm(DG, order_flattened, dout, fontsize=14, addvar=False):
    m = nx.adjacency_matrix(DG)
    eqorder = order_flattened #2,1,3,4
    latexpattern = lambda elt: r'$x_{{{}}}$'.format(elt) if addvar else r'${}$'.format(printing.latex(elt))
    labels = OrderedDict([(elt,latexpattern(elt)) for elt in DG.nodes])
    dsm_labels = [labels[dout[elt]] for elt in eqorder]
    node_order = list(DG.nodes())
    index_order = [node_order.index(dout[elt]) for elt in eqorder]
    #https://stackoverflow.com/questions/22927181/selecting-specific-rows-and-columns-from-numpy-array
    dsm = m.todense()[np.ix_(index_order,index_order)].astype(np.float32)
    dsm[np.diag_indices(len(dsm))]=0.5
    #plt.matshow(dsm.T, cmap='Greys',  interpolation='nearest');
    fig = plt.pcolormesh(dsm.T, cmap='Greys', edgecolors='lightgray', linewidth=1, vmin=0, vmax=1.1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    a,b = zip(*enumerate(dsm_labels))
    plt.xticks(np.array(a)+0.5,b, rotation = 60, fontsize=fontsize)
    plt.yticks(np.array(a)+0.5,b, fontsize=fontsize)
    ax.xaxis.tick_top()
    return ax

def getpatchwindow(lst):
    ulcorner = lst[0]
    size = max(lst)-ulcorner+1
    return ulcorner,size

def tree_incidence(root, incstr, solvefor, sequence, permutation=None, figsize=None):
    allpatches = []
    for node in PreOrderIter(root):
        if not node.is_leaf: #to speed things up a bit
            ls = [(idx,elt) for idx,elt in enumerate(sequence) if elt in PreOrderIter(node)]
            inter = [idx for idx, elt in ls if 
                    (elt.node_type==INTER and elt in node.children) or 
                    ((elt not in node.children) and node in elt.ancestors)]
            end = [idx for idx,elt in ls if elt.node_type==END and elt in node.children]
            patchparam = getpatchwindow([idx for idx,elt in ls])
            patchparam_inter = [getpatchwindow(inter)] if inter else []
            patchparam_end = [getpatchwindow(end)] if end else []
            allpatches.append(patchparam)
            allpatches += patchparam_inter
            allpatches += patchparam_end
    sequence_based_permutation = [solvefor[eqname.ref] for eqname in sequence]  # option that shows outputs
    if permutation==None:
        permutation = list(all_vars_from_incidence(incstr))
    permutation = sequence_based_permutation + [var for var in permutation if var not in sequence_based_permutation]
    A = np.zeros((len(sequence), len(permutation)))
    for idx, fxnode in enumerate(sequence):
        varsineq = [elt for elt in incstr[fxnode.ref] if not elt.always_input]
        for var in varsineq:
            col = permutation.index(var)
            color = 0.5 if (idx == col and fxnode.node_type==INTER) else 1.
            A[idx,col] = color
    tree_labels = {node: (r"{}${}$".format(pre, node.name)) for pre, _, node in RenderTree(root)}
    fig = plt.figure(figsize=figsize) if figsize else None
    fig = plt.pcolormesh(A, cmap='Greys', edgecolors='lightgray', linewidth=1, vmin=0, vmax=1.1, figure=fig)
    fontsize=16
    permute_labels = [generate_label(elt) for elt in permutation]
    xtickidx, xtags = zip(*enumerate(permute_labels))
    plt.xticks(np.array(xtickidx)+0.5,xtags, rotation = 60, fontsize=fontsize)
    sequence_labels = [tree_labels[elt] for elt in sequence]
    ytickidx, ytags = zip(*enumerate(sequence_labels))
    ax = plt.gca()
    plt.yticks(np.array(ytickidx)+0.5,ytags, fontsize=fontsize, ha = 'left')
    ax.xaxis.tick_top()
    #neqs = len(sequence)
    for ulcorner, size in allpatches:
        rect = patches.Rectangle((ulcorner,ulcorner), size, size, linewidth=2, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    yax = ax.get_yaxis()
    yax.set_tick_params(pad=100)
    ax.set_aspect('equal');
    return fig, ax


def render_tree(root, display_type=False):
    for pre, _, node in RenderTree(root):
        node_type = ', {}'.format(node.node_type.name) if node.node_type and display_type else ''
        treestr = u"{}{}{}".format(pre, node.name, node_type)
        print(treestr.ljust(16))