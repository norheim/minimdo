import networkx as nx
from collections import OrderedDict
import matplotlib.pyplot as plt

def draw(g, pos=None, edge_color='k', width=2, arc=False, figsize=(6,6), 
        prog='neato'):
    labels = OrderedDict([(elt,r'${}$'.format(str(elt))) for elt in g.nodes])
    if pos is None:
        pos = nx.drawing.nx_pydot.graphviz_layout(g, prog='neato')
    plt.figure(figsize=figsize)
    plt.margins(0.15)
    nx.draw_networkx_nodes(g, pos=pos, node_size=700, node_color='w', 
        linewidths=2, edgecolors='k')
    #https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
    connectionstyle = 'arc3, rad = 0.1' if arc else None
    nx.draw_networkx_edges(g, pos=pos, arrowsize=20, width=width, 
        edge_color=edge_color, connectionstyle=connectionstyle)
    nx.draw_networkx_labels(g, pos, labels, font_size=20);
    plt.gca().axis("off");

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

def digraph_repr(eqvars, default_output):
    edges = [(inp, default_output[eq]) for eq, inps in eqvars.items() 
        for inp in inps if inp != default_output[eq]]
    return nx.DiGraph(edges), edges