from copy import deepcopy
from anytree import Node,findall
import networkx as nx
from src.v2.nesting import END, INTER

def merge_pure(G, mergelts, mergeinto, mergewithold=True, solvefor=None):
    #Graph
    solvefor = [] if solvefor == None else solvefor
    allsolvevars = set()
    for elt in mergelts:
        if elt.node_type == END and elt.ref in solvefor:
            allsolvevars.add(solvefor[elt.ref])
    edges = G.edges()
    mergededges = [(fr,to) for fr,to in edges if fr in mergelts or to in mergelts]
    outs = {to for fr,to in edges if fr in mergelts and not all([elt in mergelts for elt in G.successors(to)])}
    outs = outs.union(allsolvevars) # for every terminal node can have one solvefor
    ins = {fr for fr,to in edges if (
        to in mergelts 
        and not any([elt in mergelts for elt in G.predecessors(fr)]) 
        and fr not in allsolvevars)}
    newedges = [(fr, mergeinto) for fr in ins]+[(mergeinto, to) for to in outs]
    if mergewithold:
        oldedges = [(fr, to) for fr,to in edges if fr not in mergelts and to not in mergelts]
        newedges += oldedges
    # Tree
    newG = nx.DiGraph(newedges)
    childG = nx.DiGraph(mergededges)
    return newG, childG

def standardize(elt):
    if elt.children: #node_type = SOLVER or None
        out = []
        for node in elt.children:
            if node.node_type != END:
                st_nodes = standardize(node)
            else:
                st_nodes = [node]
            out.extend(st_nodes)
        return out
    else:
        assert elt.node_type == INTER
        elt.node_type= END
        return [elt]

def merge(G, treeroot, mergelts, mgroupname, mergewithold=True, solvefor=None, solvefortable=None):
    #Graph
    solvefor = [] if solvefor == None else solvefor
    solvefortable = dict() if solvefortable == None else solvefortable
    allsolvevars = set()
    for solvevar, res in solvefor:
        mergelts.append(res)
        allsolvevars.add(solvevar)
    edges = G.edges()
    mergededges = [(fr,to) for fr,to in edges if fr in mergelts or to in mergelts]
    outs = {to for fr,to in edges if fr in mergelts and not all([elt in mergelts for elt in G.successors(to)])}
    outs = outs.union(allsolvevars) # for every terminal node can have one solvefor
    ins = {fr for fr,to in edges if (
        to in mergelts 
        and not any([elt in mergelts for elt in G.predecessors(fr)]) 
        and fr not in allsolvevars)}
    newedges = [(fr, mgroupname) for fr in ins]+[(mgroupname, to) for to in outs]
    if mergewithold:
        oldedges = [(fr, to) for fr,to in edges if fr not in mergelts and to not in mergelts]
        newedges += oldedges
    # Tree
    treeroot = deepcopy(treeroot)
    solvefortable = solvefortable.copy()
    solvefortable[mgroupname] = solvefor
    newsolver = Node(mgroupname, parent=treeroot)
    for elt in findall(treeroot, filter_=lambda node: node.name in mergelts):
        elt.parent = newsolver
    return nx.DiGraph(newedges), nx.DiGraph(mergededges), treeroot, solvefortable