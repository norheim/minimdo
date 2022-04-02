import networkx as nx
from representations import digraph_repr
from compute import addsolver, SOLVER, INTER, END, RefNode, SolverNode
from anytree import LevelOrderIter, PreOrderIter
from operatorsold import merge_pure, standardize

def sortf(e):
    idx, fx = e
    return (fx.node_type.value, idx)

def seq_by_type(sequence):
    return  [node for elt in sequence for idx, node in sorted(list(enumerate(elt.children)), key=sortf) if node.node_type in [INTER, END]]

def default_sequence(root, enforce_separation=False):
    if enforce_separation:
        return seq_by_type([elt for elt in PreOrderIter(root) if elt.node_type not in [INTER, END]])
    else:
        return [elt for elt in PreOrderIter(root) if elt.node_type in [INTER, END]]

def solvers_bottom_up(m, from_root=False):
    out = [node for node in LevelOrderIter(m) if 
        node.node_type==SOLVER][::-1]
    if from_root:
        out.append(m)
    return out

#make graph:
def graphs_from_incidence(branch_node, from_root=False):
    m = branch_node.ref
    _,sparsity,dout,_ = m.data_structures()
    node_list = [tree_node for tree_node in PreOrderIter(branch_node) if tree_node.node_type in [INTER, END]]
    sparsity_treeref = {tree_node: sparsity[tree_node.ref] for tree_node in node_list}
    dout_treeref  = {tree_node: dout[tree_node.ref] for tree_node in node_list}
    G, _ = digraph_repr(sparsity_treeref, dout_treeref, intermediary=True)
    merge_order = solvers_bottom_up(branch_node, from_root)
    newG = G
    graphs = dict()
    for elt in merge_order:
        newG, childG = merge_pure(newG, elt.children, elt, solvefor=m.outset)
        graphs[elt] = childG
    return G, graphs

# make DAG
def sort_scc(G, eqsn):
    C = nx.condensation(G)
    order = []
    for n in nx.topological_sort(C):
        filtereqs = {elt for elt in C.nodes[n]['members'] if elt in eqsn}
        if filtereqs:
            order.append(filtereqs)
    return order

def duplicate_nodes(branch_node, root_name='/', copy_solvers=True):
    model = branch_node.ref
    m_new = RefNode(root_name, ref=model)
    new_nodes = {elt: RefNode(elt.name, ref=elt.ref, node_type=elt.node_type) if elt.node_type !=SOLVER else SolverNode(elt.name, m_new, refonly=True) for elt in PreOrderIter(branch_node) 
    if elt != branch_node and (elt.node_type != SOLVER or copy_solvers)}
    # the second condition filters out solver nodes when copy_solvers=False
    new_nodes[branch_node] = m_new
    return new_nodes, m_new

def all_residuals(branch_node, from_root=True):
    new_nodes, m_new = duplicate_nodes(branch_node, copy_solvers=False)
    add_to_solver = []
    for node in new_nodes.values():
        if node!=m_new:
            node.parent = m_new
            node.node_type = END
            add_to_solver.append(node)
    addsolver(m_new, add_to_solver)
    return m_new
    

def make_acyclic(branch_node, graphs, method='scc', mdf=True, from_root=True):
    # Copy tree nodes
    new_nodes, m_new = duplicate_nodes(branch_node)
    # Handle solver nodes in reverse order
    merge_order = solvers_bottom_up(branch_node, from_root)
    for solver_branch in merge_order:
        scc = sort_scc(graphs[solver_branch], [elt for elt in solver_branch.children])
        order = []
        for cc in scc:
            if len(cc) == 1:
                node = next(iter(cc))
                new_nodes[node].parent = new_nodes[solver_branch]
            else:
                stcc = []
                for node in cc:
                    stcc += standardize(new_nodes[node])
                    if node.node_type==SOLVER: #we "delete' the solver node
                        new_nodes[node].parent = None
                if mdf:
                    order += [addsolver(new_nodes[solver_branch], stcc)]
                else:
                    for node in stcc:
                        node.parent = new_nodes[solver_branch]
    return m_new

def generate_execution(ex):
    q = [(0, elt) for elt in ex]
    totcounter = 0
    order = []
    while q:
        count, out = q.pop(0)
        if isinstance(out, tuple):
            vr, eq = out
            if isinstance(eq, list):
                totcounter +=1
                nontpl = [elt for elt in eq if not isinstance(elt, tuple)]
                order.append(('solver', count, totcounter, vr, nontpl))
                count = totcounter
            else:
                order.append(('exp', count, vr, eq))
        else:
            eq = out
            #order.append(('res', count, eq))
        if isinstance(eq, list):
            tpl = [elt for elt in eq if isinstance(elt, tuple)]
            q = [(count, elt) for elt in eq]+q
    return order

addgroup = lambda parent, child, solvevars, res: 'solver, {} child {} solver for {}, res={}'.format(parent, child, solvevars,res)
addexp = lambda parent, var, eq: 'add exp {} <- {} to group {}'.format(var, eq, parent)
addres = lambda parent, eq: 'add res {} to group {}'.format(eq, parent)

def printer(fx):
    def printingfx(*args):
        print(fx(*args))
    return printingfx

mapping = {'solver':printer(addgroup), 'exp':printer(addexp), 'res':printer(addres)}