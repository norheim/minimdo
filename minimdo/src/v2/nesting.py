from enum import Enum
from anytree import NodeMixin, PreOrderIter
from src.v1.inputresolver import eqvars, default_out, default_in
from src.v1.execution import eqvar

NodeTypes = Enum('NodeTypes', 'INTER END SOLVER OPT')
INTER, END, SOLVER, OPT = NodeTypes.INTER, NodeTypes.END, NodeTypes.SOLVER, NodeTypes.OPT

class RefNode(NodeMixin):
    def __init__(self, name, ref=None, node_type=None, parent=None, children=None):
         self.name = name
         self.ref = ref
         self.parent = parent
         self.node_type = node_type
         if children:  # set children only if given
             self.children = children

    def __repr__(self):
        return '<{}>'.format(str(self.name))

    def __str__(self):
        return str(self.name)

class Function():
    def __init__(self, name, original_node_type):
        self.name = name
        self.original_node_type = original_node_type

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return str(self.name)

class SolverNode(NodeMixin):
    def __init__(self, name, parent=None, refonly=False, children=None, node_type=SOLVER):
        self.ref = parent.ref
        self.name = name
        self.parent = None if refonly else parent
        self.node_type = node_type
        if children:  # set children only if given
             self.children = children

    def setsolvefor(self, solveforvars):
        # hacky solution with enumerate
        for idx, elt in enumerate(self.children):
            if elt.node_type == END:
                self.ref.outset[elt] = solveforvars[idx]

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return str(self.name)

class Model():
    def __init__(self):
        self.eqs = dict()
        self.tree = list() #might not be needed
        self.outset = dict()

    def data_structures(self):
        eqs = self.eqs
        eqv = eqvars(eqs)
        dout = default_out(eqs)
        dins = default_in(eqs)
        # Hacky solution needs fixing
        dins_clean = [elt for elt in dins if elt not in self.outset.values()]
        return eqs, eqv, dout, dins_clean


def adda(branch_node, left, right, pretty_name=False, returnfx=False, returnnode=False, *args, **kwargs):
    if isinstance(left, str):
        out, eq = eqvar(left, right, *args, **kwargs)
        outsetvar = out
    else:
        out, eq = None, (left, right)
        outsetvar = left
    m = branch_node.ref
    name_template = 'f_{{{}}}' if pretty_name else '{}'
    fname = name_template.format(len(m.eqs))
    function = Function(fname, original_node_type=INTER)
    tree_node = RefNode(fname, ref=function, node_type=INTER, parent=branch_node)
    m.outset[function] = outsetvar
    m.eqs[function] = eq
    m.tree.append(tree_node)
    if returnfx and returnnode:
        return out, function, tree_node
    elif returnfx:
        return out, function
    elif returnnode:
        return out, tree_node
    else:
        return out

def addf(branch_node, eq, name=None, nameasidx=False):
    m = branch_node.ref
    fname = name if name else (str(len(m.eqs)) if nameasidx 
                               else 'r_{{{}}}'.format(len(m.eqs)))
    function = Function(fname, original_node_type=END)
    tree_node = RefNode(fname, ref=function, node_type=END, parent=branch_node)
    m.eqs[function] = ((None, eq))
    m.tree.append(tree_node)
    return tree_node

def geteqs(root_node, outvars):
    m = root_node.ref
    tree_nodes = [node for node in PreOrderIter(root_node) if 
        node.node_type==INTER]
    return [f for f in tree_nodes if m.outset[f.ref] in outvars]

def addsolver(branch_node, sequence=None, outset=None, name=None, node_type=SOLVER):
    sequence = sequence if sequence else []
    outset = outset if outset else {}
    name = name if name else '.'
    m = branch_node.ref
    solver_node = SolverNode(name, parent=branch_node, node_type=node_type)
    for node in sequence:
        node.parent = solver_node
    for fnode_end,solvevar in outset:
        m.outset[fnode_end.ref] = solvevar
        fnode_end.parent = solver_node
    return solver_node