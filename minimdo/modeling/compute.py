from typing import Reversible
import re
import sympy as sp
from sympy.core.cache import clear_cache
import numpy as np
from autograd import grad
import autograd.numpy as anp
import openmdao.api as om
from pint import UnitRegistry
from trash.inputresolver import reassigneq, eqvars, default_out, default_in
from itertools import count
from anytree import Node, NodeMixin, PreOrderIter
from enum import Enum
from modeling.arghandling import Encoding

ureg = UnitRegistry()
NodeTypes = Enum('NodeTypes', 'INTER END SOLVER OPT')
INTER = NodeTypes.INTER
END = NodeTypes.END
SOLVER = NodeTypes.SOLVER # to be used for inter but when a solver
OPT = NodeTypes.OPT

# The function will work for sympy symbols or just plain strings
def get_latex(symbol_or_string):
    return symbol_or_string if symbol_or_string else r'\mathrm{{{}}}'.format(symbol_or_string)

def prettyprintval(x, latex=False, unit=None, rounding=None):
    rounding = rounding if rounding is not None else 3
    sci_expr = '{'+':.{}e~P'.format(rounding)+'}' if not latex else '{'+':.{}e~L'.format(rounding)+'}'
    nonsci_expr = '{'+':.{}f'.format(rounding)+'}'
    if x == None:
        return None
    elif (x>1e4 or x<1e-3) and x!=0:
        x_removed_brackets = np.squeeze(x)[()]
        return sci_expr.format(ureg.Quantity(x_removed_brackets, unit))
    else:
        unit_expr = '\ {:L~}' if latex else ' {}'
        unitstr = prettyprintunit(unit, unit_expr, latex) if unit else ''
        return r'{}{}'.format(nonsci_expr.format(x).rstrip('0').rstrip('.'),unitstr)

def prettyprintunit(x, strformat='{:P~}', latex=False):
    if x.units != ureg('year'):
        return strformat.format(x.units)
    else:
        return r'\ \mathrm{yr}' if latex else 'yr'

def get_assumed_string(assumed):
    return (r'{}={}'.format(get_latex(key),prettyprintval(val,latex=True)) for key,val in assumed.items())

def remove_frac_from_latex(latexstr):
    return re.sub(r'\\frac{(.*)}{(.*)}', r'\1/\2', latexstr)

class Var(sp.core.Symbol):
    def __new__(cls, name, value=None, unit=None, always_input=False, varid=None, forceunit=False):
        #clear_cache()  # sympys built in cache can cause unexpected bugs
        out = super().__new__(cls, name) #positive=positive)
        out.always_input = always_input
        out.varval = value
        out.varunit = ureg(unit) if unit else ureg('')
        out.forceunit = forceunit
        out.varid = varid if varid != None else name
        out.assumed = dict() # sometimes to generate varval we need to assume values
                             # for the function that computed this value 
        out.shape = None
        return out
    
    def custom_latex_repr(self):
        if self.varval != None:
            assumed = ''
            if self.assumed:
                assumed = '\ ({} )'.format(' ,'.join(get_assumed_string(self.assumed)))
            if self.varunit.dimensionless:
                varstr = prettyprintval(self.varval, latex=True)+' '
            else:
                varstr = prettyprintval(self.varval, latex=True, unit=self.varunit)
                # remove frac's for more compact notation
                varstr = remove_frac_from_latex(varstr)
            # Need to synchronize the name of 'dummy' here and in api
            namestr = '{}='.format(self.name) if self.name != 'dummy' else ''
            return '{}{}{}'.format(namestr, varstr, assumed)
        else:
            return self.name

    def _repr_latex_(self):
        return '${}$'.format(self.custom_latex_repr())

def create_vars(string, split=' '):
    return [Var(elt) for elt in string.split(split)]

def var_encoding(*vars):
    return Encoding(order=vars, shapes=
                    tuple(var.shape if var.shape is not None 
                          else (1,) for var in vars))

class Par(Var):
    _ids = count(0)
    def __new__(self, *args, **kwargs):
        if (len(args) >= 1 and not isinstance(args[0], str)) or len(args) == 0:
            args = ('p{}'.format(next(self._ids)),)+args
        out = Var.__new__(self, *args, **kwargs)
        out.always_input = True
        return out

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

def eqvar(name, right, unit=None, forceunit=False):
    newvar = Var(name, unit=unit)
    newvar.forceunit=forceunit # TODO: HACK for sympy function
    if not forceunit:
        rhs_unit = get_unit(right)
        if unit != None:
            assert ureg(unit).dimensionality == rhs_unit.dimensionality
    if unit == None:
        newvar.varunit = ureg.Quantity(1, rhs_unit.to_base_units().units)
    return newvar, (newvar, right)

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


# TODO: OLD STUFF, should delete but kept for maintainability reasons

class Expcomp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('equation')
        self.options.declare('output_name')
        self.options.declare('debug')

    def setup(self):
        equation = self.options['equation']
        output_name = self.options['output_name']
        self.add_output(output_name)
        for name in equation.input_names:
            self.add_input(name, val=1.) # add them in the order we lambdify
        self.declare_partials(output_name, equation.input_names)
            
    def compute(self, inputs, outputs):
        equation = self.options['equation']
        output_name = self.options['output_name']
        debug = self.options['debug']
        outputs[output_name] = equation.evaldict(inputs)
        if debug:
            print(output_name, outputs[output_name])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        equation = self.options['equation']
        output_name = self.options['output_name']
        J = equation.graddict(inputs)
        for idx, input_name in enumerate(equation.input_names):
            partials[output_name,input_name] = J[idx]

class Impcomp(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('equation')
        self.options.declare('output_name')
        
    def setup(self):
        equation = self.options['equation']
        output_name = self.options['output_name']
        self.add_output(output_name)
        original_inputs = [inp for inp in equation.input_names if inp != output_name]
        for name in original_inputs:
            self.add_input(name, val=1.) # add them in the order we lambdify
        self.declare_partials(output_name, equation.input_names)

    def apply_nonlinear(self, inputs, outputs, residuals):
        equation = self.options['equation']
        output_name = self.options['output_name']
        residuals[output_name] = equation.evaldict({**inputs, **outputs})
        
    def linearize(self, inputs, outputs, partials):
        equation = self.options['equation']
        output_name = self.options['output_name']
        J = equation.graddict({**inputs, **outputs})
        for idx, input_name in enumerate(equation.input_names):
            partials[output_name, input_name] = J[idx]


# recursive function
def coupled_run(eqs, seq_order, solve_order, parent, root, outset=None, 
    counter=0,  debug=False, useresiduals=False, equationcreator=None, maxiter=20):
    if equationcreator == None:
        equationcreator = Evaluable.fromsympy
    counter+=1
    group = parent.add_subsystem('group{}'.format(counter), 
        om.Group(), promotes=['*'])
    order = []
    if solve_order:
        order = solve_order
        group.linear_solver = om.DirectSolver()
        nlbgs = group.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        #nlbgs.linesearch = om.BoundsEnforceLS()
        nlbgs.options['maxiter'] = maxiter
        if seq_order:
            counter = coupled_run(eqs, seq_order, (), group, root, outset,  
                counter, debug)
    else:
        order = seq_order
        useresiduals=False
    for idx, eqnelt in enumerate(order):
        if isinstance(eqnelt, list):
            counter = coupled_run(eqs, eqnelt, (), group, root, outset, counter, debug)
        elif isinstance(eqnelt, tuple):
            if isinstance(eqnelt[0], list):
                ordered = eqnelt[0]
                unordered = eqnelt[1:]
            else:
                ordered = []
                unordered = eqnelt
            counter = coupled_run(eqs, ordered, unordered, group, root, 
                outset, counter, debug, useresiduals)
        else:
            eqn = eqnelt
            left, right = eqs[eqn]
            if useresiduals:
                    parent.add_subsystem("eq{}".format(eqn), Expcomp(
                        output_name='r{}'.format(eqn),
                        equation=equationcreator(right-left),
                        debug=debug), 
                        promotes=['*'])
                    root.add_constraint('r{}'.format(eqn), equals=0.)
            else:
                addsolver = False
                if debug:
                    print('eq{}'.format(eqn), left, right, outset.get(eqn) if outset else None)
                if outset and outset.get(eqn)!=left:
                    comp = Impcomp(output_name=str(outset.get(eqn)), 
                    equation=equationcreator(right-left))
                    addsolver = True
                else:
                    comp = Expcomp(output_name=str(left), 
                    equation=equationcreator(right, left), debug=debug)
                subs = group.add_subsystem("eq{}".format(eqn), comp, 
                    promotes=['*'])
                if addsolver:
                    subs.linear_solver = om.DirectSolver()
                    subs.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    return counter

def buildidpvars(inputs, model):
    comp = om.IndepVarComp()
    np.random.seed(5)
    for elt in inputs:
        val = elt.varval if (hasattr(elt, 'varval') 
            and elt.varval != None) else np.random.rand()
        comp.add_output(str(elt), val)
    model.add_subsystem('inp', comp, promotes=['*'])