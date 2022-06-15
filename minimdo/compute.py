from typing import Reversible
import re
import sympy as sp
from sympy.core.cache import clear_cache
import numpy as np
from autograd import grad
import autograd.numpy as anp
import openmdao.api as om
from pint import UnitRegistry
from inputresolver import reassigneq, idx_eqlist, eqvars, default_out, default_in
from itertools import count
from anytree import Node, NodeMixin, PreOrderIter
from enum import Enum
ureg = UnitRegistry()
NodeTypes = Enum('NodeTypes', 'INTER END SOLVER OPT')
INTER = NodeTypes.INTER
END = NodeTypes.END
SOLVER = NodeTypes.SOLVER # to be used for inter but when a solver
OPT = NodeTypes.OPT

def args_in_order(name_dict, names):
    return [name_dict[in_var] for in_var in names if in_var in name_dict]

# The function will work for sympy symbols or just plain strings
def get_latex(symbol_or_string):
    return symbol_or_string.custom_latex_repr() if symbol_or_string else r'\mathrm{{{}}}'.format(symbol_or_string)

def get_assumed_string(assumed):
    return (r'{}={}'.format(get_latex(key),val) for key,val in assumed.items())

def remove_frac_from_latex(latexstr):
    return re.sub(r'\\frac{(.*)}{(.*)}', r'\1/\2', latexstr)
class Var(sp.core.Symbol):
    def __new__(cls, name, value=None, unit=None, always_input=False, varid=None):
        #clear_cache()  # sympys built in cache can cause unexpected bugs
        out = super().__new__(cls, name) #positive=positive)
        out.always_input = always_input
        out.varval = value
        out.varunit = ureg(unit) if unit else ureg('')
        out.forceunit = False
        out.varid = varid if varid != None else name
        out.assumed = dict() # sometimes to generate varval we need to assume values
                             # for the function that computed this value 
        return out
    
    def custom_latex_repr(self):
        if self.varval:
            assumed = ''
            if self.assumed:
                assumed = '\ ({})'.format(','.join(get_assumed_string(self.assumed)))
            if self.varunit.dimensionless:
                varstring = self.varval
            else:
                quantity = self.varval*self.varunit
                varstring = '{:L~}'.format(quantity)
                # remove frac's for more compact notation
                varstring = remove_frac_from_latex(varstring)
            return '{}={}{}'.format(self.name, varstring, assumed)
        else:
            return self.name

    def _repr_latex_(self):
        return '${}$'.format(self.custom_latex_repr())

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

# The following class emulates being a dictionary for sympys lambdify to work
# with autograd
math_functions = ['cos', 'sin', 'tan', 'arccos', 'arcsin', 'arctan', 'sqrt', 
'exp', 'log', 'log2', 'log10']

anp_math = {elt: getattr(anp, elt) for elt in math_functions}

# The following class is a very hacky class that is used further down to recover the unit associated with a specific function. It overrides all standard operators in python
class MockFloat(float):
    def __new__(self, value):
        return float.__new__(self, value)
    def __init__(self, value):
        float.__init__(value)
    def __add__(self, other):
        return self
    def __sub__(self, other):
        return self
    def __mul__(self, other):
        return self
    def __truediv__(self, other):
        return self
    def __floordiv__(self, other):
        return self
    def __mod__(self, other):
        return self
    def __pow__(self, other):
        return self
    def __rshift__(self, other):
        return self
    def __lshift__(self, other):
        return self
    def __and__(self, other):
        return self
    def __or__(self, other):
        return self
    def __xor__(self, other):
        return self

def get_unit(expr):
    if isinstance(expr, Var):
        return expr.varunit
    else:
        free_symbols = list(expr.free_symbols)
        if free_symbols:
            fx = sp.lambdify(free_symbols, expr, np)
            args = (ureg.Quantity(MockFloat(1), free_symbol.varunit) for free_symbol in free_symbols)
            dim = fx(*args)
            # need this case rhs_unit is a float, which can happen when we have powers, e.g. 10^x
            if not isinstance(dim, ureg.Quantity):
                dim = ureg('')
            return dim
        else:
            return ureg('') #most likely we encountered a number

def get_unit_multiplier(unit):
    return unit.to_base_units().magnitude

def unit_conversion_factors(right, orig_unit, symb_order):
    unit = orig_unit if orig_unit else ureg('')
    rhs_unit = get_unit(right)
    convert = np.array([get_unit_multiplier(free_symbol.varunit) for 
            free_symbol in symb_order])
    if orig_unit:
        assert(unit.dimensionality == rhs_unit.dimensionality)
        conversion_unit = unit
    else: # unitstr was not given
        if not hasattr(rhs_unit, 'units'):
            conversion_unit = ureg('')
        else:
            conversion_unit = ureg.Quantity(1, 
                rhs_unit.to_base_units().units)

    factor = get_unit_multiplier(conversion_unit)
    return convert, factor

def evaluable_with_unit(right, symb_order, tovar=None):
    convert, factor = unit_conversion_factors(right, tovar, symb_order)
    def correction(fx):
        def rhsfx_corrected(*args):
            return fx(*(convert*np.array(args).flatten()))/factor
        return rhsfx_corrected

    return correction

def fill_args(args, input_names, attrcheck='varval'):
    fxargs = []
    idx = 0
    for elt in input_names:
        if getattr(elt, attrcheck):
            fxargs.append(elt.varval)
        else:
            fxargs.append(args[idx])
            idx+=1
    return fxargs

def partialfx(fx, input_names):
    def wrapper(*args, **kwargs):
        partial = kwargs.get('partial', None)
        if partial:
            return fx(*fill_args(args, input_names, partial))
        else:
            return fx(*args)
    return wrapper

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

def addf(branch_node, eq, name=None):
    m = branch_node.ref
    fname = name if name else 'r_{{{}}}'.format(len(m.eqs))
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

# Should be sympy agnostic
class Evaluable():
    def __init__(self, fx, input_names=None):
        input_names = input_names if input_names else fx.__code__.co_varnames
        self.input_vars = input_names
        self.input_names = list(map(str,input_names))
        self.fx = partialfx(fx, input_names)
        wrapped_fx = fx if len(input_names) == 1 else (
                lambda x: fx(*x)) #adapt to numpy
        self.jfx = grad(wrapped_fx)
        self.njfx = lambda *args: self.jfx(np.array(args).astype(float))

    @classmethod
    def fromsympy(cls, expr, tovar=None):
        input_names = list(expr.free_symbols)
        fx = sp.lambdify(input_names, expr, anp_math)
        if tovar and not isinstance(type(expr), sp.core.function.UndefinedFunction):# and hasattr(expr, 'dimensionality'): 
            # this second conditions can be dangerous but need it to fix something
            unitify = evaluable_with_unit(expr, input_names, tovar.varunit) 
            # this is to get the right multiplier, any untis checks will have been done during creation? 
            # TODO: check this
            fx = unitify(fx)
        return cls(fx, input_names)
    
    def evaldict(self, indict):
        return self.fx(*args_in_order(indict, self.input_names))
    
    def graddict(self, indict):
        args = np.array(args_in_order(indict, self.input_names))
        return self.jfx(args)


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