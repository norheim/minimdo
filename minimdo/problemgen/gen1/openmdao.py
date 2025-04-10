import openmdao.api as om
import numpy as np
from modeling.gen1.compute import Evaluable
from trash.inputresolver import getallvars

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

# This is a utility function to get the outputs after running an openmdao model
def get_outputs(eqs, model, varasstring=False):
    vrs = getallvars(eqs)
    return {(str(elt) if varasstring else elt): model.get_val(str(elt))[0] for elt in vrs}