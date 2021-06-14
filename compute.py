import sympy as sp
import numpy as np
from autograd import grad
import openmdao.api as om

def args_in_order(name_dict, names):
    return [name_dict[in_var] for in_var in names]

class Equation():
    def __init__(self, left, right):
        inputs = list(right.free_symbols)
        self.fx = sp.lambdify(inputs, right, 'numpy')
        wrapped_fx = self.fx if len(inputs) == 1 else (
                lambda x: self.fx(*x)) #adapt to numpy
        self.jfx = grad(wrapped_fx)
        self.output_name = str(left)
        self.inputs_names = list(map(str, inputs))
        
    def evaldict(self, indict):
        return self.fx(*args_in_order(indict, self.inputs_names))
    
    def graddict(self, indict):
        args = np.array(args_in_order(indict, self.inputs_names))
        return self.jfx(args)

class Expcomp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('equation')
        
    def setup(self):
        equation = self.options['equation']
        self.add_output(equation.output_name)
        for name in equation.inputs_names:
            self.add_input(name, val=1.) # add them in the order we lambdify
        self.declare_partials(equation.output_name, equation.inputs_names)
            
    def compute(self, inputs, outputs):
        equation = self.options['equation']
        outputs[equation.output_name] = equation.evaldict(inputs)
        #print(outputs[equation.output_name])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        equation = self.options['equation']
        J = equation.graddict(inputs)
        for idx, input_name in enumerate(equation.inputs_names):
            partials[equation.output_name,input_name] = J[idx]


def coupled_run(eqs, seq_order, solve_order, parent, root, counter, 
    useresiduals=False):
    counter+=1
    group = parent.add_subsystem('group{}'.format(counter), 
        om.Group(), promotes=['*'])
    order = []
    if solve_order:
        order = solve_order
        group.linear_solver = om.DirectSolver()
        nlbgs = group.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        nlbgs.options['maxiter'] = 20
        if seq_order:
            counter = coupled_run(eqs, seq_order, (), group, root, counter, 
                useresiduals)
    else:
        order = seq_order
    for idx, eqnelt in enumerate(order):
        if isinstance(eqnelt, list):
            counter = coupled_run(eqs, eqnelt, (), group, root, counter)
        elif isinstance(eqnelt, tuple):
            if isinstance(eqnelt[0], list):
                ordered = eqnelt[0]
                unordered = eqnelt[1:]
            else:
                ordered = []
                unordered = eqnelt
            counter = coupled_run(eqs, ordered, unordered, group, root, counter)
        else:
            eqn = eqnelt
            left, right = eqs[eqn]
            if useresiduals:
                    parent.add_subsystem("eq{}".format(eqn), Expcomp(
                        equation=Equation('r{}'.format(eqn), right-left)), 
                        promotes=['*'])
                    #root.add_constraint('r{}'.format(eqn), equals=0.)
            else:
                group.add_subsystem("eq{}".format(eqn), Expcomp(
                    equation=Equation(left, right)), promotes=['*'])
    return counter

