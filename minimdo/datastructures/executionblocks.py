import openmdao.api as om
from datastructures.workflow import OPT, OBJ, NEQ, EQ, SOLVE, IMPL, EXPL
class Impcomp(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('components')
         
    def setup(self):
        components = self.options['components']
        forbidden_input = {output_name[0] for _, output_name, _, _, _ in components} # variables have to be strictly separated into inputs and outputs so if there is any coupling between multiple components(inputs in one, outputs in the other) they cannot be classified as input
        for input_names, output_name, _, _, guess_vars in components:
            output_name = output_name[0]
            self.add_output(output_name, val=guess_vars) # this is what we are solving for
            for input_name in input_names:
                # make sure not to add the same input twice
                if input_name not in forbidden_input:
                    forbidden_input.add(input_name) 
                    self.add_input(input_name)
            self.declare_partials(output_name, input_names)

    def apply_nonlinear(self, inputs, outputs, residuals):
        components = self.options['components']
        for _, output_name, fx, _, _ in components:
            residuals[output_name[0]] = fx({**inputs, **outputs})
        
    def linearize(self, inputs, outputs, partials):
        components = self.options['components']
        for input_names, output_name, _, gradfx, _ in components:
            J = gradfx({**inputs, **outputs})
            for input_name in input_names:
                partials[output_name[0], input_name] = J[input_name]

class Expcomp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('component')
        self.options.declare('debug')

    def setup(self):
        input_names, output_names, _, _ = self.options['component']
        for output_name in output_names:
            self.add_output(output_name)
        for name in input_names:
            self.add_input(name, val=1.) # add them in the order we lambdify
        self.declare_partials(output_name, input_names)
            
    def compute(self, inputs, outputs):
        _, output_names, fx, _ = self.options['component']
        debug = self.options['debug']
        fxresults = fx(inputs)
        for output_name, fxresult in zip(output_names, fxresults):
            outputs[output_name] = fxresult
            if debug:
                print(output_name, outputs[output_name])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        _, _, _, gradfx = self.options['component']
        J = gradfx(inputs)
        for (outvar,invar),val in J.items():
            partials[outvar,invar] = val

def addoptimizer(mdao, parentname, solvername, design_vars, options, varoptions):
    root = mdao[parentname]
    prob = mdao['prob']
    mdao[solvername] = prob.model
    #root.add_subsystem(solvername, 
    #    om.Group(), promotes=['*'])
    for desvar in design_vars:
        if varoptions and desvar in varoptions:
            lb, ub = varoptions[desvar]
            root.add_design_var(desvar, lb, ub)
        else:
            root.add_design_var(desvar)
    prob.set_solver_print(level=1)
    prob.driver = om.ScipyOptimizeDriver(**options)
    # prob.driver.options['optimizer'] = 'differential_evolution'


def addoptfunc(mdao, functype, parentname, compname, inputs, output, fx, gradfx):
    root = mdao[parentname]
    addexpcomp(mdao, parentname, compname, inputs, (output,), fx, gradfx)
    if functype in [NEQ, EQ]:
        bnds = {'upper':0.} if functype==NEQ else {'upper':0., 'lower':0.}
        print("bounds", bnds)
        root.add_constraint(output, **bnds)
    elif functype == OBJ :
        root.add_objective(output)


def addsolver(mdao, parent_name, solver_name, kwargs, varoptions):
    parent = mdao[parent_name]
    child = mdao[solver_name] = parent.add_subsystem(solver_name, 
        om.Group(), promotes=['*'])
    solvertype = kwargs.pop('solver', 'N')
    if solvertype == 'N':
        child.linear_solver = om.DirectSolver()
        solver = om.NewtonSolver(solve_subsystems=True)
    elif solvertype == 'GS':
        solver = om.NonlinearBlockGS()
    elif solvertype == 'J':
        solver = om.NonlinearBlockJac()
    for key,var in kwargs.items():
        solver.options[key] = var
    child.nonlinear_solver = solver

def addimpcomp(mdao, parent_name, component_name, impl_components):
    parent = mdao[parent_name]
    implicit_component = Impcomp(components=impl_components)
    parent.add_subsystem(component_name, implicit_component, promotes=['*'])

def addexpcomp(mdao, parent_name, component_name, input_names, output_names, fx, gradfx, debug=False):
    parent = mdao[parent_name]
    explicit_component = Expcomp(component=(input_names, output_names, fx, gradfx), debug=debug)
    parent.add_subsystem(component_name, explicit_component, promotes=['*'])