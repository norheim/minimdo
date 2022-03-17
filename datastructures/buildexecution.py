import openmdao.api as om

class Impcomp(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('components')
         
    def setup(self):
        components = self.options['components']
        forbidden_input = {output_name for _, output_name, _, _, _ in components} # variables have to be strictly separated into inputs and outputs so if there is any coupling between multiple components(inputs in one, outputs in the other) they cannot be classified as input
        for input_names, output_name, _, _, guess_vars in components:
            self.add_output(output_name)
            for input_name in input_names:
                # make sure not to add the same input twice
                if input_name not in forbidden_input:
                    forbidden_input.add(input_name) 
                    self.add_input(input_name, val=guess_vars)
            self.declare_partials(output_name, input_names)

    def apply_nonlinear(self, inputs, outputs, residuals):
        components = self.options['components']
        for _, output_name, fx, _, _ in components:
            residuals[output_name] = fx({**inputs, **outputs})
        
    def linearize(self, inputs, outputs, partials):
        components = self.options['components']
        for input_names, output_name, _, gradfx, _ in components:
            J = gradfx({**inputs, **outputs})
            for idx, input_name in enumerate(input_names):
                partials[output_name, input_name] = J[idx]

class Expcomp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('component')
        self.options.declare('debug')

    def setup(self):
        input_names, output_name, _, _ = self.options['component']
        self.add_output(output_name)
        for name in input_names:
            self.add_input(name, val=1.) # add them in the order we lambdify
        self.declare_partials(output_name, input_names)
            
    def compute(self, inputs, outputs):
        _, output_name, fx, _ = self.options['component']
        debug = self.options['debug']
        outputs[output_name] = fx(inputs)
        if debug:
            print(output_name, outputs[output_name])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        input_names, output_name, _, gradfx = self.options['component']
        J = gradfx(inputs)
        for idx, input_name in enumerate(input_names):
            partials[output_name,input_name] = J[idx]

def addsolver(mdao, parent_name, solver_name, **kwargs):
    parent = mdao[parent_name]
    child = mdao[solver_name] = parent.add_subsystem(solver_name, 
        om.Group(), promotes=['*'])
    child.linear_solver = om.DirectSolver()
    nlbgs = child.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    for key,var in kwargs.items():
        nlbgs.options[key] = var

def addimpcomp(mdao, parent_name, component_name, impl_components):
    parent = mdao[parent_name]
    implicit_component = Impcomp(components=impl_components)
    parent.add_subsystem(component_name, implicit_component, promotes=['*'])

def addexpcomp(mdao, parent_name, component_name, input_names, output_name, fx, gradfx, debug=False):
    parent = mdao[parent_name]
    explicit_component = Expcomp(component=(input_names, output_name, fx, gradfx), debug=debug)
    parent.add_subsystem(component_name, explicit_component, promotes=['*'])