import openmdao.api as om
from engine.gen1mdao.openmdao import Expcomp, Evaluable
from modeling.compute import normalize_name

class Impcomp(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('equations')
        self.options.declare('output_names')
        
    def setup(self):
        equations = self.options['equations']
        output_names = self.options['output_names']
        for output_name in output_names:
            self.add_output(output_name)
        original_inputs = {inp for equation in equations for inp in equation.input_names if inp not in output_names}
        for name in original_inputs:
            self.add_input(name, val=1.) # add them in the order we lambdify
        for equation, output_name in zip(equations, output_names):
            self.declare_partials(output_name, equation.input_names)

    def apply_nonlinear(self, inputs, outputs, residuals):
        equations = self.options['equations']
        output_names = self.options['output_names']
        for equation, output_name in zip(equations, output_names):
            residuals[output_name] = equation.evaldict({**inputs, **outputs})
        
    def linearize(self, inputs, outputs, partials):
        equations = self.options['equations']
        output_names = self.options['output_names']
        for equation, output_name in zip(equations, output_names):
            J = equation.graddict({**inputs, **outputs})
            for idx, input_name in enumerate(equation.input_names):
                partials[output_name, input_name] = J[idx]

def optsolver(groups, eqs, parentid, childid, solvevars, res):
    parent = groups[parentid]
    childid= 0
    child = groups[childid]# = parent.add_subsystem('group{}'.format(childid), 
        #om.Group(), promotes=['*'])
    for vr in solvevars:
        child.add_design_var(str(vr))
    ineqlts, eqlts, obj = res
    constraintnames = {
        'g{}'.format(childid): (ineqlts, {'upper':0.}),
        'h{}'.format(childid): (eqlts, {'upper':0., 'lower':0.}),
        'f{}'.format(childid): (obj, 'obj'),
    }
    for constraintname, (equations, bnd) in constraintnames.items():
        for eqn in equations:
            outname = constraintname+str(eqn)
            cmp = Expcomp(output_name=outname, equation=Evaluable.fromsympy(eqs[eqn][1]), debug=False)
            child.add_subsystem("eq{}".format(eqn), cmp, promotes=['*'])
            if bnd != 'obj':
                child.add_constraint('{}{}'.format(constraintname, eqn), **bnd)
            else:
                child.add_objective(outname)

#globals: eqs, groups
def solver(groups, eqs, parentid, childid, solvevars, res, maxiter=20):
    parent = groups[parentid]
    child = groups[childid] = parent.add_subsystem('group{}'.format(childid), 
        om.Group(), promotes=['*'])
    child.linear_solver = om.DirectSolver()
    nlbgs = child.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    nlbgs.options['maxiter'] = maxiter
    solvevarnames = [str(vr) for vr in solvevars]
    reseqs = [eqs[r] for r in res]
    diffs = [left-right if left != None else right for left,right in reseqs]
    equations = [Evaluable.fromsympy(diff) for diff in diffs]
    cmp = Impcomp(output_names=solvevarnames, equations=equations)
    child.add_subsystem("res{}".format(''.join(map(str,res))), cmp, promotes=['*'])

def explicit(groups, eqs, parentid, var, eqn):
    cmp = Expcomp(output_name=str(var), equation=Evaluable.fromsympy(eqs[eqn][1], tovar=var), debug=False)
    parent = groups[parentid]
    parent.add_subsystem("eq{}".format(eqn), cmp, promotes=['*'])