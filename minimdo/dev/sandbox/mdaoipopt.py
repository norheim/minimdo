import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped

prob = om.Problem()
model = prob.model = SellarDerivativesGrouped()

prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = "IPOPT"

model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
model.add_design_var('x', lower=0.0, upper=10.0)
model.add_objective('obj')
model.add_constraint('con1', upper=0.0)
model.add_constraint('con2', upper=0.0)

prob.set_solver_print(level=0)

prob.setup(check=False, mode='rev')
prob.run_driver()