# minimdo
Toolkit for formulating conceptual design equations, and restructuring the resulting system of equations, solving them, and optimizing on top of them.

Syntax is inspired by similar packages such as cvxpy, pyomo, and gekko, but with a focus on conceptual design problems, and making it easier to run different MDO formulations.

# Minimal optimization example
This is the expected minimalist syntax for running the Sellar problem (https://openmdao.org/newdocs/versions/latest/basic_user_guide/multidisciplinary_optimization/sellar_opt.html), which is based on the 1996 paper by Sellar et al. (https://arc.aiaa.org/doi/10.2514/6.1996-714). Below is a multidisciplinary feasible (MDF) formulation:

```python
z1,z2,x,a,y1,y2,indices = symbols('z1 z2 x a y1 y2')

A1 = AnalyticalSetSympy(z2+x-0.2*y2, a, indices) # intermediate variable for demonstration purposes
A2 = AnalyticalSetSympy(z1**2+a, y1)
A3 = AnalyticalSetSympy(y1**.5+z1+z2,y2)
obj = FunctionSympy(x**2+z2+y1+exp(-y2))
ineq = EliminateAnalysisMergeResiduals(functions=[FunctionSympy(3.16-y1), 
FunctionSympy(y2-24)])

R = EliminateAnalysisMergeResiduals([A1.analysis, A2.analysis], [A3.residual])
S = ElimResidual(R, A3.structure[1], indices, solvefor_raw=True)
P = EliminateAnalysisMergeResiduals([A1.analysis, A2.analysis], [obj, ineq])
solver_indices = P.structure[0]

x0=torch.random(len(solver_indices), dtype=torch.float64)
xguess, obj, ineq, eq, dobj, dineq, deq = generate_optim_functions(P, solver_indices, x0)

## We can now use give the functions to a solver, such as scipy.optimize.minimize
```

