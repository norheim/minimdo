# minimdo
Toolkit for formulating conceptual design equations, and restructuring the resulting system of equations, solving them, and optimizing on top of them.

Syntax is inspired by similar packages such as cvxpy, pyomo, and gekko, but with a focus on conceptual design problems, and making it easier to run different MDO formulations.

# Minimal optimization example
This is the expected minimalist syntax for running the Sellar problem (https://openmdao.org/newdocs/versions/latest/basic_user_guide/multidisciplinary_optimization/sellar_opt.html), which is based on the 1996 paper by Sellar et al. (https://arc.aiaa.org/doi/10.2514/6.1996-714). Below is a multidisciplinary feasible (MDF) formulation:

```python
z1,z2,x,y2 = Var('z_1'), Var('z_2'), Var('x'), Var('y_2')

F1 = Functional()
a = F1.Var('a', z2+x-0.2*y2) # intermediate variable for demonstration purposes
y1 = F1.Var('y_1', z1**2+a)
F2 = Functional(y2, y1**.5+z1+z2)

#F = F2.eliminate(F1) # eliminates 'a' and 'y_1'
F = intersection(F1, F2) # eliminates 'y_1' and 'y_2'

obj = x**2+z2+y1+exp(-y2)
ineq = [3.16-y1, y2-24] # technically represent residual sets
# eq = F.projectable # using residuals for F1 and F2
problem = Problem(obj, ineq, eliminate=F, 
                {x:[0,10], z1:[0,10], z2:[0,10]}) # bounds

problem.solve(x0={x:1, z1:5, z2:2}) # or seed=10 for random guess
```

We can also run the AAO/IDF formulation by using the residual set as equality constraints:

```python
problem = Problem(obj, ineq, F.projectable,
                  bounds={x:(0,10), z1:(0,10), z2:(0,10)})
xopt_aao = problem.dict_in_dict_out(x0={x:1, z1:1, z2:1, 
                             a:1, y1:1, y2:1})
```