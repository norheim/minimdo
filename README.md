# minimdo
Toolkit for formulating conceptual design equations, and restructuring the resulting system of equations, solving them, and optimizing on top of them.

# Minimal example
This is the expected minimalist API for running the Sellar problem (https://openmdao.org/newdocs/versions/latest/basic_user_guide/multidisciplinary_optimization/sellar_opt.html#)

```python
z1,z2,x,y2 = Var('z_1'), Var('z_2'), Var('x'), Var('y_2')

F1 = Functional()
a = F1.Var('a', z2+x-0.2*y2)
y1 = F1.Var('y_1', z1**2+a)
F2 = Functional(y2, y1**0.5+z1+z2)

obj = x**2+z2+y1+exp(-y2)
ineq = [3.16-y1, y2-24] # technically represent residual sets
problem = Problem(obj, ineq, [F1, F2], {x:[0,10], z1:[0,10], z2:[0,10]})

problem.solve(x0={x:1, z1:5, z2:2}) # or seed=10 for random guess

```



