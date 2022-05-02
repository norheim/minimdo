# designresolver
Toolkit for formulating conceptual design equations, changing between declarative and input/output formulations and solve the resulting system of equations.

# Minimal example
This is minimal code for running the Sellar problem 
```python
z1,z2,x,y2 = Var('z1'), Var('z2'), Var('x'), Var('y2')

model = Model(solver=OPT)
m = model.root
a = adda(m, 'a', z2+x-0.2*y2)
y1 = adda(m, 'y1', z1**2+a)
adda(m, y2, y1**0.5+z1+z2)
addobj(m, x**2+z2+y1+exp(-y2)) 
addineq(m, 3.16-y1) 
addineq(m, y2-24) 
setsolvefor(m, [x,z1,z2], {x:[0,10], z1:[0,10], z2:[0,10]})
prob, mdao_in, groups = model.generate_mdao()

prob.run_driver() # optimal value about 3.183
```

# organization
+ ds: data structures
|--- ds_symbolic
|--- ds_units
|--- ds_funcstruct
|--- ds_tree
|--- ds_notation
+ cp: compute
|--- cp_mdao
|--- cp_operators
+ al: algorithms
|--- al_matching - Matching bipartite graph
|--- al_tearing - Chosing tearing variables to optimize structure
|--- al_inputopt - Chosing inputs to optimize structure
|--- al_ma - Monotonicity Analysis
+ vz: visualizations
|--- vz_graphs
|--- vz_dsm
+ app: applications
|--- app_allison
|--- app_pump - Example from Allisons thesis
|--- app_coffee
|--- app_balloon
|--- app_minisat - 5 eqs system spacecraft
|--- app_miniac - 7 eqs aircraft
|--- app_sat
|--- app_pearl

