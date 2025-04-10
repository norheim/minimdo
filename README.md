# minimdo
Toolkit for formulating conceptual design equations, and reconfiguring the resulting system of equations, solving them, and optimizing on top of them.

Syntax is inspired by similar packages such as cvxpy, pyomo, and gekko, but with a focus on conceptual design problems, and making it easier to run different MDO formulations, which we refer to here as reconfiguring (echoing the ideas by Alexandrov and Lewis from their publication on REMS in 2004).

# Minimal optimization example
```python
x, m1, m2, mt = symbolic("x", "m1", "m2", "mt")
D1 = MFunctionalSetLeaf(m1 == x**2-2*x+3)
D2 = MFunctionalSetLeaf(m2 == 0.5*mt) 
D3 = MFunctionalSetLeaf(mt == m1+m2)
FPF = MFunctionalSet().functionalsubsetof(D1, D2, D3).subsetof(
    3 <= x, x <= 10,
).minimize(mt)
# We partition D2 and D3 together to form a discipline
SPF = C.config(elim=[D1, MFunctionalSet(D2,D3).config(residuals=[D2, D3])])
xsol = P.solve(x0={"x": 0, "m1": 0, "m2": 0, "mt": 0})
# xsol = {"x": 1, "m1": 2, "m2": 2, "mt": 4}
```

# Sellar Problem
This is the expected minimalist syntax for running the Sellar problem (https://openmdao.org/newdocs/versions/latest/basic_user_guide/multidisciplinary_optimization/sellar_opt.html), which is based on the 1996 paper by Sellar et al. (https://arc.aiaa.org/doi/10.2514/6.1996-714). Below is a multidisciplinary feasible (MDF) formulation:

```python
z1,z2,x,a,y1,y2,indices = symbolic('z1 z2 x a y1 y2')

A1 = MFunctionalSetLeaf(a == z2+x-0.2*y2) # intermediate variable for demo
A2 = MFunctionalSetLeaf(y1 == z1**2+a)
A3 = MFunctionalSetLeaf(y2 == y1**.5+z1+z2)
FPF = MFunctionalSet().functionalsubsetof(A1,A2,A3).subsetof(
    3.16<=y1,
    0<=y2, y2<=24
).minimize(x**2+z2+y1+exp(-y2)) # Fundamental problem formulation

SPF = FPF.config(elim=[A1, A2], residuals=[A3]) # Specific problem formulation
xsol = P.solve(x0={"x": 0, "z1": 0, "z2": 0, "y1": 0, "y2":0, "a": 0})
```

