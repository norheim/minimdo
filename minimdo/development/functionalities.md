# Functional representable sets (Functionals)

Assume we have $x=(x_1,x_2,x_3,x_4,x_5)$, a vector in 5-dimensional (real) space. 

## A first subspace
Now we have a set $S_1$ that describes a subspace of the 5-dimensional space: the subset of points satisfying the following equations:

$$
\begin{align*}
x_2^2 + x_3^2 -2 &= 0 \\
x_2x_3x_5 -1 &= 0
\end{align*}
$$

Now this subset has an infinite number of points. However, if we fix one of the dimensions, say $x_5=2$(i.e. take the intersection of $S_1$ with the hyperplane $x_5=2$), we get a system of two equations in two variables, for which a finite number of (real) solutions (potentially zero) should exist. 

This Python library has been designed for when we can assume that $S_1$ actually has one solution for each value of $x_5$ in some domain. *We will say that* $x_5$ *parametrizes the set* $S_1$. Let's say we want to query the solutions at different values for $x_5$. Let's show how to do this with the library:
    
```python   
# This syntax will seem familiar to SymPy users
x_2, x_3, x_5 = Var('x_1'), Var('x_2'), Var('x_3'), Var('x_4'), Var('x_5')

eq1 = Component.fromsympy(x_2**2 + x_3**2-2, arg_mapping=True) 
eq2 = Component.fromsympy(x_2*x_3*x_5 - 1, arg_mapping=True)
S1 = ResidualSet([eq1, eq2])
F1 = S1.project((x_5,)) # This is the projection of S1 onto the x_5 axis for a specific value of x_5
F1.solve({x_5: 2})
# returns {x_3: 0.366, x_2: 1.366}
```

Alternativaly we could have thought of $S_1$ as the intersection of two sets: $H_1$ and $H_2$, where $H_1$ is the set of points satisfying $x_2^2 + x_3^2 -2 = 0$ and $H_2$ is the set of points satisfying $x_2x_3x_5 -1 = 0$. If we fix two of the dimensions for $H_2$, for example $x_3$ and $x_5$, we get one equation with one variable ($x_2$), and we can similarly turn it into an object that we can query for many such combinations of $x_3$ and $x_5$. Now we could use this function to eliminate $x_2$ in the equation for the first set $H_1$, and now get again an equation with one variable, $x_3$, and solve for any $x_3$(and then get $x_2$ by extension) given an $x_5$:


```python
H_2 = ResidualSet([eq2])
F_eq2_x2 = H_2.project((x_3, x_5))
F_elim = EliminationKeepSet([eq1], eliminate=F_eq2_x2).project((x_5,))
F_elim.solve({x_5: 2})
# returns {x_3: 0.37, x_2: 1.37}
```

## A second subspace

We now have a second set $S_2$, the subset of points satisfying the following equations, and assume it is projected onto $x_2,x_3,x_5$, in which case we can compute $x_1,x_4$ from a simple feed forward:

$$
\begin{align*}
x_1 &= x_3^2+x_2^2+1 \\
x_4 &= 10-x_2-x_3-x_1-x_5\\
\end{align*}
$$

```python
def black_box(x3, x2,x5):
    x1 = (x3**2+x2**2+1)**0.5
    x4 = 10+x2+x3-x1-x5
    return x1, x4
eq3 = Component(black_box, (x_3,x_2,x_5), (x_1,x_4), arg_mapping=True)
F2 = FunctionalComp(eq3)
F2.solve({x_2:1, x_3:1, x_5:2})
# returns {x_1: 1.73, x_4: 8.27}
```

## Intersecting sets
We calculate the intersection of $S_1$ and $S_2$ through two methods:

### Method 1
Since $S_1$ does not constrain $x_1$ and $x_4$, we can simply first find a set of $(x_2,x_3)$ member in $S_1$, and then find the corresponding $(x_1,x_4)$ member in $S_2$. 

```python
S2 = F2.projectable
S3 = S1.merge(S2)
F3 = S3.project((x_5,))
F3.solver = FeedForwardSolver([F1, F2])
F3.solve({x_5: 2})
# returns {x_2: 1.37, x_3: 0.37, x_1: 1.73, x_4: 8.0}
```

### Method 2
We could instead solve the larger system of equations:
$$
\begin{align*}
x_2^2 + x_3^2 -2 &= 0 \\
x_2x_3x_5 -1 &= 0 \\
x_3^2+x_2^2 + 1 - x_1 &= 0 \\
10+x_2+x_3-x_1-x_5-x_4 &= 0\\
\end{align*}
$$

```python
F3.solver = DefaultResidualSolver()
F3.solve({x_5: 2})
# returns {x_2: 1.37, x_3: 0.37, x_1: 1.73, x_4: 8.0}
```

