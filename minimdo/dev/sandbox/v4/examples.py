# New code interface
# full set of variables: x1, x2, x3, a, b, c
P1 = subproblem()
P1.addequation('x1', 'x3', lambda x3: x3**2)
P1.addequation('x2', ['x1', 'x3'], lambda x1, x3: x1 + x3)
P1.solver('ROOT', eliminate=['x1','x2']) # this is the default setting, shorthand would be sparse=True vs sparse=False which would turn off elimination and it is a subset from the eliminatable variables, which in this case are x1 and x2
rendergraph(P1, hide=['x2']) #this would hide any variables listed but still show dependencies

P2 = subproblem()
P2.addequation(0, 'a*x_3**2 + b*x_3+c')
P2.invproject('x_3')
P2.solver('ROOT')

P5 = problem(P1, P2, presolve='minarcset') #default solver to ROOT
P5.solver('ROOT') 
P5.solve() 

P5 = problem(P1, P2, presolve='minarcset') 
P5.solver('ROOT') 
P5.solve() 

P4 = P1.eliminate(P2)

P4.solve(a=1, b=2, c=3) # this assumes that x1, x2, x3 are all calculated and do not need to be solved for

P4.solver('OPT','MIN', lambda x_2: x_2)
P5 = P4.presolve() # changes design variables to optimize structure
P5.solve('OPT', x1=0,x2=0,x3=0)

## Alternative way to do the same thing with IDF:
P3 = subproblem()
P3.merge(P1, P2) # equivalent of lambda y: x1,x2=P1.solve(y), x3=P2.solve(y)
P3.solver('ROOT', eliminate=['x1','x2','x3'])
P3.solve(a=1, b=2, c=3)

P3.solver('OPT','MIN', lambda x_2: x_2)

# This one is probably not needed
# P3 = subproblem()
# P3.addequation(['a', 'b', 'c'], [1, 2, 3])
# P5 = P4.eliminate(P3)

