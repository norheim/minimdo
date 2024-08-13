import numpy as np
import cyipopt
import jax.numpy as jnp
from jax import jacfwd

class Problem:
    def __init__(self, n, constraints, constraints_jacobian):
        self.n = n
        self.constraints = constraints
        self.jacobian = constraints_jacobian

    def objective(self, x):
        return 0
    
    def gradient(self, x):
        return np.zeros(self.n)

    def constraints(self, x):
        return self.constraints(x)
    
    def jacobian(self, x):
        return self.jacobian(x)

def ipoptsolve(rx, solvevars):
    def f(y):
        constraint = lambda x: rx(jnp.hstack((x,y)))[0]
        constraints_jacobian = lambda x: jacfwd(constraint)(x)[0]
        n = m = len(solvevars)
        P = Problem(n, constraint, constraints_jacobian)
        problem = cyipopt.Problem(
            n=n,
            m=m,
            problem_obj=P,
            lb=np.zeros(n),
            ub=1e6*np.ones(n),
            cl=np.zeros(m), #equality constraints
            cu=np.zeros(m)
        )
        x0 = np.random.rand(n)
        return problem.solve(x0)
    return f