from sympy import lambdify
from scipy.optimize import fsolve
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, custom_jvp
import jax
jax.config.update('jax_platform_name', 'cpu')
from itertools import chain

# For importing sympy based models
def pure_functions(fs):
    function_data = []
    for f in fs:
        arg_order = tuple(sorted(f.free_symbols, 
                                 key=lambda s: s.name))
        l = lambdify([arg_order], f, modules=jnp)
        function_data.append((l, (), arg_order))
    return function_data

def coupled_functions(fs):
    function_data = []
    for f, outarg in fs:
        arg_order = tuple(sorted(f.free_symbols, 
                                 key=lambda s: s.name))
        l = lambdify([arg_order], f, modules=jnp)
        function_data.append((l, (outarg,), arg_order))
    return function_data

def get_allvars(Fs):
    _,allout,allin = zip(*Fs)
    allvars = ()
    for vrs in chain(allout,allin):
        allvars += tuple(vr for vr in vrs if vr not in allvars)
    coupled_out = ()
    for vrs in allout:
        coupled_out +=tuple(vr for vr in vrs if vr not in coupled_out)
    coupled_in = ()
    for vrs in allin:
        coupled_in +=tuple(vr for vr in vrs if (
            (vr not in coupled_in) and (vr not in coupled_out)))
    return allvars, coupled_in

def sequential_structure(Fs, varinfo=None):
    # Assumes a structure as follows:
    # (function, input indices, output indices)
    # to be used in combination with sequential_evaluation
    if varinfo is None:
        allvars, coupled_in = get_allvars(Fs) 
    else:
        allvars, coupled_in = varinfo
    lookup_table = []
    for Fi,youts,yins in Fs:
        lookup_table.append((Fi,
            np.fromiter((allvars.index(yout) 
                         for yout in youts),dtype=int),
            np.fromiter((allvars.index(yin) 
                         for yin in yins),dtype=int)))  
    indices_coupled_in=np.fromiter((allvars.index(vr) 
                    for vr in coupled_in), dtype=int)
    return indices_coupled_in, lookup_table, allvars, coupled_in

def sequential_evaluation(indices_coupled_in, 
                           lookup_table, allvars):
    def f(x):
        xout = jnp.zeros(len(allvars))
        xout = xout.at[indices_coupled_in].set(x)
        for fi, indexout, indexin in lookup_table:
            xout = xout.at[indexout].set(fi(xout[indexin]))
        return xout
    return f

def concat(indices_coupled_in, lookup_table, allvars, 
           eliminate_table=None):
    eliminate_table = eliminate_table if eliminate_table is not None else ()
    felim = sequential_evaluation(indices_coupled_in, 
                                  eliminate_table,
                                  allvars)
    def f(x):
        x = felim(x)
        return jnp.array([fi(x[indexin]) for 
                            fi, _, indexin 
                            in lookup_table])
    return f

def partial(eqs, solvefor, eliminate=None):
    m = len(eqs)
    eliminate = eliminate if eliminate is not None else ()
    eqs = eqs + eliminate
    allvars, coupled_in = get_allvars(eqs)
    yother = tuple(vr for vr in coupled_in 
                   if vr not in solvefor)
    coupled_in_order = solvefor + yother # order is critical
    indices_coupled_in, lookup_table, _, coupled_in = (
        sequential_structure(eqs, varinfo=(
            allvars,coupled_in_order)))
    rx = concat(indices_coupled_in, lookup_table[:m], 
                allvars, eliminate_table=lookup_table[m:])
    return rx, yother

def solver(eqs, solvefor, x0_pointer, eliminate=None):
    # x0_pointer is a hack to use a list
    # lambdify eqs
    rx, yother = partial(eqs, solvefor, eliminate)
    gradf = jacfwd(rx)
    n = len(solvefor)
    @custom_jvp
    def f(y):
        return fsolve(lambda x: rx(np.hstack((x,y))), 
                      x0=x0_pointer[0])
    @f.defjvp
    def f_jvp(primals, tangents):
        y, = primals
        x_dot,  = tangents
        fval = f(y)
        x_full = np.hstack((fval,y))
        #print(rx(x_full))
        grad_val = np.vstack(gradf(x_full)).T
        grad_hy, grad_hx = grad_val[:n,:],grad_val[n:,:]
        inv_grad_hy = np.linalg.inv(grad_hy.T)
        dJ = -np.dot(inv_grad_hy, grad_hx.T).T
        tangent_out = sum([dj*x_dot[idx] for idx,dj in enumerate(dJ)])
        return fval, tangent_out
    return f, solvefor, yother


def sequential_elimination(f_with_in, phi):
    f_to_reduce, indexin = f_with_in
    def f(x):
        xout = phi(x)
        return f_to_reduce(xout[indexin])
    return f

def compose(fs, forelimin=None):
    if forelimin is None:
        forelimin = ()
    indices_coupled_in, lookup_table, allvars, Fin = sequential_structure(fs+forelimin)
    nfs = len(fs)
    F = sequential_evaluation(indices_coupled_in, 
                           lookup_table[:nfs], allvars)
    Fdata = (F, allvars, Fin)
    Rf = concat(indices_coupled_in, lookup_table[nfs:], 
                allvars, lookup_table[:nfs])
    Rdata = (Rf, (), Fin) if forelimin else ()
    return Fdata, Rdata

def eliminate(Rdata, phi):
    Rf,_,_=Rdata
    Ff,_,_=phi
    def f(x):
        return Rf(Ff(x))
    return f

def couplingvars(Fdata,Rdata, solvevars=None):
    solvevars = (solvevars if solvevars is not None 
                 else tuple())
    feed_forward = ()
    residuals = ()
    all_invars = ()
    for F,R in zip(Fdata,Rdata):
        f,outvars,invars = F
        r,outvars_r,invars_r = R
        if len(set(outvars) & set(all_invars))==0:
            all_invars += tuple(elt for elt in invars 
                        if elt not in all_invars)
            feed_forward += ((f,outvars,invars),)
            res = () # no residual
        else:
            solvevars += tuple(elt for elt in outvars 
                                if elt not in solvevars)
            res = ((r,outvars_r,invars_r),)
        residuals += res
    return feed_forward, residuals, solvevars