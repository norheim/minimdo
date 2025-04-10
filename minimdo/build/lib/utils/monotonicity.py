import cvxpy as cv
import sympy as sp

signlookup = {
    'NONPOSITIVE': '-',
    'NONNEGATIVE': '+',
    True: '+',
    False: '-'
}

def get_monotonicites(expr):
    out = {}
    expr_symb = [symb for symb in expr.free_symbols]
    cvvar = [cv.Variable(name=s.name, pos=True) for s in expr_symb]
    for s in expr_symb:
        if not s.never_output:
            dexpr = sp.diff(expr, s)
            f = sp.lambdify(expr_symb, dexpr, {'sqrt': cv.sqrt}) #make sure we have tests for other math functions
            cvexpr = f(*cvvar)
            out[s] = signlookup.get(cvexpr.sign, 'n') if hasattr(cvexpr, 'sign') else signlookup[cvexpr>0]
            #print(s, cvexpr, cvexpr.sign if hasattr(cvexpr, 'sign') else cvexpr>0)
    return out