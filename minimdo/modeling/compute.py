from itertools import count
from modeling.unitutils import get_unit, ureg
import sympy as sp
import numpy as np
import re

# The function will work for sympy symbols or just plain strings
def get_latex(symbol_or_string):
    return symbol_or_string if symbol_or_string else r'\mathrm{{{}}}'.format(symbol_or_string)

def prettyprintval(x, latex=False, unit=None, rounding=None):
    rounding = rounding if rounding is not None else 3
    sci_expr = '{'+':.{}e~P'.format(rounding)+'}' if not latex else '{'+':.{}e~L'.format(rounding)+'}'
    nonsci_expr = '{'+':.{}f'.format(rounding)+'}'
    if x == None:
        return None
    elif (x>1e4 or x<1e-3) and x!=0:
        x_removed_brackets = np.squeeze(x)[()]
        return sci_expr.format(ureg.Quantity(x_removed_brackets, unit))
    else:
        unit_expr = '\ {:L~}' if latex else ' {}'
        unitstr = prettyprintunit(unit, unit_expr, latex) if unit else ''
        return r'{}{}'.format(nonsci_expr.format(x).rstrip('0').rstrip('.'),unitstr)

def prettyprintunit(x, strformat='{:P~}', latex=False):
    if x.units != ureg('year'):
        return strformat.format(x.units)
    else:
        return r'\ \mathrm{yr}' if latex else 'yr'

def get_assumed_string(assumed):
    return (r'{}={}'.format(get_latex(key),prettyprintval(val,latex=True)) for key,val in assumed.items())

def remove_frac_from_latex(latexstr):
    return re.sub(r'\\frac{(.*)}{(.*)}', r'\1/\2', latexstr)

class Var(sp.core.Symbol):
    def __new__(cls, name, value=None, unit=None, always_input=False, varid=None, forceunit=False):
        #clear_cache()  # sympys built in cache can cause unexpected bugs
        out = super().__new__(cls, name) #positive=positive)
        out.always_input = always_input
        out.varval = value
        out.varunit = ureg(unit) if unit else ureg('')
        out.forceunit = forceunit
        out.varid = varid if varid != None else name
        out.assumed = dict() # sometimes to generate varval we need to assume values
                             # for the function that computed this value 
        out.shape = None
        return out
    
    def custom_latex_repr(self):
        if self.varval != None:
            assumed = ''
            if self.assumed:
                assumed = '\ ({} )'.format(' ,'.join(get_assumed_string(self.assumed)))
            if self.varunit.dimensionless:
                varstr = prettyprintval(self.varval, latex=True)+' '
            else:
                varstr = prettyprintval(self.varval, latex=True, unit=self.varunit)
                # remove frac's for more compact notation
                varstr = remove_frac_from_latex(varstr)
            # Need to synchronize the name of 'dummy' here and in api
            namestr = '{}='.format(self.name) if self.name != 'dummy' else ''
            return '{}{}{}'.format(namestr, varstr, assumed)
        else:
            return self.name

    def _repr_latex_(self):
        return '${}$'.format(self.custom_latex_repr())

def create_vars(string, split=' '):
    return [Var(elt) for elt in string.split(split)]

class Par(Var):
    _ids = count(0)
    def __new__(self, *args, **kwargs):
        if (len(args) >= 1 and not isinstance(args[0], str)) or len(args) == 0:
            args = ('p{}'.format(next(self._ids)),)+args
        out = Var.__new__(self, *args, **kwargs)
        out.always_input = True
        return out
    