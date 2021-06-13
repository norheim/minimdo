import sympy as sp
import numpy as np
from autograd import grad

def args_in_order(name_dict, names):
    return [name_dict[in_var] for in_var in names]

class Equation():
    def __init__(self, left, right):
        inputs = list(right.free_symbols)
        self.fx = sp.lambdify(inputs, right, 'numpy')
        wrapped_fx = self.fx if len(inputs) == 1 else (
                lambda x: self.fx(*x)) #adapt to numpy
        self.jfx = grad(wrapped_fx)
        self.output_name = str(left)
        self.inputs_names = list(map(str, inputs))
        
    def evaldict(self, indict):
        return self.fx(*args_in_order(indict, self.inputs_names))
    
    def graddict(self, indict):
        args = np.array(args_in_order(indict, self.inputs_names))
        return self.jfx(args)