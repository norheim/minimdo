from graphutils import merge_edges
import numpy as np

def bindFunction(function, n_reversed):
    def residual(*x):
        return np.array(x[n_reversed:])-function(*x[0:n_reversed]) 
    return residual

def generate_residuals(Ein, Rin, f):
    residuals = dict()
    merged_edges = merge_edges(Ein,Rin) # this makes sure we get the same order as used during workflow generation
    for fx,ins in Rin.items():
        merged_ins = merged_edges[fx]
        fkey = (Rin[fx], Ein[fx]) #Rin encodes the old outputs
        function = f[fkey]
        n_reversed = len(ins)
        output_size = (None,)*n_reversed
        # need to do some local binding for residual function
        residuals[(merged_ins, output_size)] = bindFunction(function, n_reversed)
    return residuals