import numpy as np

def flatten_args(args):
    return np.concatenate([np.array(arg).flatten() for arg in args])

def scalar_if_possible(array_or_float):
    squeezed = np.squeeze(array_or_float)
    if squeezed.ndim == 0:
        squeezed = squeezed.item()
    return squeezed

def unflatten_args(x, shapes=None, cleanup=False):
    unflattened = tuple()
    for shape in shapes:
        size = np.prod(shape)
        reshaped = x[:size]
        if size > 1:
            reshaped = reshaped.reshape(shape)
        if cleanup:
            reshaped = scalar_if_possible(reshaped)
        unflattened += (reshaped,)
        x = x[size:]
    return unflattened

def encode(d, order, flatten=False):
    encoded = tuple(d[var] for var in order)
    if flatten:
        encoded = flatten_args(encoded)
    return encoded

def decode(x, order, shapes=None, unflatten=False, cleanup=False):
    if shapes is None:
        shapes = [(1,) for _ in order]
    if unflatten:
        x = unflatten_args(x, shapes, cleanup)
    return {var: x[idx] for idx, var in enumerate(order)}