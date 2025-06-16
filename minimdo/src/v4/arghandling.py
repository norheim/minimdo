import numpy as np
import jax.numpy as anp
import jax

jax.config.update('jax_platform_name', 'cpu')

def flatten_args(args):
    return (anp.concatenate([anp.array(arg).flatten() 
                           for arg in args]) if len(args) 
                           > 0 else anp.array([]))

def scalar_if_possible(array_or_float, cleanup=False):
    if cleanup: #bypass if no cleanup
        if np.squeeze(array_or_float).ndim == 0:
            array_or_float = array_or_float.item()
    return array_or_float

def unflatten_args(x, shapes=None, convertscalars=False, tonumpy=False):
    unflattened = tuple()
    for shape in shapes:
        size = np.prod(shape)
        reshaped = x[:size]
        if size > 1:
            reshaped = reshaped.reshape(shape)
        if tonumpy:
            reshaped = np.array(reshaped, dtype=np.float64)
        unflattened += (scalar_if_possible(reshaped, convertscalars),)
        x = x[size:]
    return unflattened

def encode(d, order, flatten=False, missingarg=None):
    encoded = tuple(d.get(var, 
                          missingarg() if missingarg is not None else None) 
                          for var in order)
    if flatten:
        encoded = flatten_args(encoded)
    return encoded

def decode(x, order, shapes=None, unflatten=False, cleanup=False, **kwargs):
    if shapes is None:
        shapes = [(1,) for _ in order]
    if unflatten:
        x = unflatten_args(x, shapes, convertscalars=cleanup, **kwargs)
    return {var: x[idx] for idx, var in enumerate(order)}

class Encoding():
    def __init__(self, order=None, shapes=None, parent=None):
        self.order = tuple(order) if order is not None else tuple()
        self.parent = parent
        self.shapes = shapes if shapes is not None else ((1,),)*len(self.order)

    def unflatten(self, x, **kwargs):
        return unflatten_args(x, self.shapes, **kwargs)

    def encode(self, *args, **kwargs):
        return encode(*args, order=self.order, **kwargs)

    def decode(self, *args, **kwargs):
        return decode(*args, order=self.order, shapes=self.shapes, **kwargs)

def var_encoding(*vars):
    return Encoding(order=vars, shapes=
                    tuple(var.shape if var.shape is not None 
                          else (1,) for var in vars))

def merge_encoders(*encoders, exclude_encoder=None):
    order = tuple()
    shapes = tuple()
    exclude_order = exclude_encoder.order if exclude_encoder is not None else tuple()
    for encoder in encoders:
        for hashable,shape in zip(encoder.order, encoder.shapes):
            if hashable not in order+exclude_order:
                order += (hashable,)
                shapes += (shape,)
    return Encoding(order, shapes, parent=None)

def encoder_diff(encoder, splitter_encoder):
    keep_encoder = [(elt,shape) for elt, shape in 
                    zip(encoder.order, encoder.shapes) 
                    if elt not in splitter_encoder.order]
    order, shapes = zip(*keep_encoder)
    return Encoding(order, shapes, parent=None)

