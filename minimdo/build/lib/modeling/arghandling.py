import numpy as np
import jax.numpy as anp
import jax
jax.config.update('jax_platform_name', 'cpu')
from itertools import chain

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

class EncodedFunction():
    def __init__(self, f, encoder=None, decoder=None, **kwargs):
        self.f = f
        self.kwargs = kwargs
        self.encoder = encoder if encoder is not None else Encoding()
        self.decoder = decoder if decoder is not None else Encoding()

    def flat_in_only(self, x, **kwargs):
        return self.f(*self.encoder.unflatten(x), 
                      **{**self.kwargs, **kwargs})
    
    def flat_in_flat_out(self, x, **kwargs):
        return flatten_args(self.flat_in_only(x, **kwargs))

    def dict_out_only(self, *args, cleanup=False, **kwargs):
        return self.decoder.decode(self.f(*args, 
                                          **{**self.kwargs, **kwargs}), 
                                   cleanup=cleanup)

    def flat_in_dict_out(self, x, **kwargs):
        return self.dict_out_only(*self.encoder.unflatten(x), 
                                  **kwargs)

    def dict_in_only(self, d=None, **kwargs):
        d = d if d is not None else dict()
        return self.f(*self.encoder.encode(d), **{**self.kwargs, **kwargs})
    
    def dict_in_flat_out(self, d=None, **kwargs):
        d = d if d is not None else dict()
        return flatten_args(self.dict_in_only(d, **kwargs))
               
    def dict_in_dict_out(self, d=None, cleanup=False, **kwargs):
        d = d if d is not None else dict()
        return self.decoder.decode(self.dict_in_flat_out(d, **kwargs),
                                    unflatten=True, cleanup=cleanup)
    
    def __repr__(self) -> str:
        return '{} <- {}'.format(str(self.decoder.order), 
                                  str(self.encoder.order))
    
class EncodedFunctionContainer(EncodedFunction):
    def __init__(self, encoded_functions=None, f=None, **kwargs):
        super().__init__(f, **kwargs)
        self.encoded_functions = list()
        encoded_functions = encoded_functions if encoded_functions is not None else tuple()
        self.add_encoded_functions(*encoded_functions)

    def add_encoded_functions(self, *encoded_functions):
        self.encoded_functions.extend(encoded_functions)
        self.decoder = merge_encoders(*chain((self.decoder,), 
            (encoded_function.decoder for encoded_function 
             in encoded_functions)))
        self.encoder = merge_encoders(*chain((self.encoder,),
            (encoded_function.encoder for encoded_function 
             in encoded_functions)), exclude_encoder=self.decoder)
    
    def __repr__(self) -> str:
        joinlist = [str(elt) for elt in self.encoded_functions]
        return '\n'.join(joinlist)
