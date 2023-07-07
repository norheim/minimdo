import numpy as np
from itertools import chain

def flatten_args(args):
    return (np.concatenate([np.array(arg).flatten() 
                           for arg in args]) if len(args) 
                           > 0 else np.array([]))

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

def encode(d, order, flatten=False, missingarg=None):
    encoded = tuple(d.get(var, 
                          missingarg() if missingarg is not None else None) 
                          for var in order)
    if flatten:
        encoded = flatten_args(encoded)
    return encoded

def decode(x, order, shapes=None, unflatten=False, cleanup=False):
    if shapes is None:
        shapes = [(1,) for _ in order]
    if unflatten:
        x = unflatten_args(x, shapes, cleanup)
    return {var: x[idx] for idx, var in enumerate(order)}

class Encoder():
    def __init__(self, order=None, shapes=None, parent=None):
        self.order = order if order is not None else tuple()
        self.parent = parent
        self.shapes = shapes if shapes is not None else ((1,),)*len(self.order)

def merge_encoders(*encoders, exclude_encoder=None):
    order = tuple()
    shapes = tuple()
    exclude_order = exclude_encoder.order if exclude_encoder is not None else tuple()
    for encoder in encoders:
        for hashable,shape in zip(encoder.order, encoder.shapes):
            if hashable not in order+exclude_order:
                order += (hashable,)
                shapes += (shape,)
    return Encoder(order, shapes, parent=None)

class EncodedFunction():
    def __init__(self, f, encoder=None, decoder=None, **kwargs):
        self.f = f
        self.kwargs = kwargs
        self.encoder = encoder if encoder is not None else Encoder()
        self.decoder = decoder if decoder is not None else Encoder()

    def dict_out_only(self, *args, **kwargs):
        return decode(self.f(*args, **{**self.kwargs, **kwargs}), 
                      self.decoder.order, shapes=self.decoder.shapes)

    def dict_in_only(self, d, **kwargs):
        return self.f(*encode(d, self.encoder.order), **{**self.kwargs, **kwargs})
    
    def dict_in_flat_out(self, d, **kwargs):
        return flatten_args(self.dict_in_only(d, **{**self.kwargs, **kwargs}))
    
    def dict_in_dict_out(self, d, **kwargs):
        return decode(self.dict_in_only(d, **{**self.kwargs, **kwargs}), 
                      self.decoder.order, shapes=self.decoder.shapes)
    
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
