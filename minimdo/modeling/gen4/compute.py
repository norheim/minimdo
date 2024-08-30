from itertools import chain
from modeling.gen4.arghandling import Encoding, flatten_args, merge_encoders

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