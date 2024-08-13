---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Encoded Functions

Regular Python function *arguments* are not easily accessed. To get around this we implemented something called Encoded Functions, which also makes it easy for functions to be evaluated with three equivalent inputs: a dictionary, a list, or numpy arrays.

## Encoding function arguments
Given a dictionary, and a list of argument names, it generate a list of arguments in the order specified by the list. There is also an option to further turn the list of arguments to a flattened numpy vector. We refer to this process as encoding, as ultimately the function arguments can be encoded into a vector, and the same vector should be decodable back to a dictionary.

```
encode({x: 1., y: np.array([2.]), z:np.array([[2.,3],[4,5]])}, A.encoder.order, flatten=True)
# np.array([1, 2, 2, 3, 4, 5])
```


## Encoding
Encoding objects contain a list of argument names, and a list of argument shapes. You can think of encoding objects as abstract numpy arrays, but without any content. It makes it possible to turn a list of vectors, arrays and other tensors into a flat list, and vice-versa to recover the exact same shape.
We will need this to interface with algorithms that take as input only a flat vector (i.e. scipy fsolve).

For the previous example, the encoding is as follows:

```
E = Encoding(('a', 'b', 'c'), ((1,), (1,), (2,2)))
```

