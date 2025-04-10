import pandas as pd
from ipywidgets import IntProgress
from IPython.display import display
from itertools import islice, product
from utils.randomstructure import random_problem_with_artifacts

def generate_problem(*args, df=None):
    data = []
    colnames = ["m", "ncoeff", "n", "sparsity", "seed"]
    for m,ncoeff,sparsity,seed in product(*args):
        n=m+int(ncoeff*m)
        if df is None or ((m, ncoeff, n, sparsity, seed) not in df[colnames].itertuples(index=False)):
            data.append([m, ncoeff, n, sparsity, seed])
    df2 = pd.DataFrame(data, columns=colnames)
    if df is not None:
        df2 = pd.concat([df, df2], ignore_index=True)
    return df2

from collections.abc import Iterable

#https://stackoverflow.com/questions/6710834/iterating-over-list-or-single-element-in-python
def get_iterable(x):
    if isinstance(x, Iterable) and not isinstance(x, str):
        return x
    else:
        return (x,)

def run_calculations(df, fxs, iter_cycles=None, filename=None, commonartifacts=None, argnames=None):
    commonartifacts = commonartifacts if commonartifacts is not None else random_problem_with_artifacts
    argnames = argnames if argnames else ['m','n','seed','sparsity']
    iter_cycles = iter_cycles if iter_cycles else len(df)
    f = IntProgress(min=0, max=iter_cycles) # instantiate the bar
    display(f)
    pd.options.mode.chained_assignment = None
    fxs = {get_iterable(names):fx for names,fx in fxs.items()}
    for names in fxs.keys():
        for name in names:
            if name not in df:
                df[name] = None
    for idx in islice(df.index, iter_cycles):
        row = df.iloc[idx]
        args = [df.at[idx,colname] for colname in argnames]
        for names,fx in fxs.items():
            if any(pd.isna(row[name]) for name in names):
                kwargs = commonartifacts(*args)
                out = fx(**kwargs)
                for name, val in zip(names, get_iterable(out)):
                    df[name][idx] = val
                if filename:
                    df.to_csv(filename)
        f.value += 1