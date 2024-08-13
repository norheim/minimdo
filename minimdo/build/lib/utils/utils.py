import re
import inspect

def invmap(original):
    invor = {}
    for k,v in original.items():
        for x in v:
            invor.setdefault(x,[]).append(k)
    return invor

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def normalize_name(name, keep_underscore=False):
    # f_{{{11}}} --> f11 
    # \rho_{hi} --> rho_hi 
    pattern = r'(\w+\_){(\w+)}'
    sub1 = re.sub(pattern, r'\1\2', name)
    if not keep_underscore:
        sub1 = re.sub(r'\_', r'', sub1)
    sub2 = re.sub(r'\\', r'', sub1)
    return sub2