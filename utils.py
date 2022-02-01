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

def normaliz_name(name):
    # f_{{{11}}} --> f11
    return re.sub(r'(.+)\_\{\{\{(.+)\}\}\}', r'\1\2', name)