from modeling.compute import prettyprintval, prettyprintunit
import pandas as pd
from graph.graphutils import VAR
from graph.matrixview import incidence_artifacts

# Based on whatever is in varval, and not on running an MDAO model
def print_values_static(model, varnames=None, get_value=None, display=True, rounding=None):
    get_value = get_value if get_value else lambda var: var.varval
    varobjs = model.comp_by_var.keys() if not varnames else (model.idmapping[varname] for varname in varnames)
    df = pd.DataFrame([('$${}$$'.format(varobj), 
               prettyprintval(get_value(varobj), rounding=rounding), 
               prettyprintunit(varobj.varunit)) for varobj in varobjs])
    if display:
        return df.style.hide(axis="columns").hide(axis="index")
    else:
        return df

def print_outputs(model, prob, namingfunc, varnames=None, display=True, rounding=None):
    get_value = lambda var: prob.get_val(namingfunc(var.varid, VAR))[0]
    return print_values_static(model, varnames, get_value, display, rounding)

def print_inputs(model, prob, namingfunc, varnames, filterparam=True, display=True):
    fltr = (lambda x: model.idmapping[x].always_input) if filterparam else lambda x: False
    varnames_noparam = {elt for elt in varnames if not fltr(elt)}
    return print_outputs(model, prob, namingfunc, varnames_noparam, display)

def update_varval(model, prob, namingfunc, varnames=None):
    var_refs = model.comp_by_var.keys() if varnames is None else (model.idmapping[varname] for varname in varnames)
    for var in var_refs:
        var.varval = prob.get_val(namingfunc(var.varid, VAR))[0]
        var.assumed = {key2: prob.get_val(namingfunc(key2.varid, VAR))[0] for key2 in var.assumed.keys()}

def print_vars_in_order(prob, edges, tree, var_mapping):
    d = {varn:prob.get_val(varn) for _,(_,varn) in var_mapping.items()}
    df = pd.DataFrame(d)
    _, permutation, _, _, _ = incidence_artifacts(edges, tree)
    return df[[var_mapping[elt][1] for elt in permutation[::-1]]].T