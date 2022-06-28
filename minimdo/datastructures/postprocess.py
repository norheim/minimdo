from compute import prettyprintval, prettyprintunit
import pandas as pd
from datastructures.graphutils import VAR

def print_inputs(model, prob, mdao_in):
    dvars = {elt:model.idmapping[elt] for elt in mdao_in if not model.idmapping[elt].always_input}
    return pd.DataFrame([(key, 
               prettyprintval(prob.get_val(key)[0]), 
               prettyprintunit(var.varunit)) for key,var in dvars.items()])

def print_outputs(model, prob, namingfunc):
    return pd.DataFrame([(key, 
               prettyprintval(prob.get_val(namingfunc(key.varid, VAR))[0]), 
               prettyprintunit(key.varunit)) for key,var in model.comp_by_var.items()])

def update_varval(model, prob, namingfunc):
    for key,var in model.comp_by_var.items():
        key.varval = prob.get_val(namingfunc(key.varid, VAR))[0]
        key.assumed = {key2: prob.get_val(namingfunc(key2.varid, VAR))[0] for key2,var2 in key.assumed.items()}