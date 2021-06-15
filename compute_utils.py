from inputresolver import getallvars

def get_outputs(eqs, model):
    vrs = getallvars(eqs)
    return {elt: model.get_val(str(elt))[0] for elt in vrs}

def check_eqs(eqs, outvals):
    return {key:(outvals[left], right.subs(outvals)) for key, (left, right) 
        in eqs.items()}
        