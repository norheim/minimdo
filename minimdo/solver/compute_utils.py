from trash.inputresolver import getallvars
from modeling.compute import Evaluable

def get_outputs(eqs, model):
    vrs = getallvars(eqs)
    return {elt: model.get_val(str(elt))[0] for elt in vrs}

def prettynum(num):
    if 1e-3 <= num <= 1e3:
        if float(num).is_integer():
            output = str(int(num))
        else:
            output = "{:.2f}".format(num)
    else:
        output = "{:.4g}".format(num)
    return output

def print_out(out, withunits=True):
    if withunits:
        return {key: '{:.2f~}'.format((val*key.varunit).to_compact()) for key,val in out.items()}
    else:
        return {key: prettynum(num) for key,num in out.items()}

def check_eqs(eqs, outvals):
    return {key:(outvals[str(left)], Evaluable.fromsympy(right).evaldict(outvals)) for key, (left, right) 
        in eqs.items()}
        