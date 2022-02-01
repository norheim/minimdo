from compute import END, INTER

# Incidence structure
def all_vars_from_incidence(incstr, include_par=False):
    return {var for varsineq in incstr.values() for var in varsineq if not var.always_input or include_par}

def notation_from_tree(tree, solvefor):
    nt = []
    endcomps = []
    solvefr = []
    sequence = tree.children
    for elt in sequence:
        if elt.is_leaf:
            if elt.node_type == END:
                endcomps.append(elt.name)
                solvefr.append(solvefor[elt.ref])
            else:
                nt.append((solvefor[elt.ref], elt.name))
        else:
            nt.append(notation_from_tree(elt, solvefor))
    nt.extend(endcomps)
    if solvefr:
        return (solvefr, nt)
    else:
        return nt