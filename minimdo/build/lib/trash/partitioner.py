from inputresolver import default_in

def default_out_condensation(alleqs, eqlist):
    out = set()
    remove = set()
    keep = set()
    for left, _ in eqlist:
        out = out.union({left})
    for _, right in eqlist:
        remove = remove.union(right.free_symbols)
    for _, right in set(alleqs)-set(eqlist):
        keep = keep.union(right.free_symbols)
    out = out.intersection(keep)
    out.union(keep-remove)
    return out

def merge(alleqs, eqs, eqname):
    din_s1 = default_in(eqs, eqdictin=False)
    din_s1 = [elt for elt in din_s1 if not elt.never_output]
    dout_s1 = default_out_condensation(alleqs, eqs)
    edges = tuple((elt, eqname) for elt in din_s1)+tuple((eqname, elt) for elt in dout_s1)
    return edges