from itertools import chain
from graphutils import solver_children, end_components

def path(Stree, s, visited=None):
    visited = visited if visited else set()
    out = []
    q = {s} if s in chain(Stree.values(),Stree.keys()) else set()
    while q:
        s = q.pop()
        if s not in visited:
            out.append(s)
            if s in Stree:
                q.add(Stree[s])
        visited.add(s)
    return out

def generate_workflow(edges, tree):
    Fend = end_components(edges[1])
    Ftree, Stree, Vtree = tree
    visited = set()
    workflow = []
    for key,parentsolver in Ftree.items():
        out = path(Stree, parentsolver, visited)
        visited = visited.union(out) 
        workflow.extend([("solver", s, Stree.get(s, None), set(solver_children(Vtree, s)), set(solver_children({F:Ftree[F] for F in Fend}, s))) for s in out[::-1]])
        if key not in Fend:
            workflow.append(("exec", key, parentsolver))
    return workflow