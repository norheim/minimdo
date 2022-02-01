from networkx.algorithms.bipartite import random_graph as bipartite_random_graph
import numpy as np
import copy

def generate_random_prob(n_eqs, n_vars, seed=8, sparsity=1.1):
    n_nodes = n_eqs + n_vars
    p = n_nodes/(n_eqs*n_vars + 1.0e-17)*sparsity
    p = min(p, 1.0 - 1.0e-6) 
    G = bipartite_random_graph(n_eqs, n_vars, p, seed)
    #G = nx.bipartite.gnmk_random_graph(n_eqs, n_vars, int(0.2*n_vars*n_eqs), seed)
    eqs = list(range(n_eqs))
    vrs = list(range(n_eqs, n_eqs+n_vars))
    rng = np.random.default_rng(8)
    # Make sure that there is at least one matching possible
    M = dict(zip(rng.permutation(eqs), rng.permutation(vrs)))
    # Complete the incidence matrix with the matching
    for node in eqs:
        G.add_edge(node, M[node])
    eqv = {elt: tuple(G[elt]) for elt in eqs}
    varinc = {elt: tuple(G[elt]) for elt in vrs}
    #allowed = copy.deepcopy(eqv)
    return eqv, varinc, M