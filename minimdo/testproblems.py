from networkx.algorithms.bipartite import random_graph as bipartite_random_graph
import networkx as nx
import numpy as np
from datastructures.graphutils import edges_E, flat_graph_formulation
from datastructures.tearing import dir_graph

def generate_random_prob(n_eqs, n_vars, seed=8, sparsity=1.0):
    n_nodes = n_eqs + n_vars
    p = n_nodes/(n_eqs*n_vars + 1.0e-17)*sparsity
    p = min(p, 1.0 - 1.0e-6) 
    G = bipartite_random_graph(n_eqs, n_vars, p, seed)
    #G = nx.bipartite.gnmk_random_graph(n_eqs, n_vars, int(0.2*n_vars*n_eqs), seed)
    eqids = list(range(n_eqs))
    bip_varids = list(range(n_eqs, n_eqs+n_vars))
    rng = np.random.default_rng(seed)

    # Make sure that there is at least one matching possible
    M = dict(zip(rng.permutation(eqids), rng.permutation(bip_varids)))
    # Complete the incidence matrix with the matching
    for node in eqids:
        G.add_edge(node, M[node])
    # Make sure all input variables have at least one connection
    isolated_nodes = [n for n in bip_varids if len(G[n])==0]
    # Make sure there is no independent variable
    
    for varnode in isolated_nodes:
        G.add_edge(varnode, rng.choice(eqids))
    eqv = {elt: tuple(G[elt]) for elt in eqids}
    varinc = {elt: tuple(G[elt]) for elt in bip_varids}
    #allowed = copy.deepcopy(eqv)
    return eqv, varinc, M

def random_problem_with_artifacts(m,n,seed,sparsity):
    seed = int(seed) # required for the way we generate random problems
    eq_incidence, var_incidence, outset = generate_random_prob(m, n, seed, sparsity)
    edges_varonleft = edges_E(eq_incidence)
    eqnidxs = eq_incidence.keys()
    varidxs = var_incidence.keys()
    D = nx.DiGraph(dir_graph(edges_varonleft, eqnidxs, outset.items()))
    kwargs = {
        'm': m,
        'n':n,
        'seed':seed,
        'sparsity':sparsity,
        'eq_incidence':eq_incidence,
        'var_incidence':var_incidence,
        'edges_varonleft':edges_varonleft,
        'outset': outset,
        'eqnidxs':eqnidxs,
        'varidxs':varidxs,
        'D':D
    }
    return kwargs