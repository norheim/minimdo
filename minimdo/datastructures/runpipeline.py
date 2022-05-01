from graphutils import flat_graph_formulation, solver_children, Node, SOLVER, COMP, VAR
from operators import sort_scc, reorder_merge_solve
from workflow import get_f, generate_workflow, implicit_comp_name
from assembly import build_archi
from executeformulations import perturb_inputs, run_and_save_archi
import numpy as np

def get_solver_implicit_system(groups, tree, solver_idx):
    s = groups[str(Node(solver_idx, SOLVER))]
    name = implicit_comp_name((Node(idx, COMP) for idx in solver_children(tree[0], solver_idx)))
    return getattr(s, name)

def run_full(edges, tree, components, symb_mapping, xref=None, solver_rand_range=(0,1)):
    optres_all = []
    solver_idx = 2
    G = flat_graph_formulation(*edges)
    merge_order = sort_scc(G)
    nedges, ntree = reorder_merge_solve(edges, tree, merge_order, 1, True)
    lookup_f = get_f(components, nedges)
    workflow = generate_workflow(lookup_f, nedges, ntree, {2:{'maxiter':100}})
    prob, mdao_in, groups = build_archi(nedges, ntree, workflow)
    print(mdao_in)
    impl_system = get_solver_implicit_system(groups, ntree, solver_idx) # we use this for counting
    scc_guess_vars = [str(Node(vr, VAR)) for vr in solver_children(ntree[2], solver_idx)]
    rng = np.random.default_rng(1234)
    root_rand_range = (0,0)
    print(scc_guess_vars)
    x0_in = perturb_inputs(mdao_in, root_rand_range, scc_guess_vars, solver_rand_range, xref, rng)
    print(x0_in)
    optres, optres_save = run_and_save_archi(prob, x0_in, symb_mapping)
    optres_save.update({'count': impl_system.iter_count_apply}) 
    optres_all.append(optres_save)
    return optres, x0_in, prob