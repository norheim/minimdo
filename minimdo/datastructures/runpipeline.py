from graphutils import flat_graph_formulation, solver_children, root_solver, Node, SOLVER, COMP
from execution import generate_components_and_residuals
from datastructures.workflow import namefromid
from operators import sort_scc, reorder_merge_solve
from workflow import get_f, order_from_tree, default_solver_options, mdao_workflow, implicit_comp_name
from workflow_mdao import mdao_workflow_with_args
from assembly import build_archi
from executeformulations import perturb_inputs, run_and_save_archi
import numpy as np

def get_solver_implicit_system(groups, tree, solver_idx):
    s = groups[str(Node(solver_idx, SOLVER))]
    name = implicit_comp_name((Node(idx, COMP) for idx in solver_children(tree[0], solver_idx)))
    return getattr(s, name)

def nestedform_to_mdao(edges, tree, components, solver_options, comp_options, var_options, nodetyperepr, mdf=True):
    namingfunc = namefromid(nodetyperepr)
    G = flat_graph_formulation(*edges)
    merge_order = sort_scc(G)
    merge_parent = root_solver(tree) # all merged components will have this solver as the parent
    ordered_edges, ordered_tree = reorder_merge_solve(edges, tree, merge_order, merge_parent, mdf)
    sequence = order_from_tree(ordered_tree[0], ordered_tree[1], ordered_edges[1])
    solvers_options = default_solver_options(ordered_tree, solver_options)
    wf = mdao_workflow(sequence, solvers_options, comp_options, var_options)
    all_components = generate_components_and_residuals(components, ordered_edges)
    lookup_f = get_f(all_components, ordered_edges)
    wfmdao = mdao_workflow_with_args(wf, lookup_f, namingfunc)
    prob, mdao_in, groups = build_archi(ordered_edges, ordered_tree, wfmdao)
    return prob, mdao_in, groups

    # print(mdao_in)
    # impl_system = get_solver_implicit_system(groups, ntree, solver_idx) # we use this for counting
    # scc_guess_vars = [str(Node(vr, VAR)) for vr in solver_children(ntree[2], solver_idx)]
    # rng = np.random.default_rng(1234)
    # root_rand_range = (0,0)
    # print(scc_guess_vars)
    # x0_in = perturb_inputs(mdao_in, root_rand_range, scc_guess_vars, solver_rand_range, xref, rng)
    # print(x0_in)
    # optres, optres_save = run_and_save_archi(prob, x0_in, symb_mapping)
    # optres_save.update({'count': impl_system.iter_count_apply}) 
    # optres_all.append(optres_save)
    # return optres, x0_in, prob