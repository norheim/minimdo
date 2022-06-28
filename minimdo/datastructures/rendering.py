from datastructures.graphutils import (edges_to_Ein_Eout, all_variables, all_components, dfs_tree, end_components, root_solver, solver_children, VAR, COMP)
from datastructures.workflow import order_from_tree, ENDCOMP, COMP
import matplotlib.patches as patches
import numpy as np
from representations import plot_incidence_matrix

def sequence_permutation_from_order(order, Ein, Eout, Vtree, Ftree):
    Fend = end_components(Eout)
    sequence = [fx for _,fx,s in order]
    solvefor = Eout.copy() # assumes only one output
    f_at_level = {}
    for solvevar, solver in Vtree.items():
        if solver not in f_at_level:
            f_at_level[solver] = (elt for elt in solver_children(Ftree, solver) if elt in Fend)
        fend = next(f_at_level[solver])
        solvefor[fend] = (solvevar,)
    permutation = [solvefor[s][0] for s in sequence if solvefor.get(s, False) and solvefor[s][0] is not None]
    all_vars_random_order = all_variables(Ein, Eout)
    permutation += [var for var in all_vars_random_order if var not in permutation]
    return sequence, permutation

def all_comps_below(order, Ftree, Stree, branch):
    applicable_solvers = dfs_tree(Stree, branch)
    allcomps = [(idx, end_or_inter==COMP or Ftree[fx]!=branch) for idx, (end_or_inter, fx, solver) in enumerate(order) if solver in applicable_solvers]
    allcompsout, inter, end = tuple(), tuple(), tuple()
    for idx, inter_or_end in allcomps:
        elt = (idx, False)
        if inter_or_end:
            inter += (elt,)
        else:
            end += (elt,)
        allcompsout += (elt,)
    return allcompsout, inter, end

def solver_artifact_iterator(tree, order):
    Ftree, Stree, _ = tree
    for branch in dfs_tree(Stree, root_solver(tree)):
        yield all_comps_below(order, Ftree, Stree, branch)

def incidence_artifacts(edges, tree, generatesolverstruct=False):
    Ein, Eout = edges_to_Ein_Eout(edges)
    Ftree, Stree, Vtree = tree
    order = order_from_tree(Ftree, Stree, Eout, includesolver=False, mergeendcomp=False)
    sequence, permutation = sequence_permutation_from_order(order, Ein, Eout, Vtree, Ftree)
    solver_artifact = solver_artifact_iterator(tree, order) if generatesolverstruct else None
    return sequence, permutation, Ein, Eout, solver_artifact

def generate_incidence_matrix(Ein, Eout, sequence, permutation, diagonalgray=True):
    A = np.zeros((len(sequence), len(permutation)))
    for idx, fxid in enumerate(sequence):
        varsineq = Ein[fxid]+tuple(elt for elt in Eout[fxid] if elt is not None)
        for var in varsineq:
            col = permutation.index(var)
            color = 0.5 if (var in Eout[fxid] and diagonalgray) else 1.
            A[idx,col] = color
    return A

def getpatchwindow(lst):
    ulcorner = lst[0]
    size = max(lst)-ulcorner+1
    return ulcorner,size

def getpatchwindow_end(lst):
    equivalent_lst, matched_ends = zip(*lst)
    ulcorner,size = getpatchwindow(equivalent_lst)
    return ulcorner,size-sum(matched_ends)

def get_patches(solver_iterator):
    allpatches=[]
    for allcomps, inter, end in solver_iterator:
        patchparam = getpatchwindow_end(allcomps)
        patchparam_inter = [getpatchwindow_end(inter)] if inter else []
        patchparam_end = [getpatchwindow_end(end)] if end else []
        allpatches.append(patchparam)
        allpatches += patchparam_inter
        allpatches += patchparam_end
    return allpatches

def plot_patches(ax, allpatches, patchwidth=2):
    for ulcorner, size in allpatches:
        rect = patches.Rectangle((ulcorner,ulcorner), size, size, linewidth=patchwidth, edgecolor='#737373', facecolor='none')
        ax.add_patch(rect)

def render_incidence(edges, tree, namingfunc, displaysolver=True, **kwargs):
    patchwidth = kwargs.pop('patchwidth', 2)
    sequence, permutation, Ein, Eout, solver_iterator = incidence_artifacts(edges, tree, displaysolver)
    A = generate_incidence_matrix(Ein, Eout, sequence, permutation)
    column_labels = ['${}$'.format(namingfunc(var, VAR)) for var in permutation]
    row_labels = ['${}$'.format(namingfunc(fx, COMP)) for fx in sequence]
    fig, ax =plot_incidence_matrix(A, column_labels, row_labels, **kwargs)
    if displaysolver:
        allpatches=get_patches(solver_iterator)
        plot_patches(ax, allpatches, patchwidth)
    return fig, ax