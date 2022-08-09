from collections import defaultdict
from datastructures.graphutils import (edges_to_Ein_Eout, all_variables, all_components, dfs_tree, end_components, root_solver, solver_children, VAR, COMP, SOLVER, namefromid)
from datastructures.workflow import order_from_tree, ENDCOMP, COMP
import matplotlib.patches as patches
import numpy as np
from representations import plot_incidence_matrix
import matplotlib.pyplot as plt

def sequence_permutation_from_order(order, Ein, Eout, Vtree, Ftree):
    Fend = end_components(Eout)
    sequence = [fx for _,fx,s in order]
    solver_vars = defaultdict(list)
    for solvevar, solver in Vtree.items():
        solver_vars[solver].append(solvevar)
    solver_vars = {key:sorted(val) for key,val in solver_vars.items()}

    permutation = []
    for comp in sequence:
        permutation += Eout[comp] if Eout[comp][0] != None else []
        if comp in Fend:
            parent_solver = Ftree[comp]
            if parent_solver in solver_vars:
                permutation += solver_vars[parent_solver]
                solver_vars[parent_solver] = []
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

def generate_windows(upper_left_row, upper_left_col, row_inter, row_total, col_inter, col_total):
    row_end = row_total-row_inter
    col_end = col_total-col_inter
    rect_inter = ((upper_left_row,upper_left_col), (row_inter,col_inter))
    rect_end = ((upper_left_row+row_inter,upper_left_col+col_inter), (row_end,col_end))
    rect_full = ((upper_left_row,upper_left_col), (row_total,col_total))
    return rect_inter, rect_end, rect_full

def generate_patches(order, Stree, Vtree, branch):
    all_solvers_below = dfs_tree(Stree, branch)
    row_inter,col_inter,row_total,col_total = 0,0,0,0
    solvers_visited = set()
    beginnig_flag = True
    for comp_type,_,parent in order:
        if parent in all_solvers_below and beginnig_flag:
            upper_left_row, upper_left_col = row_total, col_total
            row_inter,col_inter,row_total,col_total = 0,0,0,0 # reset counter
            beginnig_flag = False
        if parent not in all_solvers_below and not beginnig_flag:
            break
        row_total+=1
        if comp_type == COMP:
            row_inter+=1
            col_inter+=1
            col_total+=1
        else:
            if parent != branch:
                row_inter+=1
            if parent not in solvers_visited:
                number_of_vars=len(list(solver_children(Vtree,parent)))
                if parent != branch:
                    col_inter+=number_of_vars
                col_total+=number_of_vars
                solvers_visited.add(parent)
    return generate_windows(upper_left_row, upper_left_col, row_inter, row_total, col_inter, col_total)

def solver_artifact_iterator(tree, order):
    _, Stree, Vtree = tree
    for branch in dfs_tree(Stree, root_solver(tree)):
        yield generate_patches(order, Stree, Vtree, branch)

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

def plot_patches(ax, allpatches, patchwidth=2):
    for ulcorner, (row_size, col_size) in allpatches:
        rect = patches.Rectangle(ulcorner, col_size, row_size, linewidth=patchwidth, edgecolor='#737373', facecolor='none')
        ax.add_patch(rect)

def is_end_comp(Eout, fx, dispendcomp=True):
    return Eout[fx] == (None,) and dispendcomp
    
def render_incidence(edges, tree, namingfunc=None, displaysolver=True, rawvarname=False, **kwargs):
    if namingfunc is None:
        varnameformat = '{}' if rawvarname else'x_{{{}}}'
        nodetyperepr = {VAR: varnameformat, COMP: 'f_{{{}}}', SOLVER: 's_{{{}}}'}
        namingfunc = namefromid(nodetyperepr)
    patchwidth = kwargs.pop('patchwidth', 2)
    sequence, permutation, Ein, Eout, solver_iterator = incidence_artifacts(edges, tree, displaysolver)
    A = generate_incidence_matrix(Ein, Eout, sequence, permutation)
    column_labels = ['${}$'.format(namingfunc(var, VAR)) for var in permutation]
    dispendcomp = kwargs.pop('dispendcomp', False)
    row_labels = ['${}$'.format('h_{}'.format(fx) if is_end_comp(Eout, fx, dispendcomp) else namingfunc(fx, COMP) ) for fx in sequence]
    fig, ax =plot_incidence_matrix(A, column_labels, row_labels, **kwargs)
    if displaysolver:
        allpatches=(patch for patches in solver_iterator for patch in patches)
        plot_patches(ax, allpatches, patchwidth) 
    figname = kwargs.pop('figname', False)   
    if kwargs.pop('save',False) and figname:
        dpi = kwargs.pop('dpi', 200) 
        plt.savefig(figname, dpi=dpi, bbox_inches='tight')
    return fig, ax