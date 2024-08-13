from trash.notation import default_sequence
from graph.graphview import tree_incidence
import matplotlib.pyplot as plt

def auto_incidence_tree(m, **kwargs):
    savefig = kwargs.pop('savefig', None)
    model = m.ref
    eqs, eqv, dout, dins = model.data_structures()
    sequence_m = default_sequence(m)
    fig, ax = tree_incidence(m, eqv, model.outset, sequence_m, **kwargs)
    if savefig:
        plt.sca(ax)
        plt.savefig(savefig, dpi=600, bbox_inches='tight')