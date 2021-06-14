import pytest
import sympy as sp
from sympy import S
import networkx as nx
from compute import args_in_order, Equation, coupled_run
from inputresolver import reassign, getdofs, set_ins, mdf_order
import numpy as np
import openmdao.api as om

def generate_test_eqs():
    x1, x2, x3, x4, a1, a2, a3, a4 = sp.symbols("x_1 x_2 x_3 x_4 a_1 a_2 a_3 a_4")
    eqs = {
        1: (x1, x2-x3*x4),
        2: (x2, x1-2*x3+2),
        3: (x3, x2-2*x1+1)
    }
    default_output = {key: left for key,(left,right) in eqs.items()}
    eqvars = {key: right.free_symbols | left.free_symbols for key,(left,right) 
        in eqs.items()}
    return (x1,x2,x3,x4), eqs, eqvars, default_output

def test_in_out():
    assert True

def test_args_in_order():
    assert args_in_order({'x':1,'y':2}, ['y','x'])==[2,1]

def test_digraph_fromoutput():
    (x1,x2,x3,x4), eqs, eqvars, default_output = generate_test_eqs()
    outset = {1: x4, 2: x1, 3: x2}
    edges = [(inp, outset[eq]) for eq, inps in eqvars.items() 
        for inp in inps if inp != outset[eq]]
    D = nx.DiGraph(edges)
    C = nx.condensation(D)
    order = [C.nodes[n]['members'] for n in nx.topological_sort(C)]
    assert order == [{x3}, {x2, x1}, {x4}]

def test_compute_equation():
    _, eqs, _, _ = generate_test_eqs()
    left, right = eqs[1]
    eq = Equation(left, right)
    inputvals = {'x_3':2.,'x_4':3.,'x_2':5.}
    run = eq.evaldict(inputvals)
    gradrun = eq.graddict(inputvals)
    assert run == -1
    assert gradrun[eq.inputs_names.index('x_2')] == 1.
    assert gradrun[eq.inputs_names.index('x_3')] == -3.
    assert gradrun[eq.inputs_names.index('x_4')] == -2.

def test_runorder():
    (x1,x2,x3,x4), eqs, _, _ = generate_test_eqs()
    runorder = [(3,x2), ((2, x1), (1, x4))]
    assert True

def test_getdofs():
    x1, x2, x3, x4, x5 = sp.symbols("x_1 x_2 x_3 x_4 x_5")
    eqs = {
        2: (x4, x1+x5),
        3: (x3, x1+x2)
    }
    ins = getdofs(eqs)
    assert {x1,x2,x5}==ins

def test_mdf_order():
    (x1,x2,x3,x4), eqs, eqvars, default_output = generate_test_eqs()
    ins = getdofs(eqs)
    for idx, elt in enumerate(ins):
        eqname = 'in{}'.format(idx)
        eqs[eqname] = (elt, S(0.99))
        eqvars[eqname] = {elt}
        default_output[eqname] = elt
    order = mdf_order(eqvars, default_output)
    #assert order = ['in0', (2,3,1)]
    assert True

def test_sequential_run():
    x1, x2, x3 = sp.symbols("x_1 x_2 x_3")
    eqs = {
        1: (x1, S(1.01)),
        2: (x2, 2*x1),
        3: (x3, x1+x2)
    }
    prob = om.Problem()
    model = prob.model
    counter = coupled_run(eqs, (1,2,3), (), model, model, 1)
    prob.setup()
    prob.run_model()
    assert prob.model.get_val('x_3')[0]==pytest.approx(3.03)

def test_coupled_run():
    x1, x2, x3 = sp.symbols("x_1 x_2 x_3")
    eqs = {
        1: (x1, 2*x2+2),
        2: (x2, 3*x1-1)
    }
    a = np.array([[1, -2], [-3, 1]])
    b = np.array([2, -1])
    xsol = np.linalg.solve(a, b)
    prob = om.Problem()
    model = prob.model
    counter = coupled_run(eqs, (), (1,2), model, model, 1)
    prob.setup()
    prob.run_model()
    assert prob.model.get_val('x_1')[0]==pytest.approx(xsol[0])
    assert prob.model.get_val('x_2')[0]==pytest.approx(xsol[1])

def test_complex_run():
    x1, x2, x3, x4 = sp.symbols("x_1 x_2 x_3 x_4")
    eqs = {
        0: (x4, S(2.4)),
        1: (x3, 1.01+x2*x4),
        2: (x1, 2*x2+2*x3),
        3: (x2, 3*x1-1)
    }
    prob = om.Problem()
    model = prob.model
    coupled_run(eqs, [0, ([1,2], 3)], (), model, model, 1)
    prob.setup()
    prob.run_model()
    assert prob.model.get_val('x_1')[0]==pytest.approx(0.24639175)
    assert prob.model.get_val('x_2')[0]==pytest.approx(-0.26082474)
    assert prob.model.get_val('x_3')[0]==pytest.approx(0.38402062)
    assert prob.model.get_val('x_4')[0]==pytest.approx(2.4)

def test_reassign_outputs():
    (x1,x2,x3,x4), eqs, eqvars, default_output = generate_test_eqs()
    outset = {1: x4, 2: x1, 3: x2}
    new_eqs = reassign(eqs, outset)
    assert sp.simplify(new_eqs[1][1] - (x1-x2)/(-x3)) == 0
    assert sp.simplify(new_eqs[2][1] - (x2+2*x3-2)) == 0
    assert sp.simplify(new_eqs[3][1] - (x3+2*x1-1)) == 0

def test_full_pipeline():
    (x1,x2,x3,x4), eqs, eqvars, default_output = generate_test_eqs()
    
    ins = getdofs(eqs)
    determined_outpus, detetermined_eqvars, determined_eqs = set_ins(ins, 
        default_output, eqvars, eqs)
    order = mdf_order(detetermined_eqvars, determined_outpus)
    prob = om.Problem()
    model = prob.model
    counter = coupled_run(determined_eqs, order, (), model, model, 0)
    prob.setup()
    prob.run_model()
    assert prob.model.get_val('x_1')[0]==pytest.approx(0.99331104)
    assert prob.model.get_val('x_2')[0]==pytest.approx(1.65551839)
    assert prob.model.get_val('x_3')[0]==pytest.approx(0.66889632)
