import pytest
import sympy as sp
from sympy import S
import networkx as nx
from compute import args_in_order, Equation, coupled_run
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


def test_sequential_run():
    x1, x2, x3 = sp.symbols("x_1 x_2 x_3")
    eqs = {
        1: (x1, S(1.01)),
        2: (x2, 2*x1),
        3: (x3, x1+x2)
    }
    prob = om.Problem()
    model = prob.model
    counter = coupled_run(eqs, (1,2,3), (), model, 1)
    prob.setup()
    prob.run_model()
    assert prob.model.get_val('x_3')[0]==pytest.approx(3.03)

def test_coupled_run2():
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
    counter = coupled_run(eqs, (), (1,2), model, 1)
    prob.setup()
    prob.run_model()
    assert prob.model.get_val('x_1')[0]==pytest.approx(xsol[0])
    assert prob.model.get_val('x_2')[0]==pytest.approx(xsol[1])