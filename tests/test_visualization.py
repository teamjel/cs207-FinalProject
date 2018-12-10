import pytest
import numpy as np
from autodiff.node import *
from autodiff.operators import *
from autodiff.visualization import *

# Test unary operators: negation, log, sin, cos, tan, power, exponential.
def test_unary_result():
    # function: a, derivative: 1
    a = Variable("a")
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(a)
    assert(len(graph.added_nodes) == 1)
    assert(len(graph.added_edges) == 0)
    # function: -b
    b = Variable("b")
    negB = -b
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(negB)
    assert(len(graph.added_nodes) == 2)
    assert(len(graph.added_edges) == 1)
    # function: sin(d)
    d = Variable("d")
    sinD = sin(d)
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(sinD)
    assert(len(graph.added_nodes) == 2)
    assert(len(graph.added_edges) == 1)
    # function: cos(d), derivative: -sin(d)
    e = Variable("e")
    cosE = cos(e)
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(cosE)
    assert(len(graph.added_nodes) == 2)
    assert(len(graph.added_edges) == 1)
    # function: sqrt(f), derivative: 1/2(x)^(-1/2)
    f = Variable("f")
    sqrtF = sqrt(f)
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(sqrtF)
    assert(len(graph.added_nodes) == 2)
    assert(len(graph.added_edges) == 1)
    # function: e^h, derivative: e^h
    h = Variable("h")
    expH = exp(h)
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(expH)
    assert(len(graph.added_nodes) == 2)
    assert(len(graph.added_edges) == 1)

def test_binary_errors():
    # function: log(c)
    c = Variable("c")
    logC = log(c)
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(logC)
    assert(len(graph.added_nodes) == 3)
    assert(len(graph.added_edges) == 2)
    log10C = log(c, 10)
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(log10C)
    assert(len(graph.added_nodes) == 3)
    assert(len(graph.added_edges) == 2)
    # function: g**3
    g = Variable("g")
    powerG = g**3
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(powerG)
    assert(len(graph.added_nodes) == 3)
    assert(len(graph.added_edges) == 2)
    # function: g**h
    g = Variable("g")
    h = Variable("h")
    powerGH = g**h
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(powerGH)
    assert(len(graph.added_nodes) == 3)
    assert(len(graph.added_edges) == 2)
    # function: a + b
    a = Variable("a")
    b = Variable("b")
    c = a + b
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(c)
    assert(len(graph.added_nodes) == 3)
    assert(len(graph.added_edges) == 2)
    # function: a - b
    e = a - b
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(e)
    assert(len(graph.added_nodes) == 3)
    assert(len(graph.added_edges) == 2)
    # function: a * b
    g = a * b
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(g)
    assert(len(graph.added_nodes) == 3)
    assert(len(graph.added_edges) == 2)
    # function a / b
    i = a / b
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(i)
    assert(len(graph.added_nodes) == 3)
    assert(len(graph.added_edges) == 2)

def test_composition_result():
    a = Variable("a")
    b = Variable("b")
    c = Variable("c")
    d = Variable("d")
    y = cos((-a)**2/c) - 4*sin(b) * log(exp(d) + 1, 10)
    graph = ADDigraph()
    graph.add_node_subgraph_to_plot_graph(y)
    assert(len(graph.added_nodes) == 19)
    assert(len(graph.added_edges) == 18)
