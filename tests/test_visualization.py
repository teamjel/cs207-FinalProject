import pytest
import numpy as np
from autodiff.node import *
from autodiff.operators import *
from autodiff.visualization import *

def test_composition_result():
    a = Variable("a")
    b = Variable("b")
    c = Variable("c")
    d = Variable("d")
    y = cos((-a)**2/c) - 4*sin(b) * log(exp(d) + 1, 10)

    graph = y.get_comp_graph()
    assert(len(graph.added_nodes) == 19)
    assert(len(graph.added_edges) == 18)

    table = y.get_comp_table()
    assert(table.shape == (19,7))