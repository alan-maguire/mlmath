#!/usr/bin/python3

import GaussianElimination as g
import numpy as np

class UnexpectedResult(Exception):
    pass

def Validate(A, b, debug=False):
    g.DebugPrint("Solve Ax = b, where A = ", debug)
    g.DebugPrint(A, debug)
    g.DebugPrint("b = ", debug)
    g.DebugPrint(b, debug)
    x = g.GaussianElimination(A,b)
    g.DebugPrint("x = ", debug)
    g.DebugPrint(x, debug)
    expected = np.linalg.solve(A,b)
    g.DebugPrint("expected = ", debug)
    g.DebugPrint(expected, debug)
    if np.allclose(x,expected) == False:
        raise UnexpectedResult(x, expected)

A = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
b = np.array([4,5,6])

Validate(A,b)
