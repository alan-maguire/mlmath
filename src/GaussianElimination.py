#!/usr/bin/python3

import numpy as np

class MatrixIsSingular(Exception): pass

class MatrixIsNotSquare(Exception): pass

def DebugPrint(msg, debug=False):
    if debug:
        print(msg)
 
#
# Reduce row r by first subtracting multiples of rows above
# such that we have 0s to the left of A[r,r].  Because
# A[r,r] can be 0,  we add rows below to make it non-zero.
#
# Parameters:   A: (m+1)Xm augmented matrix
#               r: row to reduce
# Returns:      A with row reduced (1 at A[r,r], zeros to the left of A[r,r]
#
def RowReduce(A, r, debug=False):
    DebugPrint('\nReducing row ' + str(r) + ' of:', debug)
    DebugPrint(A, debug)
    for i in range(r):
        DebugPrint('\nr_' + str(r) + ' -> r_' + str(r) + ' - (' + str(A[r,i]) + ')r_' + str(i) + ':\n', debug)
        A[r] = A[r] - A[r, i] * A[i]
        DebugPrint(A, debug)
    for j in range(r+1,np.shape(A)[0]):
        if np.allclose(A[r,r],0) != True:
            break
        DebugPrint('\nr_' + str(r) + ' -> r_' + str(r) + 'r_' + str(j) + ':\n', debug)
        A[r] = A[r] + A[j]
        DebugPrint(A, debug)
    if A[r,r] == 0:
         raise MatrixIsSingular()
    DebugPrint('\nr_' + str(r) + ' -> (r_' + str(r) + ')/' + str(A[r,r]) + ':\n', debug)
    A[r] = A[r] / A[r,r]
    DebugPrint(A, debug)
    return A

#
# Back-substitute multiples of row below to get 0s to the right of A[r,r]
#
# Parameters:   A: row-reduced augmented matrix in row echelon form
#               r: row to back-substitute
# Returns:      A with back-substition (1 at A[r,r], zeros to the right of A[r,r]
#
def BackSubstitute(A, r, debug=False):
    DebugPrint('\nBack-substituting row ' + str(r) + ' of: ', debug)
    DebugPrint(A, debug)
    for i in range(r + 1, np.shape(A)[0]):
        DebugPrint('\nr_' + str(r) + ' -> r_' + str(r) + ' - (' + str(A[r, i]) + ')r_' + str(i) + ':\n', debug)
        A[r] = A[r] - A[r, i] * A[i]
        DebugPrint(A, debug)
    return A

#
# Carry out Gaussian Elimination on mXm matrix A with mx1 vector b to solve
# Ax = b
#
# First carry out row reduction to get to row-echelon form, then
# back-substitute to get to reduced row-echelon form.  At this point,
# return the rightmost column as the solutions of x.
#
# Parameters:   A: mXm matrix
#               b: mX1 vector
# Returns:      mX1 vector solution
#
def GaussianElimination(A, b, debug=False):

    m = np.shape(A)[0]
    n = np.shape(A)[1]
    if m != n:
        raise MatrixIsNotSquare()
    if m != np.shape(b)[0]:
        raise VectorMatrixMismatch() 
    Aug = np.hstack((A, np.reshape(b, (m,1))))
    Aug = Aug.astype(float)

    for i in range(m):
        Aug = RowReduce(Aug, i, debug)
    for i in reversed(range(m)):
        Aug = BackSubstitute(Aug, i, debug)
    return Aug[:,n]

