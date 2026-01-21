# Basic Linear Algebra

## Vectors

Vectors are represented as a 1-dimensional ordered array of numbers,
usually in columnar form:

$$
x = \begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

Vectors are often conceptualized geometrically as a line in
n-dimensional space, with each element being the length along
the relevant axis.

In python, we create a vectors using numpy

```
$ python3
>>> import numpy as np
>>> x = np.array([[1],[2],[3]])
>>> x
array([[1],
       [2],
       [3]])
>>> 
```

We can index a vector element as
$x_i$ : on python we start from index 0:

```
>>> x[0]
array([1])
```

x is a column vector; to create row vector

$$
x = \begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
$$

```
>>> x2 = np.array([[1,2,3]])
>>> x2
array([[1, 2, 3]])

```

We can flip a matrix from columnar to row form (or vice
versa) using the transpose operation:

$$
x^T = \begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
$$

In python:

```
>>> z = x.T
>>> z
array([[1, 2, 3]])
```

We can get the size of a vector via `np.size()`; `np.shape()`
can be used to get the dimensions (`np.size()`) simply returns
the number of elements:

```
>>> np.shape(x)
(3, 1)
>>> np.size(x)
3
```

Addition, subtraction and scalar multiplication all operate
on each member; for example

$$
x = \begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

$$
y = \begin{bmatrix}
4 \\
5 \\
6
\end{bmatrix}
$$

$$
x + y = \begin{bmatrix}
5 \\
7 \\
8
\end{bmatrix}
$$

There are many different ways to represent the size (norm)
of a vector; the $L^1$ norm is simply the sum of the elements.

The most commonly used is the $L^2$ norm; it matches the usual
Pythagorean length:

$$
||x||_2 = \sqrt{\sum_i(|x_i|^2)}
$$

```
>>> np.linalg.norm(x)
3.7416573867739413
```

This formula also generalizes beyond 2 to any norm p, using
the p'th root and p'th power.  From this we see the $L^1$
norm is simply the sum of the absolute value of the elements.

We can multiply two vectors using the dot product operation;
each corresponding element is multiplied and then summed.
The convention is to multiply row by column, so we take the
transpose of x if it is a column vector:

$$
x^T . y = (1)(4)+(2)(5)+(3)(6) = 32
$$

In python we can either use `np.dot()` or the equivalent
`@` operator:

```
>>> x = np.array([[1],[2],[3]])
>>> y = np.array([[4],[5],[6]])
>>> x.T@y
array([[32]])
>>> np.dot(x.T,y)
array([[32]])
```

Sometimes the dot product of x and y is written as $<x,y>$.

Notice that the square of the $L^2$ norm is equivalent to
$x^T.x$ .  The squared $L^2$ norm is often used in machine
learning as it is easy to calcuate, but one problem with it
is it is very small when the vector elements are close to 0
since the square of a small number is a smaller number.  In
such cases the $L^1$ norm can be used as a measure of vector
size.

The dot product can also be rewritten as a product
of $L^2$ norms and the angle $\theta$ between the vectors:

$$
x^T.y = ||x||.||y||\cos(\theta)
$$

One consequence of this is that perpendicular vectors have
a dot product of 0, while parallel ones have a dot product
of $||x||.||y||$ (the product of their $L^2$ norms).  In fact,
the dot product is equivalent to magnitude of projection of vector
x onto vector y multiplied by the length of y.  We can see
this using Pythagorus theorem; drop a perpendicular from
x onto y; the adjacent to angle $\theta$ is the projection
$||x||.\cos(\theta)$ .  All that is missing is to multiply
by $||y||$ and we have the dot product formula.

This also makes it much easier to compute projections;
to compute the projection x onto y. If

$$
x^T.y = ||proj(x,y)||.||y||
$$

$$
proj(x, y) = \frac{x^T.y}{||y||} y
$$

Note tha additional y vector since the value in front of it
simply gives the length of the projection along y; we still
need the y vector to have the magnitude and direction of the
projection vector.

This geometric interpretation also tells us when the projection
of x onto y is positive, the dot product of $x^T.y$ will be
positive, while if it is negative the dot product will be too.
And of course if they are orthogonal the projection is zero.

## Matrices

A matrix is a 2-dimensional mXn array of numbers, where

- m is the number of rows
- n is the number of columns

Here is a 2x3 (2 rows, 3 columns) matrix A:

$$
A = \begin{bmatrix}
1 & 2 & 3 \\
7 & 8 & 9
\end{bmatrix}
$$

In python we define it as follows:

```
>>> A = np.array([[1,2,3],[7,8,9]])
>>> A
array([[1, 2, 3],
       [7, 8, 9]])
>>> A.T
array([[1, 7],
       [2, 8],
       [3, 9]])
>>> np.shape(A)
(2, 3)
```

We index elements via row i, column j as $A_{i,j}$ ,
and in python we can select a whole row or column using `:`
instead of an index:

```
>>> A[1,2]
9
>>> A[:,1]
array([2, 8])
```

Addition, subtraction and scalar multiplication work similarly
to vectors.

Matrix multiplication is achieved by taking the dot product of
the rows of the left-hand matrix with the rows of the right.
Each m,n element is computed as the dot product of the m'th
row with the n'th column.  This is a generalization of the
process of taking the dot product of a matrix with a column
vector.

For this to work the number of columns of the left hand matrix
must match the number of rows on the right, so if we have
an (mXn matrix) (pXq matrix)

- n must equal p (inside rule)
- resulting matrix will be mXq

Example

$$
P = \begin{bmatrix}
5 & 2 & 0 \\
7 & 3 & 1 \\
\end{bmatrix}
$$

$$
Q = \begin{bmatrix}
2 & 1 \\
7 & 3 \\
8 & 0
\end{bmatrix}
$$

Dimensions of P, Q are 2x3 and 3x2 respectively - this tells us
that multiplication is possible since the number of columns of
P matches the number of rows of Q (inside rule 3 == 3) and that
the resulting matrix will be 2x2 (outside rule).

$$
P.Q = \begin{bmatrix}
24 & 11 \\
43 & 16
\end{bmatrix}
$$

$P.Q_{1,1}$ (0,0 in python) is computed by taking dot product of
first row of P with first column of Q; i.e.

$$
(5)(2) + (2)(7) + (0)(8) = 24
$$

Similarly to vectors, we can view matrices geometrically.  Specifically
we can view them as comprising linear transformations of vectors.
A linear transformation takes each vector in a space to another vector
in the transformed space.  A good way to visualize the effects of the
transformation is to plot where (0,0), (1, 0), (0,1) and (1,1) are
transformed to by matrix multiplication.  The square that comprises
these points is often transformed to a parallelogram which stretches
in one dimension but can shrink in another.  In fact the columns
of the matrix are actually the transformations of the unit basis
vectors (1,0), (0,1).  For example the matrix

$$
A = \begin{bmatrix}
3 & 2 \\
1 & 1 \\
\end{bmatrix}
$$

transforms $I_1$ (1,0) -> (3,1), and $I_2$ (0,1) -> (2,1).  We can see
that because the dot product of the rows of A with the identity vectors
preserves A's column values:

$$
A.I_1 = \begin{bmatrix}
(3)(1)+(2)(0) \\
(1)(1)+(1)(0)
\end{bmatrix}
$$

The identity matrix can be viewed as the identity transformation;
it maps vectors in $R^n$ to themselves:

$$
I = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

Matrix multiplications can in this way be viewed as sequences of
linear transformations. Because for matrices/vectors

$$A.(B.x) = (A.B)x$$

multiplying A.B by x is equivalent to first applying the linear
transformation B to x and then applying A to the result.

Inverting a matrix - where $A^{-1}$ for square matrix A is
defined as the matrix that when multiplied by A (on either the
left or right) gives us the identity matrix; i.e.

$$A^{-1}A = AA^{-1} = I$$

So the inverse is the matrix equivalent to 1/n for scalar n -
$(n)(1/n) = 1$.  With matrices, instead of 1 we have the
identity matrix.

We will see later that non-square matrices - which are common
in linear algebra since we often have differing numbers of
data attributes (rows) and data samples (columns) - have
a pseudo-inverse process called Singular Value Decomposition (SVD).
Rather than being symmetric it requires both a left and
right matrix.

To compute the inverse in python, use `np.linalg.inv()`.

```
>>> D = np.array([[1,3,4],[2,7,2],[5,4,0]])
>>> np.linalg.inv(D)
array([[ 0.09302326, -0.18604651,  0.25581395],
       [-0.11627907,  0.23255814, -0.06976744],
       [ 0.31395349, -0.12790698, -0.01162791]])
>>> np.linalg.inv(D)@D
array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 1.38777878e-17,  1.00000000e+00,  0.00000000e+00],
       [ 1.73472348e-18, -2.08166817e-17,  1.00000000e+00]])
>>> 
```

This is really the identity matrix; the off-diagonal nonzero values are
very close to 0.

We can round to 10 decimal places via `np.round()`

```
>>> np.round(np.linalg.inv(D)@D,decimals=10)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0., -0.,  1.]])
```

Computing the inverse for a square matrix becomes a set of
simultaneous equations to solve.

For example 

$$
P = \begin{bmatrix}
5 & 2 \\
1 & 2
\end{bmatrix}
$$

$$
P.P^{-1} = I
$$

$$
=> \\
\begin{bmatrix}
5 & 2 \\
1 & 2
\end{bmatrix} \\
. \\
\begin{bmatrix}
a & b \\
c & d 
\end{bmatrix} \\
= \\
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

$$
=> 5a+2c = 1
$$

$$
   1a+2c = 0
$$

$$
   5b+2d = 0
$$

$$
   1b+2d = 1
$$

$$
=> a = 0.25, c = -0.125, b = -0.25, d = 0.625
$$

$$
=> P^{-1} = \begin{bmatrix}
0.25 & -0.25 \\
-0.125 & 0.625
\end{bmatrix}
$$

We will check our work:

```
>>> P = np.array([[5, 2],[1,2]])
>>> np.linalg.inv(P)
array([[ 0.25 , -0.25 ],
       [-0.125,  0.625]])

```

Not all square matrices have inverses, just as not all numbers
do (0 does not have a multiplicative inverse since 1/0 is not
defined).

- if a matrix does not have an inverse it is said to be singular
- invertible matrices are non-singular

When considering matrices as representing as systems of equations,
a non-singular system is one with a unique solution; e.g.

$$
a + b = 10
$$

$$
2a + b = 15
$$

$$
=> \begin{bmatrix}
1 & 1 \\
2 & 1 \\
\end{bmatrix} \\
. \\
\begin{bmatrix}
a \\
b \\
\end{bmatrix} \\
= \\
\begin{bmatrix}
10 \\
15 \\
\end{bmatrix}
$$

Here we have unique solution

$$ a = 5, b = 5 $$

We can also have cases where there are infinitely many solutions;
for example:

$$
a + b = 10
$$

$$
2a + 2b = 20
$$

and no solution (where the equations are inconsistent):

$$
a + b = 10
$$

$$
2a + 2b = 30
$$

Note that all that matters for singularity is the coefficient matrix;
the results determine whether we have no or infinite solutions, but
in both cases the matrix is singular.

There is a quick test for singularity; computing the determinant.

foo
bar
baz
