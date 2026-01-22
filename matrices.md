# Matrices

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

$P.Q_{1,1}$ ([0,0] in python) is computed by taking dot product of
first row of P with first column of Q; i.e.

$$
(5)(2) + (2)(7) + (0)(8) = 24
$$

In general to compute the element $pq_{x,y}$ in the product $P.Q$

$$
pq_{x,y} = \sum_{i=1}^n p_{x,i}.q_{i,y}
$$

where n is the number of columns of p == number of rows of q.

Properties of matrix multiplication

- associative: $A.(B.C) = (A.B).C$
- distributive: $A(B + C) = A.B + A.C$
- non-commutative: $A.B \ne B.A$
- multiplicative identity: $A.I = I.A = A$
- multiplicative property of 0: $A.0 = 0$
- scalar multiplication $c(A.B) = (cA)B = A(cB)$ for scalar c
- transpose of a product ${(A.B)}^T = B^T.A^T$
- if $A.B = 0$ it does not necessarily mean A=0 or B=0

We can prove associativity via the definition of multiplication:

$$
ab_{x,y} = \sum_{i=1}^n a_{x,i}.b_{i,y}
$$


This gives us Eq 1:

$$
(ab)c_{x,z} = \sum_{j=1}^m (\sum_{i=1}^n a_{x,i}.b_{i,y}).c_{j,z}
$$

Meanwhile:

$$
bc_{y,z} = \sum_{j=1}^m b_{y,j}.c_{j,z}
$$

And Eq 2:

$$
a(bc)_{x,z} = \sum_{i=1}^n a_{x,i} (\sum_{j=1}^m b_{y,j}.c_{j,z})
$$

Rearranging, Eq 1 and 2 are equivalent; all that changes is the order
of the multiplication.

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

We can prove the left and right inverses are equivalent;
starting with

$$AA^{-1} = I$$

We multiply both sides by A:

$$AA^{-1}A = A$$

$$=> A(A^{-1}A) = A$$

This tells us the matrix product $A^{-1}A$ must be $I$.

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
=> \begin{bmatrix}
5 & 2 \\
1 & 2
\end{bmatrix} . \begin{bmatrix}
a & b \\
c & d 
\end{bmatrix} = \begin{bmatrix}
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
\end{bmatrix} . \begin{bmatrix}
a \\
b \\
\end{bmatrix} = \begin{bmatrix}
10 \\
15 \\
\end{bmatrix}
$$

Here we have unique solution

$$ a = 5, b = 5 $$

Another way to arrive at the solution is to compute the inverse
$P^{-1}$ of the coefficient matrix $P$ ; if we have that we can multiply
both sides on the left by it and compute the result.

In the above case,

$$
P^{-1} = \begin{bmatrix}
-1 & 1 \\
2 & -1 \\
\end{bmatrix}
$$

Multiplying $P^{-1}$ by the right-hand-side;

$$
\begin{bmatrix}
-1 & 1 \\
2 & -1 \\
\end{bmatrix} . \begin{bmatrix}
10 \\
15
\end{bmatrix} = \begin{bmatrix}
5 \\
5
\end{bmatrix} = \begin{bmatrix}
a \\
b
\end{bmatrix}
$$

This is our solution from above.

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
if it is singular, the right-hand side constant vector determines whether
we have zero or infinite solutions, but in both those cases the matrix
is singular.

If a matrix P has an inverse, there is always a solution since we start
with

$$
P.x = c
$$

where c is the constant vector.  If we have an inverse for P (non-singular),
whatever c is we can simply multiply both sides by $P^{-1}$ :

$$
P^{-1}.P.x = P^{-1}.c
$$

$$
=> I.x = P^{-1}.c  => x = P^{-1}.c
$$

And with the above we can simply read off the x values from the vector
multiplication of the inverse and c.  This shows that invertibility
implies a _single_ solution; non-invertible (singular) matrices have
either no solution or infinitely many.

To generate infinitely many solutions, the rows must be redundant
and one must be expressible in terms of the others.  To generate
no solutions we have linear dependence also, but with contradicting
values for the variables.  So we see how linear (in)dependence,
and (non-)singularity are entwined.

A good question to ask is if the linear transformation spans $R^n$ -
i.e. can all points in $R^n$ be reached after the linear transformation?

If a linear transformation does not span $R^n$ it must be linearly
dependent - one of the rows is attainable from a linear combination of
the others; for example

$$
\begin{bmatrix}
1 & 4 \\
4 & 16
\end{bmatrix}
$$

Row 2 is 4 X row 1.  With this matrix we can only reach points that
are multiples of the vector

$$
\begin{bmatrix}
1 \\
4
\end{bmatrix}
$$

i.e. the transformation transforms points in $R^2$ to points along
the line represented by that vector.

We can use this fact about linear dependence to construct a test for
linear dependence, and hence singularity.

For a 2x2 matrix

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

it is linearly dependent if

$$
ka = c
$$

and

$$
kb = d
$$

rearranging these

$$
k = c/a = d/b => ad = bc => ad - bc = 0
$$

So multiply the main diagonal elements of a 2x2 matrix and subtract
the product of the off-diagonal elements; if this is 0 the matrix
is singular.

This is a quick test for linear dependence and hence singularity;
it is called the determinant and generalizes to larger square matrices.

For larger than 2x2 matrices there are a few techniques we can use.

- For large matrices, we can use Gaussian elimination and once we
  have an upper triangular matrix (where all entries below diagonal are
  zero) we can simply get the product of the main diagonal entries
  to compute the determinant.

- We can use Laplace expansion; it breaks down the determinant of
  an nXn matrix into the weighted sum of determinants of (n-1)x(n-1)
  sub-matrices, so is recursive until we hit 2/3 and can compute the
  determinants directly. The process is
	- choose row/column i,j
	- get the determinant of the (n-1)x(n-1) matrix
	  formed by deleting row i, column j from the matrix; this is
	  the $cofactor_{i,j}$
	- compute the determinant via

$$
		det(A) = \sum_{j=1}^n (-1)^{i+j} . a_{i,j} . cofactor(A_{i,j})
$$

foo
bar
baz
