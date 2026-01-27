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
>>> np.ndim(A)
2
>>> np.size(A)
6
```

- `np.ndim()` is the number of dimensions (2 for a matrix)
- `np.shape()` is the number of rows/columns
- `np.size()` is the number of elements

We index elements via row i, column j as $A_{i,j}$ ,
and in python we can select a whole row or column using `:`
instead of an index:

```
>>> A[1,2]
9
>>> A[:,1]
array([2, 8])
```

We can index whole rows/columns using double parentheses; e.g.
to create B which is A with second and third rows swapped:

```
>>> A
array([[2, 1, 5],
       [1, 3, 1],
       [3, 4, 6]])
>>> B=np.copy(A)
>>> B[[2,1]]=B[[1,2]]
>>> B
array([[2, 1, 5],
       [3, 4, 6],
       [1, 3, 1]])
>>> 
```

We can create special matrices, consisting of all ones, zeros
diagonal, or identity:

```
>>> A = np.zeros([2,2])
>>> A
array([[0., 0.],
       [0., 0.]])
>>> A = np.ones([2,2])
>>> A
array([[1., 1.],
       [1., 1.]])
>>> A = np.diag(np.array([1,1]))
>>> A
array([[1, 0],
       [0, 1]])
>>> A = np.identity(2)
>>> A
array([[1., 0.],
       [0., 1.]])

```

There are many options for random matrices, for example
`np.random.randn(r,c)` for random matrix between -1,1,
`np.random.randint(low,high,shape)`:

```
>>> A = np.random.randn(2,3)
>>> A
array([[-1.18656341, -0.40310253,  0.28816619],
       [ 0.23597831, -0.13745851,  0.93330382]])
>>> A = np.random.randint(10, 20, [2,2])
>>> A
array([[10, 14],
       [14, 11]])
>>> 

```

We can reshape a matrix vi `np.reshape(matrix, [r,c])`:

```
>>> A = np.ones([2,3])
>>> np.reshape(A,(6,1))
array([[1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.]])
```

## Matrix operations

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

In python, matrix multiplication is done via `np.matmul()` or
the '@' operator, and `np.dot()` works too via broadcasting:

```
>>> P = np.array([[5,2,0],[7,3,1]])
>>> Q = np.array([[2,1],[7,3],[8,0]])
>>> P@Q
array([[24, 11],
       [43, 16]])
>>> np.matmul(P,Q)
array([[24, 11],
       [43, 16]])
>>> np.dot(P,Q)
array([[24, 11],
       [43, 16]])
```
 
## Properties of matrix multiplication

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

## Broadcasting, slicing, stacking in numpy

The concept of broadcasting is important; it allows us to carry
out operations between objects whose dimensions do not match; for
example multiplying a vector by a scalar.

In numpy it extends to operations like adding a vector to a matrix:

```
>>> A = np.ones([2,3])
>>> B = np.ones([2,1])
>>> A+B
array([[2., 2., 2.],
       [2., 2., 2.]])

>>>
```

We can also retrieve a subset by indexing using `[start]:[end]`; for
example to retrieve the 2/3rd rows/columns from 

```
>>> A = np.ones([3,3])
>>> A[1:,1:]
array([[1., 1.],
       [1., 1.]])
```

We can also stack arrays horizontally, vertically via `np.hstack()`
and `np.vstack()`:

```
>>> B = 2 * A
>>> B
array([[2., 2., 2.],
       [2., 2., 2.],
       [2., 2., 2.]])
>>> np.hstack((A,B))
array([[1., 1., 1., 2., 2., 2.],
       [1., 1., 1., 2., 2., 2.],
       [1., 1., 1., 2., 2., 2.]])
```

## Geometric interpretation

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

For example consider the matrix representing a $90^o$ clockwise
rotation (to see that this is the right matrix, remember the first
column is where $\begin{bmatrix}
1 \
0
\end{bmatrix} is mapped too; ditto for the second column.

The transformation is this:

$$
\begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix}
$$

Applying this transformation twice is equivalent to multiplying
A by itself or rotating $180^o$:

$$
\begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix} \begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix} = \begin{bmatrix}
-1 & 0 \\
0 & -1
\end{bmatrix}
$$

## Matrix inverse

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

## Solving systems of linear equations with matrices

We can use matrices to solve systems of linear equations.

1. We construct a matrix where the rows are the coefficents
   of the variables
2. We augment it with the constants on the right-hand side
   of the equations in a column
3. We then carry out Gaussian elimination
	- divide each row by its leftmost coefficient
	- subtract top row from each other row
	- repeat

We end up with an upper triangular matrix where each leftmost
element is 1 (a pivot); the rank of the matrix is the number of
such pivots. It is in row-echelon form.  We can back-substitute
to solve or fix up to be reduced-row echelon form (where there
are 0s above each 1).

Example:

Solve

$$
x + 3y = 15
$$

$$
3x + 12y = 3
$$

Create the augmented matrix:

$$
\begin{bmatrix}
1 & 3 & 15 \\
3 & 12 & 3 \\
\end{bmatrix}
$$

r2 -> 1/3(r2)

$$
\begin{bmatrix}
1 & 3 & 15 \\
1 & 4 & 1 \\
\end{bmatrix}
$$

r2 -> r1 - r2

$$
\begin{bmatrix}
1 & 3 & 15 \\
0 & -1 & 14 \\
\end{bmatrix}
$$

r2 -> -r2

$$
\begin{bmatrix}
1 & 3 & 15 \\
0 & 1 & -14 \\
\end{bmatrix}
$$

At this point we are in row-echelon form and can back-substitute.
To finish in reduced row-echelon form:

r1 -> r1 -3(r2)

$$
\begin{bmatrix}
1 & 0 & 57 \\
0 & 1 & -14 \\
\end{bmatrix}
$$

So we see

$$
x = 57, y = -14
$$

We can use `np.linalg.solve()` to solve Ax = b; for example

```
>>> A
array([[4, 1],
       [1, 1]])
>>> b = np.array([2,1])
>>> np.linalg.solve(A,b)
array([0.33333333, 0.66666667])
>>> A@np.linalg.solve(A,b)
array([2., 1.])
```

## Matrices as systems of equations

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

Note that $Ax = b$ must have either
- a unique solution (non-singular)
- no solution (singular)
- infinite solutions (singular)

It is not possible to have > 1 solution but < an infinite number.

To prove this consider x and y as solutions; we can construct
a new vector with aribrary real value $\alpha$ :

$$
z = \alpha . x + (1 - \alpha)y
$$

This works as a solution becauase if

$$
Ax = b ; Ay = b
$$

$$
Az = A(\alpha .x ) + A(1 - \alpha)y
$$

$$
=> \alpha(A.x) + Ay - \alpha(A.y)
$$

But since $A.x$ = b, $A.y = b$ , this becomes

$$
=> \alpha.b + b - \alpha.b = b
$$

Therefore $A.z = b$ and z is a solution also for any $\alpha$.

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

In a row-centric view of a matrix, we think of it as the coefficients
of the variables x in $Ax = b$ , but we can also take a column-centric
view, where the transformation denoted by A is a linear combination
of the n columns; i.e.

$$
x_1.A_{:,1} + x_2.A_{:,2} + ... + x_n.A_{:,n}
$$

The span of the transformation represented by A is the set of all points
obtainable by such linear combinations. This space is known as the
column space of A.

For $Ax = b$ to have a solution for all possible b, the column space must
to reach all
span $R^m$ (where m is the number of rows, hence the dimension of the
space).

So we see why linear dependence between columns is a problem; it restricts
the span to < $R_m$.

## Some special Matrices

- Identity matrix: we have already seen the identity matrix; 1s on main
  diagonal with zeros everywhere else;
- Diagonal matrix: non-zero on main diagonal (not necessarily 1s).
  Multiplication is easy because $diag(v).x$ just requires multiplying
  each vector component by the appropriate diagonal entry.
- Symmetric matrix: $A^T = A$
- Orthogonal vectors: vectors are orthogonal if $x^Ty = 0$
- Orthonormal vectors: orthogonal vectors with $||x||^2 = ||y||^2 = 1$
- Orthonogonal matrix:  columns in $A$ are _orthonormal_, each with $L^2$
  norm of 1.  An important fact is $A^TA = AA^T = I$. This is because
  when we take the transpose of $A$, its orthonormal columns become rows;
  then when we multiply each row with the columns, only the matching column
  will be non-zero, so we end up with the identity matrix.  From this
  we can see that $A^T = A^-1$ , since $A^TA = AA^T = I$.

## Determinant

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

- For 3x3 matrices we can use a particular technique of Laplace expansion;
  we sum the products of all the diagonals from top left to bottom right,
  wrapping around and subtracting diagonals from top right to bottom left.

For example, to compute the determinant for matrix

$$
\begin{bmatrix}
1 & 2 & 5 \\
0 & 3 & -2 \\
2 & 4 & 10 \\
\end{bmatrix}
$$

we get

$$
((1)(3)(10)) + ((2)(-2)(2)) + ((5)(0)(4)) - ( ((5)(3)(2)) + ((1)(-2)(4)) + ((2)(0)(10)) )
$$

$$
=> 30 - 8 + 0 - (30 - 8 + 0) = 0
$$

Hence it is singular.

We can use

```
np.linalg.det(A)
```

to compute the determinant of a matrix in python.

One thing to note in passing about determinants and Gaussian elimination;
it uses two mechanisms that preserve the singularity/non-singularity

- adding one row to another results in the same determinant
- multiplying a row by a constant makes a non-zero determinant non-zero,
  and a zero determinant still zero

Also note that the determinant of $A^-1$ is $\frac{1}{det(A)}$ ; we can
see this since $A^-1.A = I$ and $det(I) = 1$.

The geometric interpretation of the determinant is that it is the
area or volume of the transformation of the unit square formed by
$[0,0], [0,1],[1,0],[1,1]$; thus when the determinant is zero
we get a degenerate shape of area/volume 0 like a line or point in 2d.

## Eigenvectors and eigenvalues

Recall that a basis is a set of vectors that

- spans a vector space; and
- is linearly independent

So for $R^3$ for example we would need 3 linearly independent
vectors to form a basis to span the space.

Eigenvectors are vectors $v_n$ such that

$$
Av_n = \lambda v_n
$$

What this means geometrically is that $v_n$ points in the same
direction after the linear transformation.  We call $\lambda$
the eigenvalue and $v_n$ the eigenvector.  The eigenvalue
gives us the amout of stretch/shrink that is applied.

An eigenbasis is a basis for a linear transformation that
consists of eigenvectors.  What is critical about the eigenbasis
is that it much more efficient.

If we have the eigenvalues/eigenvectors for a matrix, if we
can figure out the transformation for a vector much more
cheaply. We simply

1. express the vector as a linear combination of the eigenvectors
2. take the coefficents of each of these, multiply them by the
   associated eigenvalue

For example, for the following linear transformation A:

$$
A = \begin{bmatrix}
2 & 1 \\
0 & 3
\end{bmatrix}
$$

We have eigenvectors $e_1$

$$
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

with eigenvalue $\lambda_1$ 2; and eigenvector $e_2$

$$
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

with eigenvalue $\lambda_2$ 3.

To get the transformation of

$$
\begin{bmatrix}
-1 \\
2
\end{bmatrix}
$$

we express it as a linear combination of $e_1 , e_2$ :

$$
-3 (e1) + 2 (e2)
$$

So $A$ mutiplied by the above will be

$$
\lambda_1 (-3)(e1) + \lambda_2 (2) (e2)
$$

$$
=> (2)(-3)\begin{bmatrix}
1 \\
0
\end{bmatrix} + (3)(2)\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

$$
=> \begin{bmatrix}
-6 \\
0
\end{bmatrix} + \begin{bmatrix}
6 \\
6
\end{bmatrix}
$$

$$
=> \begin{bmatrix}
0 \\
6
\end{bmatrix}
$$

## Computing eigenvectors and eigenvalues

If

$$
Av = \lambda v
$$

$$
Av - \lambda v = 0
$$

So

$$
(A-\lambda I)v = 0
$$

This is true for v at $(0,0,...)$ but for the non-trivial
eigenvector case it is true when

$$
det(A-\lambda I)v = 0
$$

...since if $(A - \lambda I)$ was invertible, $det(A - \lambda I)$
would be non-zero and we could just multiply by the inverse;
then we would get

$v = 0$ for $v \neq (0, 0, ...)$ which is a contradiction.

We can solve for the above, getting the characteristic equation.
Consider

$$
A = \begin{bmatrix}
4 & 1 \\
2 & 3
\end{bmatrix}
$$

$$
A - \lambda I = \begin{bmatrix}
4 - \lambda & 1 \\
2 & 3 - \lambda
\end{bmatrix}
$$

So, given that the determinant is 0,

$$
(4 - \lambda)(3 - \lambda) - 2 = 0
$$

THe characteristic polynomial is:

$$
=> \lambda^2 -7\lambda + 10 = 0
$$

$$
=> (\lambda -5)(\lambda - 2) = 0
$$

So

$$
\lambda = 5, 2
$$

Now we have the eigenvalues, to get the eigenvector
for each, substitute $\lambda$ into $(A-\lambda I)$ :

$$
A - 5I = \begin{bmatrix}
4-5 & 1 \\
2 & 3-5
\end{bmatrix} = \begin{bmatrix}
-1 & 1 \\
2 & -2
\end{bmatrix}
$$

Now we use this in $(A-\lambda I)v = 0$ as follows:

$$
=> \begin{bmatrix}
-1 & 1 \\
2 & -2
\end{bmatrix} 
\begin{bmatrix}
x \\
y
\end{bmatrix} = \begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

From this, we get equations

$$
-x + y = 0
$$

$$
2x -2y = 0
$$

These simplify to $y = x$ so we use eigenvector

$$
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

Similarly, for $\lambda = 2$, $A - 2I$ becomes

$$
\begin{bmatrix}
2 & 1 \\
2 & 1
\end{bmatrix}
$$

So again we use this in $(A-\lambda I)v = 0$ as follows:

$$
=> \begin{bmatrix}
2 & 1 \\
2 & 1 
\end{bmatrix} 
\begin{bmatrix}
x \\
y
\end{bmatrix} = \begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

From this, we get equation

$$
2x + y = 0
$$

So for this we can use the vector

$$
\begin{bmatrix}
1 \\
-2
\end{bmatrix}
$$

as eigenvector.

Eigenvalues can be computed via `np.linalg.eigvals()`;
eigenvalues and eigenvectors can be computed via
`np.linalg.eig()`:

```
>>> A = np.array([[4,1],[2,3]])
>>> np.linalg.eigvals(A)
array([5., 2.])
>>> eigenvals, eigenvecs = np.linalg.eig(A)
>>> eigenvals
array([5., 2.])
>>> eigenvecs
array([[ 0.70710678, -0.4472136 ],
       [ 0.70710678,  0.89442719]])
```

Note that `np.linalg.eig()` always gives vectors with
$L^2$ norm 1.

Note that multiple eigenvectors can share the same eigenvalue,
but where eigenvalues are distinct the associated eigenvalues
will be linearly independent.

The conceptual value of eigenvalues/eigenvectors is this - in
the normal course of events, a linear transformation mixes
up values associated with the variables.  However with an
eigenbasis, we can always express the result as a set of
scalar multiplications of those eigenbases (we are not always
guaranteed that we will have n eigenvalues, so the eigenvalues
may not map to the full $R^n$ however).

Further the eigenvalues tell us the degree of growth/shrink
associated with each eigenvector so we can compress our
repesentations to make them a combination of the first n
eigenvectors; we will see the steps later.

foo
bar
baz
