# Vectors

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

Vectors are abstracted as n-tuples that obey scalar multiplication
and addition; see below for how these are done.

In python, we create a vectors using numpy

```
$ python3
>>> import numpy as np
>>> x = np.array([1],[2],[3])
>>> x
array([1],
       [2],
       [3])
>>> 
```

We can index a vector element as
$x_i$ : on python we start from index 0:

```
>>> x[0]
1
```

x is a column vector; to create row vector

$$
x = \begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
$$

```
>>> x2 = np.array([1,2,3])
>>> x2
array([1, 2, 3])

```

## Transpose

We can flip a vector from columnar to row form (or vice
versa) using the transpose operation, but we must have
defined it as 2-dimensional first;

$$
x^T = \begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
$$

In python:

```
>>> x = np.array([[1],[2],[3]])
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

## Vector operations

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

## Norm

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

## Dot product

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

## Geometric dot product

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
proj(x, y) = \frac{x^T.y}{||y||||y||} y
$$

Note the additional y vector since the value in front of it
simply gives the length of the projection along y; we still
need the y vector to have the magnitude and direction of the
projection vector. We also need to scale by the length of y
to get a unit vector in the direction of y to multiply the
scalar projection by, hence the additional $||y||$ in the denominator.

This geometric interpretation also tells us when the projection
of x onto y is positive, the dot product of $x^T.y$ will be
positive, while if it is negative the dot product will be too.
And of course if they are orthogonal the projection is zero.

## Change of basis

A vector in $R^n$ can be expressed via different linear combinations
of a set of n basis vectors.  The standard bases are sometimes called
$e_1, e_2, .. e_n$ and these are vectors of the form

$$
\begin{bmatrix}
1 \\
0 \\
.. \\
0
\end{bmatrix}, \begin{bmatrix}
0 \\
1 \\
0 \\
..
\end{bmatrix}, ...,  \begin{bmatrix}
0 \\
.. \\
0 \\
1
\end{bmatrix}
$$

We can however describe a vector with an arbitrary set of basis
vectors.  It is always best to use orthogonal basis vectors as
these simplify the change of basis.

With an change of basis to an orthogonal set of basis, we can use
projection of new vectors $r_1,...r_n$ onto $e_1,..,e_n$.

First verify new basis is orthogonal, i.e. $(r_1)(r_2)...(r_n) = 0$.

For example change basis of
$$
r = \begin{bmatrix}
3 \\
4
\end{bmatrix}
$$

to be expressed via $b_1, b_2$ :

$$
b_1 = \begin{bmatrix}
2 \\
1
\end{bmatrix}
$$

$$
b_2 = \begin{bmatrix}
-2 \\
4
\end{bmatrix}
$$

First verify $b_1.b_2 = 0$ :

$$
(2)(-2) - (1)(4) = 0
$$

Now compute projection of $r_1$ onto $b_1$ :

$$
\frac{r_e . b_1}{|b_1|^2} = \frac{(3)(2) + (4)(1)}{(2^2 + 1^2} = \frac{10}{5} = 2
$$

$$
=> \frac{r_e . b_1}{|b_1|^2} b_1 = 2 \begin{bmatrix}
2 \\
1
\end{bmatrix} = \begin{bmatrix}
4 \\
2
\end{bmatrix}
$$

Now compute projection of $r_2$ onto $b_2$ :

$$
\frac{r_e . b_2}{|b_2|^2} = \frac{(3)(-2) + (4)(4)}{(2^2 + 1^2} = \frac{10}{20} =
\frac{1}{2}
$$

$$
=> \frac{r_e . b_2}{|b_2|^2} b_2 = \frac{1}{2} \begin{bmatrix}
-2 \\
4
\end{bmatrix} = \begin{bmatrix}
-1 \\
2
\end{bmatrix}
$$

So in the new basis we get

$$
2 \begin{bmatrix}
2 \\
1
\end{bmatrix} + \frac{1}{2} \begin{bmatrix}
-2 \\
4
\end{bmatrix} = \begin{bmatrix}
4 \\
2
\end{bmatrix} + \begin{bmatrix}
-1 \\
2
\end{bmatrix} = \begin{bmatrix}
3 \\
4
\end{bmatrix}
$$

...which was our original vector.  So we see the projection
gives us the coefficients to use in front of the new basis
after the change of basis.

## Basis

A basis is a set of n vectors that

- are not linear combinations of each other (are linearly independent)
- span the space $R^n$

It is best to use orthonormal basis vectors where each vector is

- orthogonal to all the others
- has length 1

We often want to change basis to a basis that is more representative
of our data so we can express instances of data in a more concise
form.  We will see later how one such approach uses a special basis
called the eigenbasis to support such conciseness.

Many neural networks transform input into more salient features using
a basis change such that the bases are the features of interest.

