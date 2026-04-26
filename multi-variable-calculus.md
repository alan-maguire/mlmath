# Multi variable calculus

## Partial derivatives

When dealing with multiple variables, we take partial derivatives;
we take a derivative with respect to a specific variable, treating
others as a constant.  For example for $f(x,y) = x^2 + y^2$

$$
\frac{\delta f}{\delta x}(x^2 + y^2) = 2x
$$

$$
\frac{\delta f}{\delta y}(x^2 + y^2) = 2y
$$

We group these together into a gradient

$$
\nabla f = \begin{bmatrix} 
\frac{\delta f}{\delta x} \\
\frac{\delta f}{\delta y}
\end{bmatrix}
$$

In this case it represents the two slopes of the lines that form the
tangent plane to the curve.  It allows us to assess the "slope" of
the curve at a particular point by plugging in particular x and y values.

## Minimization

When minimizing in one variable we minimize where slope is 0, i.e.
set the derivative to 0.

In multiple variables, we minimize where the tangent plane lines have
slopes of 0; i.e. when the gradient composed of the partial derivatives
is equal to 0.

So in the above example, the minimum point is that which corresponds
to where the gradient is 0, i.e. where $2x = 0$ and $2y = 0$ ; i.e.
$(x,y) = (0,0)$.

For more complex cases, we can use linear algebra to do elimination
across multiple variables to find solutions.  Sometimes we need to
test solutions to determine if they are minima or maxima.

However sometimes computing an analytical solution - especially where
a large number of variables are involved - is too expensive.  Sometimes
it is hard to solve the gradient value.

## Gradient descent

Rather than solving analytically, gradient descent tries to locally
move down gradient to find a minimum.  Starting at a position $x_n$,
we find the next position

$$
x_{n+1} = x_0 - \alpha f'(x_n)
$$

i.e. we use a small learning rate ($\alpha$) part of the gradient
to move agaist the gradient (downhill).  A learning rate is used
because if the slope is large we will make too-big jumps.

Because the gradient reduces as we approach a minimum, the algorithm
slows down as we come near to it.  A too-large $\alpha$ can result in
oscillating around the minimum.

One worry is that we can get stuck in local minima with this approach,
where there are better optimizations available.

In mulitple dimensions, we use the gradient:

$$
x_{n+1} = x_0 - \alpha \nabla f(x_n)
$$


## Regression with a perceptron

A perceptron can be used for linear regression; in such a case the
loss function is the squared error between actual output and
prediction:

$$
L(y,y_{pred}) = \frac{1}{2}(y - y_{pred})^2
$$

The prediction is computed via


$$
y_{pred} = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b
$$

Our aim is to comute $w_1$, ...,$w_n$ and $b$ such that
we can make accurate predictions.

We then compute the gradient using the chain rule

$$
\frac{\delta L}{\delta b} = \frac{\delta L}{\delta y_{pred}} \frac{\delta y_{pred}}{\delta b}
$$

And calculating each of these:

$$
\frac{\delta L}{\delta y_{pred}} = -(y - y_{pred})
$$

$$
\frac{\delta y_{pred}}{\delta b} = 1
$$

(because $y_{pred}$ only has a single $b$ term).

Similarly

$$
\frac{\delta L}{\delta w_i} = \frac{\delta L}{\delta y_{pred}} \frac{\delta y_{pred}}{\delta w_i}
$$

And here

$$
\frac{\delta y_{pred}}{\delta w_i} = x_i
$$

(since only the $w_i x_i$ term is left when differentiating with respect
to $w_i$).

So our gradient is

$$
\begin{bmatrix}
-(y-y_{pred}).x_1 \\
-(y-y_{pred}).x_2 \\
... \\
-(y-y_{pred}).x_n \\
-(y-y_{pred}) \\
\end{bmatrix}
$$

And we use $-\alpha$ (learning rate) of this to do our updates to
$w_1, .., w_n, b$.

## Classification with a perceptron

To do classification with a perceptron we apply an activation function
to apply to

$$
z = x_1 w_1 + x_2 w_2 + ... + x_n w_n + b
$$

The idea is we want it to make a yes (1)/no (0) choice in probabilistic
terms.

A popular choice is the sigmoid function which gives a result between
0 and 1:

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

The sigmoid is an s-shaped function asymptotic to 0 and 1, never quite touching
0 as it goes more negative and never quite reaching 1 as it goes more positive.
It crosses the y-axis at $z = \frac{1}{2}$.

## Deriviative of the sigmoid function

$$
\sigma(z) = (1+e^{-z})^{-1}
$$

Using the chain rule:

$$
\frac{d}{dz} \sigma (z) = (-1)(1+e^{-z})^-2 . - e^{-z}
$$

Simplifying we get

$$
\frac{e^{-z}}{(1+e^{-z})^2}
$$

Using a trick we can add and subtract 1 to the numerator:

$$
\frac{1 + e^{-z} - 1}{(1+e^{-z})^2}
$$

Rewriting as two fractions we get:

$$
\frac{1 + e^{-z}}{(1 + e^{-z})^2} - \frac{1}{(1+e^{-z})^2}
$$

This is equivalent to 

$$
\sigma (z) - \sigma (z)^2
$$

or

$$
\sigma (z) (1 - \sigma (z))
$$

For classification by a perceptron, we use a log-loss measure

$$
L(y,y_{pred}) = -yln(y_{pred}) - (1 - y)ln(1 - y_{pred})
$$

To get the derivative

$$
\frac{\delta L}{\delta w_1}
$$

we use the chain rule calculating

$$
\frac{\delta L}{\delta w_1} = \frac{\delta L}{\delta y_{pred}} . \frac{\delta y_{pred}}{\delta w_1}
$$

The deriviative of the first component of this with respect to $y_{pred}$ is

$$
\frac{\delta L}{\delta y_{pred}} = \frac{-y}{y_{pred}} + \frac{1 - y}{1 - y_{pred}}
$$

This becomes

$$
\frac{-y + y y_{pred} + y_{pred} - y y_{pred}}{y_{pred}(1 - y_{pred})}
$$

and then

$$
\frac{-(y - y_{pred})}{y_{pred}(1 - y_{pred})}
$$

For the second component:

$$
\frac{\delta y_{pred}}{\delta w_1} = y_{pred}(1 - y_{pred})x_1
$$

For b, the first component is the same and the second is

$$
\frac{\delta y_{pred}}{\delta b} = y_{pred}(1 - y_{pred})
$$

Putting the pieces together

$$
\frac{\delta L}{\delta w_1} = \frac{-(y - y_{pred})}{y_{pred}(1 - y_{pred})} y_{pred}(1 - y_{pred}) x_1
$$

which becomes the simple

$$
\frac{\delta L}{\delta w_1} = -(y - y_{pred})x_1
$$

Similarly for $b$ we get

$$
\frac{\delta L}{\delta b} = -(y - y_{pred})
$$

Intuitively the weight adjustments do a combination of error differencing
and blame assignment, where the difference between the expected output and
the predicted is weighted by the output of the unit weight $w_i$.

## Multiple layers and backpropagation

The key difference when we introduce multiple layers is we have to
propagage the error signal back through the network to the weights
in earlier layers using multiple applications of the chain rule.

We denote weights in particular layers as

$$
w^{(n)}_{i,j}
$$

Where the above is the weight of the connection from unit i to unit j
in layer $(n)$.

Consider a 2-layer network using sigmoid activation functions. To
determine

$$
\frac{\delta L}{\delta w^{(n)}_{i,j}}
$$

we need to work backwards from the output layer:


$$
\frac{\delta L}{\delta w^{(1)}_{1,1}} =
	\frac{\delta L}{\delta y_{pred}}
	\frac{\delta y_{pred}}{\delta z^{(2)}_1}
	\frac{\delta z^{(2)}_1}{\delta a^{(1)}_1}
	\frac{\delta a^{(1)}_1}{\delta z^{(1)}_1} 
	\frac{\delta z^{(1)}_1}{\delta w^{(1)}_{1,1}}
$$

In the above, the y terms are the results of applying the sigmoid
to the weighted-sum of inputs.

Each of these terms are (moving from the output to the input)

- the deriviative of the loss with respect to the predicted output
  (the derivative of our log-loss wrt $y_{pred}$ the activation of
  the output unit

- the derivative of the output of the output unit with respect to
  the weighted sum of inputs of the output unit

- the derivative of that weighted sum with respect to the sigmoid activation of
  the unit that contributed from $w_{1,1}$ to it. At this point, we are
  backpropgating to the previous layer.

- the deriviative of that sigmoid activation with respect to the weighted
  sum

- the derivative of that weighted sum with respect to $w_{1,1}$.
  
So we use the chain rule to build a relationship between the derivative
of the loss with respect to the prediction all the way through the network
back to the weight that contributed to it.
 
## Newtons method

Netwons method is effective in finding the zeros of a function and it is
quite similar to gradient descent in that it is an interative algorithm
using the tangent lines to find zeros.

We start with a point $x_0$ and get the tangent of the function at $x_0$;
where this crosses the origin is $x_1$ ; we then use the tangent at $x_1$
for our next point $x_2$.

We can get the update rule as follows

$$
\frac{f(x_0)}{x_0 - x_1} = f'(x_0)
$$

Rearranging

$$
\frac{f(x_0)}{f'(x_0)} = x_0 - x_1
$$

So we end up with update rule

$$
x_1= x_0 - \frac{f(x_0)}{f'(x_0)}
$$

We can adapt this process; instead finding zeros, in optimization
we want to find minima (zeros of the derivative).

In this case, instead of $f(x)$ being the function, it is the derivative
of the function we are interested in $g(x)$.  Then the process is the same,
the update rule is:

$$
x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}
$$

or

$$
x_{k+1} = x_k - \frac{g'(x_k)}{g''(x_k)}
$$

where $g(x)$ is the original function and $f(x) = g'(x)$

## Second derivatives

We saw earlier that when the second derivative at a zero point is greater
than zero it is a concave up (minimum) and when it is less than zero
it is a concave down (maximumx).

With multiple variables, we can take second derivatives with respect to
multiple variables, so we wind up with a matrix of second derivatives
(the Hessian) from the vector of first derivatives (the gradient).

It looks like this:


$$
\begin{bmatrix}
\frac{\delta ^2 f}{\delta x^2}
&    \frac{\delta f}{\delta x}\frac{\delta f}{\delta y} \\
\frac{\delta f}{\delta y}\frac{\delta f}{\delta x}
&    \frac{\delta ^2 f}{\delta y^2} \\
\end{bmatrix}
$$

So in a case like

$$
f(x,y) = x^2 + y^2
$$

$$
\frac{\delta f}{\delta x} = 2x
$$

$$
\frac{\delta f}{\delta y} = 2y
$$

$$
\frac{\delta ^2f}{\delta x^2} =  \frac{\delta ^2f}{\delta y^2} =  2 ;

\frac{\delta f}{\delta x} \frac{\delta f}{\delta y} = \frac{\delta f}{\delta y} \frac{\delta f}{\delta x} = 0
$$

Note that in general

$$
\frac{\delta f}{\delta x} \frac{\delta f}{\delta y} = \frac{\delta f}{\delta y} \frac{\delta f}{\delta x}
$$

since it is just the order of differentiation that changes.

With the above we can assemble the Hessian:

$$
\begin{bmatrix}
2 & 0 \\
0 & 2 \\
\end{bmatrix}
$$

In one variable the second derivative tells us the direction of concavity.

What does it mean for a matrix to be positive?

For the Hessian, the equivalent is that the solutions for $\lambda$ -
the roots of the equation

$$
det(H - \lambda I) = 0
$$

i.e. the eigenvalues - are all positive, this is a minimum. Similarly if 
all roots are negative it is a maximum.


