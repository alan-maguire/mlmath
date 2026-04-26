# Calculus

Calculus is important in machine learning as calculating derivatives
is a prelude to finding function maxima and minima; these are used
in machine learning to optimize (minimize) a loss function.

A simple example is linear regression; find the line that minimizes
the distance from the line to the data points.

## Derivative

A derivative is the instantaneous rate of change of a function.
We can define a derivative in terms of the limit of the slope
of the function between two points as they approach each other;
this becomes a tangent line orthogonal to the function at that
point.

## Notation

We denote the derivative as

$$
\frac{dy}{dx}
$$

or using Lagrange notation, $f'(x)$ for a derivative,
$f''(x)$ for a second derivative, where $y = f(x)$.

## Basic derivatives

- Derivative of a constant $f(x) = c$ is 0, since the
  function never changes, no change in slope 0.

- Derivative of a line $f(x) = mx + c$ is slope m.
  Slope never changes so derivative is constant.

  Proof: Substituting in values for slope between
  $x$ and $x + \Delta$ , we get

$$
\frac{\Delta y }{\Delta x } = \frac{ mx + m\Delta x + c - ( mx + c)}{x + \Delta  x - x}
$$

  Simplifying we end up with

$$
\frac{ m\Delta x }{\Delta x } = m
$$

  If we then take the

$$
lim_{\Delta x -> 0}\frac{\Delta y }{\Delta x } = m
$$

  ...this becomes m since we have eliminated $\Delta x$ , and hence
the division by zero.  Many proofs take this form.


 - Derivative of $f(x) = x^2$ is $2x$ .

   Proof:

$$
\frac{\Delta y }{\Delta x } = \frac{x^2 + 2\Delta x + \Delta x ^2 - x^2}{\Delta x}
$$

   This simplifies to

$$
2 x + \Delta x
$$

   Taking the limit as $\Delta x -> 0$ the $\Delta x$ term vanishes, so

$$
lim_{\Delta x -> 0}\frac{\Delta y }{\Delta x } = 2x
$$

 - Derivative of $f(x) = x^n$ is $nx^{n-1}$

   For n=3 we can work the proof as above; expanding $(x + \Delta x)^3$
   is the key; we wind up with

$$
x^3 + \Delta x^3 + x^2 \Delta x + 2x^2 \Delta x + x \Delta x^2 + 2 x \Delta x^2
$$

   The $x^3$ term is subtracted, and the terms with $\Delta x^2$
   disappear since they become $\Delta x$ when divided by $\Delta x$ and
   the limit brings them to 0, leaving us with

$$
lim_{\Delta x -> 0}\frac{\Delta y }{\Delta x } = 3x^2
$$

   since these are the only terms not containing $\Delta x$


 - Derivative of $f(x) = \frac{1}{x}$

$$
\frac{\Delta y }{\Delta x } = \frac{\frac{1}{x + \Delta x} - \frac{1}{x}}{\Delta x}
$$

   This becomes

$$
\frac{\frac{x - (x + \Delta x)}{(x+\Delta x)(x)}}{\Delta x}
$$

   or

$$
\frac{\frac{- \Delta x)}{(x+\Delta x)(x)}}{\Delta x}
$$

   simplifying to

$$
\frac{-1}{(x+\Delta x)(x)}
$$

   Taking limit as $\Delta x -> 0$ we wind up with

$$
lim_{\Delta x -> 0}\frac{\Delta y }{\Delta x } = -1 x^{-2}
$$

 - Derivative of $f^{-1}(x)$ , the inverse of a function $f(x)$
   If a function $f(x)$ has an inverse function $g(x) = f^{-1}(x)$
   then for every point $[x,f(x)]$ for $f(x)$ there is a point
   $[f(x),x]$ for $g(x)$.  Because of this the slope of $g(x)$
   is the reciprocal of the slope of $f(x)$ at every point, and as
   a result, if we know $f'(x)$, the derivative of $f(x)$, we can
   get the derivative of the inverse $g(x)$ via
$$
g'(x) = \frac{1}{f'(x)}
$$

   the reciprocal of the derivative of $f(x)$.

 - Derivative of trigonometric functions.

   For $f(x) = sin(x)$ the derivative is
$$
f'(x) = cos(x)
$$

   And for $f(x) = cos(x)$ the derivative is

$$
f'(x) = -sin(x)
$$

   Proof for $f(x) = sin(x)$

$$
\frac{sin(x + \Delta x) - sin(x)}{\Delta x}
$$

   Expanding the first term we get

$$
\frac{sin(x)cos(\Delta x) + cos(x)sin(\Delta x) - sin(x)}{\Delta x}
$$

   which we can rearrange as

$$
\frac{sin(x)(cos(\Delta x) - 1) + cos(x)sin(\Delta x)}{\Delta x}
$$

   Taking limit and separating we get

$$
sin(x).lim_{\Delta x -> 0} (\frac{cos(\Delta x) - 1}{\Delta x}) + cos(x).lim_{\Delta x -> 0} (\frac{sin(\Delta x)}{\Delta x})
$$

   Using the fact that $lim_{h->0} \frac{sin(h)}{h} = 1$ (think of slope at h = 0)
   and $lim_{h->0} \frac{cos(h) - 1}{h} = 0$ we get

$$
f'(x) = sin(x).(0) + cos(x).(1) = cos(x)
$$

 - Derivative of Euler's number $f(x) = e^x$

   $e$ is the irrational number $2.71828182..$ that can be defined as

$$
lim_{n->\infty}(1 + \frac{1}{n})^n
$$

   Derivative is special; when $f(x) = e^x$

$$
f'(x) = f(x)
$$

 - Derivative of $ln(x)$.  $y = ln(x)$ is the inverse of $x = e^y$
   because $ln(e^y) = y$.  So if we differentiate both sides of
   $x = e^y we get:

$$
\frac{d}{dx}(e^y) = \frac{d}{dx}(x)
$$

   The LHS becomes (from chain rule)

$$
\frac{d}{dx}(e^y) = e^y \frac{dy}{dx}
$$

   And the right hand side becomes:

$$
\frac{dx}{dx} = 1
$$

   Rearranging we get

$$
\frac{dy}{dx} = \frac{1}{e^y}
$$

   And since $e^y = x$, we end up with

$$
\frac{dy}{dx} = \frac{1}{x}
$$

## Differentiability

A function with a discontinuity or an infinite slope is not everywhere
differentiable.

## Rules for derivatives

 - Scalar multiplication: if $g(x) = a f(x)$ for scalar $a$

$$
g'(x) = a f'(x)
$$

   Intuition is if the $y$s are $a$ times the original, the slopes
   will be too.

 - Sum rule if $f(x) = g(x) + h(x)$ then

$$
f'(x) = g'(x) + h'(x)
$$

   Intuition is if the $y$ s are added, slopes are also added.

 -  Product rule; if $f(x) = g(x)h(x)$ then

$$
f'(x) = g'(x)h(x) + g(x)h'(x)
$$

 - Chain rule

   For a composition $f(x) = g(h(x))
$$
\frac{df}{dx} = \frac{dg}{dh} . \frac{dh}{dx}
$$

   We see that Leibnitz notation is quite suggestive of the chain rule.

## Using SymPy to symbolically differentiate

We can use SymPy to differentiate functions symbolically after defining
symbols:

```
>>> import sympy as sp
>>> x = symbols('x')
>>> sp.diff(x**3,x)
3*x**2
```

We can evaluate expressions by substituting values:

```
>>> expr = sp.diff(x**3,x)
>>> expr.evalf(subs={x:1})
3.00000000000000
```

Symbolic differentiation is useful for cases where we are exploring solutions.

## Using numpy to numerically differentiate

We use numerical differentiation which uses approximation at nearby points
to evaluate the derivative.  We can pass an arbitrary function to
`np.gradient`:

```
>>> def f(x):
...  return x*x
... 
>>> x = np.array([1,2,3,4])
>>> np.gradient(f(x),x)
array([3., 4., 6., 7.])
```

## Inflection points

An inflection point is where the derivative is zero; these
can be maxima, minima (possibly local) or saddle points.
At these points the first derivative is zero.

We can use the second derivative (the slope of the slope)
to characterize them; if the second derivative at a point is
negative it is a maximum, if it is positive it is a minimum.

We are particularly interested in minima of error functions
in machine learning; optimization problems are problems that
require us to minimize a loss/error function.

It is important to remember a point of inflection can be a
local rather than global minimum.

## Optimization using squared loss

Often when optimizing we optimize the squared loss; in such cases
we do not care about the direction of the error, but rather its
magnitude.  The loss function is usually a sum of such squared errors.
In a simple case, if the loss function is


$$
L(x) = (x-a_1)^2 + (x-a_2)^2 + .. (x-a_n)^2
$$

The solution is

$$
x = \frac{a_1+a_2+..+a_n}{n}
$$

Because when differentiating L(x) and setting it to 0 we get

$$

L'(x) = 2(x-a_1) + 2(x-a_2) + 2(x-a_n) = 2nx - 2(a_1 + a_2 + ... + a_n) = 0
$$

(since differentiating $(x-a)^2$ with the chain rule gives $2(x-a)$ ).

Therefore to minimize $L(x)$ (where $L'(x) = 0),

$$
x = \frac{a_1+a_2+...+a_n}{n}
$$

## Optimization using log loss

Sometimes taking the log of the loss function can greatly simplify
minimization; it allows us to convert powers to multiples, i.e.

$$
ln(x^8) = 8ln(x)
$$

Also the log of a product becomes the sum of two logs, i.e. 

$$
ln(x^7(1-x)^3) = ln(x^7) + ln(1-x)^3 = 7ln(x) + 3ln(1-x)
$$

This becomes much more easily differentiable;

$$
\frac{d}{dx} = \frac{7}{x} + \frac{3}{1-x}(-1)
$$

When dealing in probabilities in machine learning we often take the
negative log loss because we often deal in probabilities and taking
the negative of a log betwee 0-1 gives us a positive value.

We then minimize that positive number to minimize loss.

