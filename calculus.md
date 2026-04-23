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
## Inflection points

An inflection point is where the derivative is zero; these
can be maxima, minima (possibly local) or saddle points.
We can use the second derivative (the slope of the slope)
to characterize them; if the second derivative at a point is
negative it is a maximum, if it is positive it is a minimum.

