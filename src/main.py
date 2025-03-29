from pipeline import Pipeline

# Your input data
data = """
In our analysis above, we focused our attention on the global minimum of the loss function. You may be wondering: what about
the local minimum thatʼs just to the left?
If we had chosen a different starting guess for , or a different value for the learning rate , our algorithm may have gotten
"stuck" and converged on the local minimum, rather than on the true optimum value of loss.
If the loss function is convex, gradient descent is guaranteed to converge and find the global minimum of the objective function.
Formally, a function is convex if:
for all in the domain of and .
To put this into words: if you drew a line between any two points on the curve, all values on the curve must be
on or below the
line. Importantly, any local minimum of a convex function is also its global minimum so we avoid the situation where the
algorithm converges on some critical point that is not the minimum of the function.
"""

pipe = Pipeline(data)