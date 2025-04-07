from pipeline import Pipeline

# Your input data
# data = """
# Gradient descent is an algorithm that numerically estimates where a function outputs its lowest values. That means it finds local minima, but not by setting \[\nabla f = 0\] like we've seen before. Instead of finding minima by manipulating symbols, gradient descent approximates the solution with numbers. Furthermore, all it needs in order to run is a function's numerical output, no formula required. The way gradient descent manages to find the minima of functions is easiest to imagine in three dimensions.
# Think of a function \[f(x, y)\]  that defines some hilly terrain when graphed as a height map. We learned that the gradient evaluated at any point represents the direction of steepest ascent up this hilly terrain. That might spark an idea for how we could maximize the function: start at a random input, and as many times as we can, take a small step in the direction of the gradient to move uphill. In other words, walk up the hill.
# To minimize the function, we can instead follow the negative of the gradient, and thus go in the direction of steepest descent. This is gradient descent. Formally, if we start at a point \[x_0\]  and move a positive distance \[\alpha\] in the direction of the negative gradient, then our new and improved  \[x_1\]  will look like this: \[x_1 = x_0 - \alpha \nabla f(x_0)\] More generally, we can write a formula for turning  \[x_n\] into \[x_{n + 1}\]:\[x_{n + 1} = x_n - \alpha \nabla f(x_n)\]
# """

data = """
Gradient descent is an optimization algorithm used in machine learning to find the minimum of a function (cost function) by iteratively adjusting parameters in the direction of the negative gradient, aiming to find the optimal set of parameters
"""
pipe = Pipeline(data)