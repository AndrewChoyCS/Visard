from pipeline import Pipeline

# Your input data
# data = """
# Gradient descent is an algorithm that numerically estimates where a function outputs its lowest values. That means it finds local minima, but not by setting \[\nabla f = 0\] like we've seen before. Instead of finding minima by manipulating symbols, gradient descent approximates the solution with numbers. Furthermore, all it needs in order to run is a function's numerical output, no formula required. The way gradient descent manages to find the minima of functions is easiest to imagine in three dimensions.
# Think of a function \[f(x, y)\]  that defines some hilly terrain when graphed as a height map. We learned that the gradient evaluated at any point represents the direction of steepest ascent up this hilly terrain. That might spark an idea for how we could maximize the function: start at a random input, and as many times as we can, take a small step in the direction of the gradient to move uphill. In other words, walk up the hill.
# To minimize the function, we can instead follow the negative of the gradient, and thus go in the direction of steepest descent. This is gradient descent. Formally, if we start at a point \[x_0\]  and move a positive distance \[\alpha\] in the direction of the negative gradient, then our new and improved  \[x_1\]  will look like this: \[x_1 = x_0 - \alpha \nabla f(x_0)\] More generally, we can write a formula for turning  \[x_n\] into \[x_{n + 1}\]:\[x_{n + 1} = x_n - \alpha \nabla f(x_n)\]
# """

# data = """
# A local minimum is the lowest point in a specific region or neighborhood of a function, while a global minimum is the absolute lowest point across the entire function's domain. 
# Here's a more detailed explanation:
# Local Minimum:
# A point where the function value is lower than all nearby points. 
# It's a "dip" or valley within a specific area of the function's graph. 
# A function can have multiple local minima. 
# A local minimum is also called a relative minimum. 
# Global Minimum:
# The lowest point of the entire function, or the absolute minimum value. 
# It's the lowest point across the entire domain of the function. 
# A function can only have one global minimum. 
# A global minimum is also called an absolute minimum. 
# Relationship:
# A global minimum is always a local minimum, but a local minimum is not necessarily a global minimum. 
# If a function is convex, every local minimum is also a global minimum. 
# Examples
# Imagine a landscape with multiple valleys. Each valley represents a local minimum, while the deepest valley represents the global minimum. 
# In machine learning, finding the global minimum means finding the best possible parameters for a model. 
# Optimization algorithms can get stuck in local minima, preventing them from finding the true global minimum. 
# """
# data = """

# Perpendicular Lines – Definition, Symbol, Properties, Examples
# Perpendicular lines are two lines that intersect, or meet, at a 90-degree angle (a right angle). 
# Here's a more detailed explanation:
# Definition: Perpendicular lines form a right angle where they intersect. 
# Right Angle: A right angle is exactly 90 degrees. 
# Visual Representation: A small square is often used to indicate a right angle, and therefore, perpendicular lines. 
# Examples:
# The sides of a square or rectangle are perpendicular. 
# The intersection of a wall and a floor. 
# The letter "T". 
# The intersection of roads at an intersection. 
# Opposite of Perpendicular: The opposite of perpendicular lines are parallel lines, which never intersect. 
# """
data = "Convexity plays a crucial role in many machine learning algorithms, especially in optimization problems. Convex optimization problems are easier to solve because they guarantee the existence of a unique global minimum, unlike non-convex problems which can have multiple local minima. Understanding convexity helps in designing more reliable and efficient machine learning models. "

pipe = Pipeline(data, "Gradient Descent")