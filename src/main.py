from pipeline import Pipeline
from populate_pipeline import PopulatePipeline
from generate_synthetic_data import SyntheticDataGenerator
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
# data = "Convexity plays a crucial role in many machine learning algorithms, especially in optimization problems. Convex optimization problems are easier to solve because they guarantee the existence of a unique global minimum, unlike non-convex problems which can have multiple local minima. Understanding convexity helps in designing more reliable and efficient machine learning models. "
# data ="Based on the amount of data the algorithm uses, there are three types of gradient descent: Batch Gradient Descent Batch gradient descent uses cyclic training epochs to calculate the error for each example within the training dataset. The training samples should be evaluated to determine if they update the model. The batch gradient descent is computationally efficient meaning it has a stable error gradient and a stable convergence. A drawback is that the stable error gradient can converge in a spot that isn’t the best the model can achieve. It also requires the whole training set to be loaded into the memory. Stochastic Gradient Descent Stochastic gradient descent (SGD) updates the parameters for each training example one by one. In some scenarios, SGD is faster than batch gradient descent. An advantage is that frequent updates provide a rather detailed rate of improvement. However, SGD is computationally more expensive than BGD. Also, the frequency of the updates can result in noisy gradients, which may cause the error rate to increase instead of slowly decreasing. Mini-Batch Gradient Descent Mini-batch gradient descent is a combination of the SGD and BGD algorithms. It divides the training dataset into small batches and updates each of these batches. This combines the efficiency of BGD and the robustness of SGD. Typical mini-batch sizes range around 100, but like other ML techniques, it varies for different applications. This is the preferred algorithm for training a neural network, and it’s the most common type of gradient descent in deep learning."

# data = "Show me an example of aribitary gradient descent, on a 2D graph"
data = ["Gradient descent is a method for unconstrained mathematical optimization. It is a first-order iterative algorithm for minimizing a differentiable multivariate function. The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent. Conversely, stepping in the direction of the gradient will lead to a trajectory that maximizes that function; the procedure is then known as gradient ascent. It is particularly useful in machine learning for minimizing the cost or loss function.", "The loss functions for linear models always produce a convex surface. As a result of this property, when a linear regression model converges, we know the model has found the weights and bias that produce the lowest loss.If we graph the loss surface for a model with one feature, we can see its convex shape. The following is the loss surface of the miles per gallon dataset used in the previous examples. Weight is on the x-axis, bias is on the y-axis, and loss is on the z-axis:"]
#wikipedia intro
# data ="Batch gradient descent sums the error for each point in a training set, updating the model only after all training examples have been evaluated. This process referred to as a training epoch."
# data = "Stochastic gradient descent (SGD) runs a training epoch for each example within the dataset and it updates each training example's parameters one at a time. Since you only need to hold one training example, they are easier to store in memory. While these frequent updates can offer more detail and speed, it can result in losses in computational efficiency when compared to batch gradient descent. Its frequent updates can result in noisy gradients, but this can also be helpful in escaping the local minimum and finding the global one."
# data = "Gradient descent is a mathematical technique that iteratively finds the weights and bias that produce the model with the lowest loss. Gradient descent finds the best weight and bias by repeating the following process for a number of user-defined iterations. The model begins training with randomized weights and biases near zero, and then repeats the following steps: Calculate the loss with the current weight and bias. Determine the direction to move the weights and bias that reduce loss. Move the weight and bias values a small amount in the direction that reduces loss. Return to step one and repeat the process until the model can't reduce the loss any further."
# data = "The loss functions for linear models always produce a convex surface. As a result of this property, when a linear regression model converges, we know the model has found the weights and bias that produce the lowest loss.If we graph the loss surface for a model with one feature, we can see its convex shape. The following is the loss surface of the miles per gallon dataset used in the previous examples. Weight is on the x-axis, bias is on the y-axis, and loss is on the z-axis:"




# pipe = Pipeline(data, "Gradient Descent")
pipe = PopulatePipeline(data, "Gradient Descent")