from pipeline import Pipeline
from populate_pipeline import PopulatePipeline
from generate_synthetic_data import SyntheticDataGenerator

#topic 1 - Gradient Descent
g1= "Gradient descent is a method for unconstrained mathematical optimization. It is a first-order iterative algorithm for minimizing a differentiable multivariate function. The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent. Conversely, stepping in the direction of the gradient will lead to a trajectory that maximizes that function; the procedure is then known as gradient ascent. It is particularly useful in machine learning for minimizing the cost or loss function." #Wikipedia (compare and contrast present)
g2=r"""One thing to note, however, is that the techniques we used above can only be applied if we make some big assumptions. For the calculus approach, we assumed that the loss function was differentiable at all points and that we could algebraically solve for the zero points of the derivative; for the geometric approach, OLS *only* applies when using a linear model with MSE loss. What happens when we have more complex models with different, more complex loss functions? The techniques we've learned so far will not work, so we need a new optimization technique: **gradient descent**. Looking at the function across this domain, it is clear that the function's minimum value occurs around $\theta = 5.3$. Let's pretend for a moment that we *couldn't* see the full view of the cost function. How would we guess the value of $\theta$ that minimizes the function? Let's consider an arbitrary function. Our goal is to find the value of $x$ that minimizes this function.```def arbitrary(x): return (x**4 - 15*x**3 + 80*x**2 - 180*x + 144)/10 It turns out that the first derivative of the function can give us a clue. In the graph below, the function and its derivative are plotted, with points where the derivative is equal to 0 plotted in light green. > **BIG IDEA**: use an iterative algorithm to numerically compute the minimum of the loss.Looking at the function across this domain, it is clear that the function's minimum value occurs around $\theta = 5.3$. Let's pretend for a moment that we *couldn't* see the full view of the cost function. How would we guess the value of $\theta$ that minimizes the function?  It turns out that the first derivative of the function can give us a clue. In the graph below, the function and its derivative are plotted, with points where the derivative is equal to 0 plotted in light green.Say we make a guess for the minimizing value of $\theta$. Remember that we read plots from left to right, and assume that our starting $\theta$ value is to the left of the optimal $\hat{\theta}$. If the guess undershoots the true minimizing value – our guess for $\theta$ is lower than the value of the $\hat{\theta}$ that minimizes the function – the derivative will be **negative**. This means that if we increase $\theta$ (move further to the right), then we **can decrease** our loss function further. If this guess overshoots the true minimizing value, the derivative will be positive, implying the converse. We can use this pattern to help formulate our next guess for the optimal $\hat{\theta}$. Consider the case where we've undershot $\theta$ by guessing too low of a value. We'll want our next guess to be greater in value than our previous guess – that is, we want to shift our guess to the right. You can think of this as following the slope downhill to the function's minimum value.If we've overshot $\hat{\theta}$ by guessing too high of a value, we'll want our next guess to be lower in value – we want to shift our guess for $\hat{\theta}$ to the left.""" #Data 100 note (a lot of context + example function)
g3="Intuition for Gradient Descent Think of a large bowl like what you would eat cereal out of or store fruit in. This bowl is a plot of the cost function (f).A random position on the surface of the bowl is the cost of the current values of the coefficients (cost).The bottom of the bowl is the cost of the best set of coefficients, the minimum of the function. The goal is to continue to try different values for the coefficients, evaluate their cost and select new coefficients that have a slightly better (lower) costs. Repeating this process enough times will lead to the bottom of the bowl and you will know the values of the coefficients that result in the minimum cost." #Src: https://machinelearningmastery.com/gradient-descent-for-machine-learning/ (intuitive explanation)
g4="Gradient Descent Algorithm Gradient Descent Algorithm iteratively calculates the next point using gradient at the current position, scales it (by a learning rate) and subtracts obtained value from the current position (makes a step). It subtracts the value because we want to minimise the function (to maximise it would be adding). This process can be written as:p_{n+1} = p_n - η * ∇f(p_n) There’s an important parameter η which scales the gradient and thus controls the step size. In machine learning, it is called learning rate and have a strong influence on performance. The smaller learning rate the longer GD converges, or may reach maximum iteration before reaching the optimum point If learning rate is too big the algorithm may not converge to the optimal point (jump around) or even to diverge completely. In summary, Gradient Descent method’s steps are: 1-choose a starting point (initialisation), 2-calculate gradient at this point, 3-make a scaled step in the opposite direction to the gradient (objective: minimise), 4-repeat points 2 and 3 until one of the criteria is met: maximum number of iterations reached step size is smaller than the tolerance (due to scaling or a small gradient)." #Src: https://medium.com/data-science/gradient-descent-algorithm-a-deep-dive-cf04e8115f21 / step by step algorithm

#topic 2 - Convexity
c1="A function f : R n → R is convex if its domain is a convex set and for all x, y in its domain, and all λ ∈ [0, 1], we have f(λx + (1 − λ)y) ≤ λf(x) + (1 − λ)f(y).• In words, this means that if we take any two points x, y, then f evaluated at any convex combination of these two points should be no larger than the same convex combination of f(x) and f(y). Geometrically, the line segment connecting (x, f(x)) to (y, f(y)) must sit above the graph of f. If f is continuous, then to ensure convexity it is enough to check the definition with λ = 1/2 (or any other fixed λ ∈ (0, 1)). This is similar to the notion of midpoint convex sets that we saw earlier. We say that f is concave if −f is convex." #Src: https://www.princeton.edu/~aaa/Public/Teaching/ORF523/S16/ORF523_S16_Lec7_gh.pdf (recap, general description)
c2=r""" The most basic duality theorem is that for any closed convex set C, and any point x0 /∈ C, there exists a hyperplane (equivalently, a functional x∗ ∈ X∗) that separates x0 from C. This may seem fairly obvious in R2 by drawing a picture, but the result holds in arbitrary (in fact, infinite!) dimensions. The result is often referred to as the geometric Hahn–Banach theorem, of which there are many different but related variations; we state a few below. Theorem 3.5 (The Separation Theorem).Let C ⊂ X be a closed convex set, and x0 ∈ X \ C. There exists a nonzero x∗ ∈ X∗ and δ > 0 such that ⟨x∗, x0⟩ + δ ≤ ⟨x∗, x⟩, ∀x ∈ C. In other words, x∗ (strictly) separates x0 from C.""" #Src: EECS 127 Sp25 handout by T.Courtade (The Separation Theorem)
c3="A convex function is a continuous function whose value at the midpoint of every interval in its domain does not exceed the arithmetic mean of its values at the ends of the interval.If f(x) has a second derivative in [a,b], then a necessary and sufficient condition for it to be convex on that interval is that the second derivative f^('')(x)>=0 for all x in [a,b]." #Src: https://mathworld.wolfram.com/ConvexFunction.html (General Definiton + extra context)
c4="To simply things, think of convex sets as shapes where any line joining 2 points in this set is never outside the set. This is called a convex set.Consider the graph of a function f. An epigraph is a set of points lying on or above the function’s graph.A function f is said to be a convex function if its epigraph is a convex set. This means that every line segment drawn on this graph is always equal to or above the function graph. " #Src:https://towardsdatascience.com/understand-convexity-in-optimization-db87653bf920/ (Sequence of definitions)

#topic 3 - Least Squares
# l1=""
# l2=""
# l3=""
# l4=""

#topic 4 - Linear Regression
# r1=""
# r2=""
# r3=""
# r4=""

#topic 5 - Derivative at minimum/maximum
# d1=""
# d2=""
# d3=""
# d4=""

gen=SyntheticDataGenerator(4)

# data_dict={"Gradient Descent":[g1,g2,g3,g4],"Convexity":[c1,c2,c3,c4], "Least Squares": [l1,l2,l3,l4],"Linear Regression":[r1,r2,r3,r4], "Maxima and Minima": [d1,d2,d3,d4]}
data_dict={"Gradient Descent":[g1,g2,g3,g4],"Convexity":[c1,c2,c3,c4]}

for data_topic, data_list in data_dict.items():
    original=list(data_list)
    for text in original:
        paraphrases = gen._generate_data2(text)
        data_list.extend(paraphrases)
    PopulatePipeline(data_list, data_topic)
        

# pipe = Pipeline(data, "Gradient Descent")
# pipe = PopulatePipeline(data, "Gradient Descent")