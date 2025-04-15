import numpy as np
import matplotlib.pyplot as plt

# Define the function and its gradient
def f(x, y):
    return (x ** 2 + y ** 2)

def gradient(x, y):
    return np.array([2 * x, 2 * y])

# Create a grid of points
X = np.linspace(-2, 2, 100)
Y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

# Create the figure and axis
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
plt.colorbar(contour)

# Define initial points and the learning rate
initial_points = [(-1.5, 1.5), (1.5, -1.5), (0.5, 1.0)]
learning_rate = 0.1
steps = 10

# Plot gradient descent paths
for start in initial_points:
    x, y = start
    path_x = [x]
    path_y = [y]
    
    for _ in range(steps):
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        path_x.append(x)
        path_y.append(y)

    plt.plot(path_x, path_y, marker='o', markersize=5, label=f'Start: {start}', linewidth=2)
    for i in range(len(path_x) - 1):
        plt.arrow(path_x[i], path_y[i], path_x[i+1] - path_x[i], path_y[i+1] - path_y[i], 
                  head_width=0.1, head_length=0.1, color='black', alpha=0.5)

# Mark the minimum point
plt.scatter(0, 0, color='red', s=100, label='Minimum Point (0,0)')

# Annotations and titles
plt.title('Gradient Descent Optimization on a 2D Convex Surface', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.legend()
plt.grid()
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.savefig("research_results/bello.png")