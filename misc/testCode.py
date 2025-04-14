import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return x**2
x = np.linspace(-2, 2, 400)
y = f(x)
x1, x2 = -1.5, 1.5
y1, y2 = f(x1), f(x2)
t = np.linspace(0, 1, 100)
x_line = x1 * (1 - t) + x2 * t
y_line = y1 * (1 - t) + y2 * t
plt.plot(x, y, label="Convex function $f(x) = x^2$")
plt.plot(x_line, y_line, 'r--', label="Chord between two points")
plt.scatter([x1, x2], [y1, y2], color='red')
plt.title("Demonstration of Convexity")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
