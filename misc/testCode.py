import matplotlib.pyplot as plt
import numpy as np

# Set figure dimensions and DPI
fig = plt.figure(figsize=(8, 6), dpi=100)

# Define the x and y coordinates for the two lines
x1 = [0, 1, 2, 3, 4]
y1 = [0, 1, 2, 3, 4]
x2 = [1, 2, 3, 4, 5]
y2 = [1, 2, 3, 4, 5]

# Define the colors for the two lines
color1 = '#3498db'
color2 = '#e74c3c'

# Create the plot
plt.plot(x1, y1, color=color1)
plt.plot(x2, y2, color=color2)

# Add the right angle label
plt.annotate('Right Angle', xy=(2, 2), xytext=(2.5, 2.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')

# Add the parallel lines
plt.plot([0, 5], [0, 5], color='#95a5a6', linestyle='--')

# Add the square diagram
plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], color='#95a5a6', linestyle='--')

# Add the key for the notation
plt.annotate('Key:', xy=(0.5, 0.5), xycoords='figure fraction',
             horizontalalignment='center', verticalalignment='center')
plt.annotate('Right Angle:', xy=(0.5, 0.6), xycoords='figure fraction',
             horizontalalignment='center', verticalalignment='center')
plt.annotate('Parallel Lines:', xy=(0.5, 0.7), xycoords='figure fraction',
             horizontalalignment='center', verticalalignment='center')

# Add the title and labels
plt.title('90-Degree Intersection', fontsize=16, fontweight='bold')
plt.xlabel('X-Axis', fontsize=12, fontweight='bold')
plt.ylabel('Y-Axis', fontsize=12, fontweight='bold')

# Set the font sizes and line weights
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

# Use tight layout to avoid element crowding
plt.tight_layout()

# Show the plot
plt.show()