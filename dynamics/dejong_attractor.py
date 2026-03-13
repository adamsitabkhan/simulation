import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import random

@njit
def generate_dejong(n_points, a, b, c, d):
    # Pre-allocate arrays for speed
    x = np.empty(n_points)
    y = np.empty(n_points)
    
    # Initial starting point
    x[0], y[0] = 0.1, 0.1
    
    for i in range(1, n_points):
        x[i] = np.sin(a * y[i-1]) - np.cos(b * x[i-1])
        y[i] = np.sin(c * x[i-1]) - np.cos(d * y[i-1])
        
    return x, y

# Parameters (Try changing these!)
n = 10_000_000
# a, b, c, d = -2.640, -1.902, 2.316, 2.525
a, b, c, d = random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3)
print(a, b, c, d)

# Generate the data
x_vals, y_vals = generate_dejong(n, a, b, c, d)

# Create a 2D histogram (the "image")
img, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=1000)

# Use a log scale for the colors so faint areas are visible
cmaps = ['magma']
plt.figure(figsize=(10, 10), facecolor='black')
plt.imshow(np.log1p(img), cmap=random.choice(cmaps), origin='lower')
plt.axis('off')
plt.show()