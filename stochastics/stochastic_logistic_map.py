import numpy as np
import matplotlib.pyplot as plt

def logistic_map(a, x, noise_level=0):
    noise = np.random.uniform(-noise_level, noise_level)
    next_x = a * x * (1 - x) + noise
    return np.clip(next_x, 0, 1)  # Keep x within [0, 1]

def simulate_bifurcation(min_a=2.5, max_a=4.0, n_a=1000, n_steps=1000, noise=0):
    last_n = 100
    a_values = np.linspace(min_a, max_a, n_a)
    x_results = []
    a_results = []

    for a in a_values:
        x = 0.5
        for i in range(n_steps):
            x = logistic_map(a, x, noise_level=noise)
            if i >= (n_steps - last_n):
                x_results.append(x)
                a_results.append(a)
    
    return a_results, x_results

min_a=2.5
max_a=4.0
noise=0.05

# Generate data
a_clean, x_clean = simulate_bifurcation(min_a=min_a, max_a=max_a, noise=0)
a_noisy, x_noisy = simulate_bifurcation(min_a=min_a, max_a=max_a, noise=noise)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

ax1.scatter(a_clean, x_clean, s=0.1, color='black', alpha=0.2)
ax1.set_title("Deterministic Logistic Map")
ax1.set_xlabel("Growth Rate (a)")
ax1.set_ylabel("Population (x)")

ax2.scatter(a_noisy, x_noisy, s=0.1, color='crimson', alpha=0.2)
ax2.set_title(f"Stochastic Logistic Map (Noise = {noise})")
ax2.set_xlabel("Growth Rate (a)")

plt.tight_layout()
plt.show()