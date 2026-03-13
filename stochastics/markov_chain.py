import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import binom

def simulate_markov_chain(P, initial_distribution, n_steps, n_trajectories=1):
    """
    Simulate trajectories of a discrete-time Markov chain.

    Parameters:
    -----------
    P : numpy.ndarray
        Transition matrix of shape (n_states, n_states). 
        P[i, j] is the probability of transitioning from state i to state j.
    initial_distribution : numpy.ndarray or int
        Initial state distribution (array of probabilities) or a specific starting state (int).
    n_steps : int
        Number of time steps to simulate.
    n_trajectories : int
        Number of independent trajectories to simulate.

    Returns:
    --------
    trajectories : numpy.ndarray
        Array of shape (n_trajectories, n_steps + 1) containing the state history.
    """
    n_states = P.shape[0]
    
    # Check if P is a valid stochastic matrix
    assert np.allclose(P.sum(axis=1), 1), "Rows of transition matrix must sum to 1."
    
    trajectories = np.zeros((n_trajectories, n_steps + 1), dtype=int)
    
    # Initialize starting states
    if isinstance(initial_distribution, int):
        trajectories[:, 0] = initial_distribution
    elif isinstance(initial_distribution, str) and initial_distribution == 'uniform':
        trajectories[:, 0] = np.random.randint(0, n_states, size=n_trajectories)
    else:
        trajectories[:, 0] = np.random.choice(
            n_states, 
            size=n_trajectories, 
            p=initial_distribution
        )
    
    # Pre-compute cumulative probabilities for faster sampling
    P_cumsum = np.cumsum(P, axis=1)
    
    for t in range(n_steps):
        current_states = trajectories[:, t]
        
        # Vectorized sampling
        # Generate random numbers for all trajectories
        r = np.random.rand(n_trajectories)
        
        # Find the next state for each trajectory
        # We need P_cumsum[current_states] which gives the CDF for the transition 
        # from the current state of each trajectory
        thresholds = P_cumsum[current_states]
        
        # Determine next state by finding where r falls in the CDF
        # argmax with boolean gives the index of the first True
        next_states = (thresholds > r[:, None]).argmax(axis=1)
        
        trajectories[:, t + 1] = next_states
        
    return trajectories

def simulate_ehrenfest(M, initial_distribution, n_steps, n_trajectories=1):
    """
    Simulate trajectories of the Ehrenfest model.
    """
    n_states = M + 1
    
    trajectories = np.zeros((n_trajectories, n_steps + 1), dtype=int)
    
    # Initialize starting states
    if isinstance(initial_distribution, int):
        trajectories[:, 0] = initial_distribution
    elif isinstance(initial_distribution, str) and initial_distribution == 'uniform':
        trajectories[:, 0] = np.random.randint(0, n_states, size=n_trajectories)
    else:
        trajectories[:, 0] = np.random.choice(
            n_states, 
            size=n_trajectories, 
            p=initial_distribution
        )
    
    # Pre-compute transition probabilities
    P = get_ehrenfest_matrix(M)

    print("Transition Matrix P:")
    print(P)
    
    # Use the general simulation function
    return simulate_markov_chain(P, initial_distribution, n_steps, n_trajectories)

def simulate_wright_fisher(M, initial_distribution, n_steps, n_trajectories=1):
    """
    Simulate trajectories of the Wright-Fisher model for genetic drift.
    State i represents the number of copies of allele A in a population of size M (diploids -> 2N = M).
    """
    n_states = M + 1
    
    # Pre-compute transition probabilities
    P = get_wright_fisher_matrix(M)
    
    return simulate_markov_chain(P, initial_distribution, n_steps, n_trajectories)

def simulate_tsetlin(N, p1, p2, initial_distribution, n_steps, n_trajectories=1):
    """
    Simulate trajectories of a 2-Action Tsetlin Automaton.
    N states per action (Total 2N states).
    Action 1: States 0 to N-1.
    Action 2: States N to 2N-1.
    p1, p2: Penalty probabilities for Action 1 and Action 2.
    """
    P = get_tsetlin_matrix(N, p1, p2)
    return simulate_markov_chain(P, initial_distribution, n_steps, n_trajectories)

def get_ehrenfest_matrix(M):
    """
    Generate transition matrix for Ehrenfest model.
    """
    n_states = M + 1
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        P[i, i] = 0
        if i > 0:
            P[i, i - 1] = i / M
        if i < M:
            P[i, i + 1] = (M - i) / M
    return P

def get_wright_fisher_matrix(M):
    """
    Generate transition matrix for Wright-Fisher model.
    """
    n_states = M + 1
    P = np.zeros((n_states, n_states))
    # P[i, j] = P(X_{t+1} = j | X_t = i) = Binom(M, i/M) evaluated at j
    for i in range(n_states):
        p = i / M
        # Vectorized version for a whole row:
        P[i, :] = binom.pmf(np.arange(n_states), M, p)
    return P

def get_tsetlin_matrix(N, p1, p2):
    """
    Generate transition matrix for Tsetlin Automaton with 2N states.
    """
    n_states = 2 * N
    P = np.zeros((n_states, n_states))
    
    # Action 1: States 0 to N-1
    for i in range(N):
        # Reward (prob 1-p1): Move towards 0
        target_reward = max(0, i - 1)
        P[i, target_reward] += (1 - p1)
        
        # Penalty (prob p1): Move towards N
        target_penalty = i + 1
        P[i, target_penalty] += p1
        
    # Action 2: States N to 2N-1
    for i in range(N, n_states):
        # Reward (prob 1-p2): Move towards 2N-1
        target_reward = min(n_states - 1, i + 1)
        P[i, target_reward] += (1 - p2)
        
        # Penalty (prob p2): Move towards N-1
        target_penalty = i - 1
        P[i, target_penalty] += p2
        
    return P

def compute_distribution_evolution(P, initial_distribution, n_steps):
    """
    Compute the evolution of the probability distribution over time.

    Parameters:
    -----------
    P : numpy.ndarray
        Transition matrix.
    initial_distribution : numpy.ndarray or int or str
        Initial state distribution.
    n_steps : int
        Number of time steps.

    Returns:
    --------
    distributions : numpy.ndarray
        Array of shape (n_steps + 1, n_states) where row t is the distribution at time t.
    """
    n_states = P.shape[0]
    distributions = np.zeros((n_steps + 1, n_states))

    # set initial distribution
    if isinstance(initial_distribution, int):
        distributions[0, initial_distribution] = 1.0
    elif isinstance(initial_distribution, str) and initial_distribution == 'uniform':
        distributions[0, :] = 1.0 / n_states
    else:
        distributions[0, :] = initial_distribution

    # evolve distribution
    for t in range(n_steps):
        distributions[t + 1] = distributions[t] @ P

    return distributions

def visualize_distribution_evolution(distributions, state_labels=None):
    """
    Visualize the evolution of the probability distribution as a heatmap.
    """
    n_steps, n_states = distributions.shape
    # We want time on x-axis, states on y-axis
    # distributions shape is (time, states), so we transpose for plotting (states, time)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(distributions.T, aspect='auto', origin='lower', cmap='plasma', interpolation='nearest')
    
    plt.colorbar(label='Probability')
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.title('Evolution of Probability Distribution')

    # thin y-ticks if too many states
    TARGET_NTICKS = 20
    all_states = np.arange(n_states)
    if n_states > TARGET_NTICKS:
        step = n_states // TARGET_NTICKS
        y_ticks = all_states[::step]
    else:
        y_ticks = all_states
    
    plt.yticks(y_ticks)
    
    if state_labels:
         # Create labels for displayed ticks
        labels = [state_labels.get(s, str(s)) for s in y_ticks]
        plt.yticks(y_ticks, labels)

    plt.tight_layout()
    plt.show()

def visualize_trajectories(trajectories, n_states=None, state_labels=None):
    """
    Plot the trajectories of the Markov chain.
    
    Parameters:
    -----------
    trajectories : numpy.ndarray
        Array of shape (n_trajectories, n_steps + 1).
    n_states : int, optional
        Total number of states in the system. If None, inferred from data.
    state_labels : dict, optional
        Dictionary mapping state indices to labels.
    """
    n_trajectories, n_steps = trajectories.shape
    time = np.arange(n_steps)
    
    plt.figure(figsize=(12, 6))
    
    # Add slight jitter to y-values to visualize overlapping paths
    jitter_strength = 0.05
    
    for i in range(n_trajectories):
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=n_steps)
        # Removed marker='o' and reduced linewidth
        plt.plot(time, trajectories[i] + jitter, alpha=0.6, linewidth=0.8)
        
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.title(f'Discrete-Time Markov Chain Simulation\n({n_trajectories} Trajectories)')
    
    # Determine state space to display
    if n_states is None:
        if state_labels:
             # Use the max key if labels are provided, assuming 0-indexed
             n_states = max(state_labels.keys()) + 1
        else:
             n_states = np.max(trajectories) + 1
             
    all_states = np.arange(n_states)
    
    # Subsample y-ticks if too many states
    TARGET_NTICKS = 20
    if n_states > TARGET_NTICKS:
        step = n_states // TARGET_NTICKS
        y_ticks = all_states[::step]
    else:
        y_ticks = all_states
        
    plt.yticks(y_ticks)
    plt.ylim(-0.5, n_states - 0.5)
    
    if state_labels:
        # Create labels for displayed ticks
        labels = [state_labels.get(s, str(s)) for s in y_ticks]
        plt.yticks(y_ticks, labels)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def compute_stationary_distribution(P):
    """
    Compute the stationary distribution of the Markov chain using eigendecomposition.
    Stat dist pi satisfies pi @ P = pi.
    """
    eigenvals, eigenvecs = np.linalg.eig(P.T)
    # Find the eigenvector corresponding to eigenvalue 1 (closest to 1)
    idx = np.argmin(np.abs(eigenvals - 1))
    pi = np.real(eigenvecs[:, idx])
    return pi / pi.sum()

if __name__ == "__main__":
    # Simulation parameters
    n_steps = 50
    n_sims = 1000
    initial_state = 0  # Start with Sunny
    
    print(f"\nSimulating {n_sims} trajectories for {n_steps} steps...")
    
    # Run simulation
    # data = simulate_markov_chain(P, initial_state, n_steps, n_sims)
    # Increase M to demonstrate label thinning (e.g., M=100 -> ~101 states)
    M = 50
    n_states_total = M + 1
    
    # Example 1: Simulate Trajectories (Monte Carlo)
    print(f"\nExample 1: Simulating {n_sims} trajectories (Tsetlin Automaton) for {n_steps} steps...")
    
    # Tsetlin parameters
    N = 25  # States per action (Total 50)
    p1 = 0.2 # Penalty prob for Action 1 (Good action)
    p2 = 0.4 # Penalty prob for Action 2 (Bad action)
    M = 2 * N
    
    # Get matrix
    P = get_tsetlin_matrix(N, p1, p2)
    
    # Simulate
    # Start uniform or at the center (N-1 or N)
    data = simulate_markov_chain(P, initial_distribution='uniform', n_steps=n_steps, n_trajectories=n_sims)
    
    # Calculate empirical distribution from last step
    final_states = data[:, -1]
    unique, counts = np.unique(final_states, return_counts=True)
    empirical_dist = counts / n_sims
        
    print("\nEmpirical Distribution at final step (small sample size):")
    for u, p in zip(unique, empirical_dist):
        print(f"State {u}: {p:.4f}")
        
    # Visualize Trajectories
    print("\nDisplaying trajectory plot...")
    
    # Custom labels for Tsetlin
    state_labels = {i: f"A1_{N-i}" for i in range(N)}
    state_labels.update({i: f"A2_{i-N+1}" for i in range(N, 2*N)})
    
    visualize_trajectories(data, n_states=M, state_labels=state_labels)

    # Example 2: Distribution Evolution (Exact Calculation)
    print("\nExample 2: Visualizing Distribution Evolution (Exact Calculation)...")
    
    # Start from Uniform Distribution across all states
    dist_evolution = compute_distribution_evolution(P, initial_distribution='uniform', n_steps=n_steps)

    visualize_distribution_evolution(dist_evolution, state_labels=state_labels)
