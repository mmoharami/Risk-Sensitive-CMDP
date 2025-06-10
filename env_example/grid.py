import numpy as np

# Define dimensions
def grid_5x5():
    rows, cols = 5, 5
    H = rows + cols - 1 # goal state should be part of horizon

    # Reward: goal is high reward, some intermediate spots give misleading rewards
    r = np.array([
        [0.0, 0.1, 0.2, 0.2, 0.1],
        [0.5, 0.1, 1.5, 0.5, 0.3],
        [0.1, 0.1, 0.4, 0.3, 0.2],
        [0.1, 0.1, 0.3, 0.1, 0.6],
        [0.1, 0.2, 0.3, 0.1, 0.0]  # goal state
    ])

    # Utility: traps (high utility cost) are located around high-reward spots
    u = np.array([
        [0.1, 0.1, 0.2, 0.1, 0.1],
        [0.4, 0.2, 0.1, 0.0, 0.0],  
        [0.3, 0.4, 1.0, 0.0, 0.1],
        [0.2, 0.5, 0.4, 0.2, 0.1],
        [0.1, 0.1, 0.4, 0.2, 0.0] 
    ])

    # Probability of correct action: more reliable near start and goal
    p = np.array([
        [0.9, 0.9, 0.7, 0.5, 1.0],
        [0.9, 0.9, 0.5, 0.5, 1.0],
        [0.7, 0.9, 0.9, 0.6, 1.0],
        [0.9, 0.8, 0.8, 0.5, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]  # terminal row is deterministic
    ])

    # Return the parameters
    return r, u, p, H