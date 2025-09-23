import numpy as np

# These define the transition probabilities between each seniority
transition_matrices = {
    0: np.array([
        [0.00, 0.00, 1.00],   # from New_job
        [0.10, 0.10, 0.80],   # from Promotion
        [0.15, 0.15, 0.70],   # from Stay
    ]),
    1: np.array([
        [0.00, 0.00, 1.00],
        [0.075, 0.075, 0.85],
        [0.125, 0.125, 0.75],
    ]),
    2: np.array([
        [0.00, 0.00, 1.00],
        [0.05, 0.05, 0.90],
        [0.10, 0.10, 0.80],
    ]),
    3: np.array([
        [0.00, 0.00, 1.00],
        [0.00, 0.00, 1.00],
        [0.00, 0.00, 1.00],
    ]),
}
